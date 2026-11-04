"""Versioned dual-GPU SOMA->H2 retarget runner for a full paired catalog.

The runner never overwrites an existing CSV.  It indexes BVH and SMPL by
basename, keeps only unambiguous pairs, splits the deterministic selection over
two GPUs, and publishes each CSV atomically after schema/frame validation.

`--batch-size-total 512` means 256 motions per GPU when two GPUs are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


ROOT_HEADER = [
    "Frame", "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def index(root: Path, suffix: str) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = {}
    for path in root.rglob(f"*{suffix}"):
        grouped.setdefault(path.stem, []).append(path.resolve())
    unique, ambiguous = {}, {}
    for key, paths in grouped.items():
        if len(paths) == 1:
            unique[key] = paths[0]
        else:
            ambiguous[key] = [str(p) for p in sorted(paths)]
    return unique, ambiguous


def bvh_frames(path: Path) -> int:
    with path.open() as f:
        for line in f:
            if line.strip().startswith("Frames:"):
                return int(line.split(":", 1)[1])
    raise ValueError(f"Missing Frames header: {path}")


def validate_csv(path: Path, frames: int, joint_names: list[str]) -> None:
    with path.open(newline="") as f:
        rows = csv.reader(f)
        header = next(rows)
        if header[:7] != ROOT_HEADER or len(header) != 38 or set(header[7:]) != set(joint_names):
            raise ValueError(f"Invalid header: {path}")
        count = 0
        for count, row in enumerate(rows, 1):
            if len(row) != 38:
                raise ValueError(f"Invalid columns row {count}: {path}")
            values = [float(v) for v in row]
            if not all(math.isfinite(v) for v in values) or values[0] != count - 1:
                raise ValueError(f"Invalid values row {count}: {path}")
        if count != frames:
            raise ValueError(f"Frame mismatch {count} != {frames}: {path}")


def atomic_csv(save_csv, path: Path, names: list[str], buffer) -> None:
    if path.exists():
        validate_csv(path, buffer.num_frames, names)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        save_csv(str(tmp), names, buffer)
        validate_csv(tmp, buffer.num_frames, names)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(args) -> None:
    if args.out.exists():
        raise FileExistsError(f"Use a new output directory: {args.out}")
    bvh, amb_bvh = index(args.bvh_root, ".bvh")
    smpl, amb_smpl = index(args.smpl_root, ".pkl")
    keys = sorted(set(bvh) & set(smpl))
    if not keys:
        raise ValueError("No unambiguous BVH/SMPL pairs")
    args.out.mkdir(parents=True)
    selection = {
        "schema": "h2_full_paired_retarget_selection.v1",
        "motions": [
            {
                "key": key,
                "bvh": str(bvh[key].relative_to(args.bvh_root.resolve())),
                "bvh_sha256": sha256(bvh[key]),
                "smpl": str(smpl[key]),
                "smpl_sha256": sha256(smpl[key]),
            }
            for key in keys
        ],
        "counts": {
            "bvh_unique": len(bvh), "smpl_unique": len(smpl), "paired": len(keys),
            "bvh_ambiguous": len(amb_bvh), "smpl_ambiguous": len(amb_smpl),
        },
        "ambiguous": {"bvh": amb_bvh, "smpl": amb_smpl},
    }
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection["counts"], indent=2), flush=True)


def worker(args) -> None:
    # Imports happen after CUDA_VISIBLE_DEVICES is fixed by the parent launcher.
    import numpy as np
    import warp as wp
    import newton
    from soma_retargeter.io import bvh, csv as csv_io
    from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
    from soma_retargeter.utils import newton_utils
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

    selection = json.loads(args.selection.read_text())
    all_items = selection["motions"]
    items = all_items[args.shard_id :: args.num_shards]
    cfg = json.loads(args.retarget_config.read_text())
    first = args.bvh_root / items[0]["bvh"]

    with wp.ScopedDevice("cuda:0"):
        skeleton, _ = bvh.BVHImporter().create_skeleton(first)
        pipeline = SomaRetargetingPipeline(skeleton, "soma", "unitree_h2", retarget_config=cfg)
        names = newton_utils.get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
        offset = SpaceConverter(get_facing_direction_type_from_str("Mujoco")).transform(wp.transform_identity())
        completed = 0
        start = time.time()
        for base in range(0, len(items), args.batch_size):
            batch = items[base : base + args.batch_size]
            pending = []
            for item in batch:
                src = args.bvh_root / item["bvh"]
                dst = args.csv_root / Path(item["bvh"]).with_suffix(".csv")
                if dst.exists():
                    validate_csv(dst, bvh_frames(src), names)
                    completed += 1
                else:
                    pending.append((item, src, dst))
            if pending:
                pipeline.clear()
                animations = [bvh.load_bvh(src, skeleton)[1] for _, src, _ in pending]
                pipeline.add_input_motions(animations, [offset] * len(animations), True)
                buffers = pipeline.execute()
                if len(buffers) != len(pending):
                    raise ValueError("Retarget output batch size mismatch")
                for (_, src, dst), buffer in zip(pending, buffers, strict=True):
                    atomic_csv(csv_io.save_csv, dst, names, buffer)
                    validate_csv(dst, bvh_frames(src), names)
                    completed += 1
            print(json.dumps({
                "shard": args.shard_id, "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "completed": completed, "total": len(items), "seconds": round(time.time() - start, 1),
            }), flush=True)
    print(f"SHARD_COMPLETE {args.shard_id} {completed}/{len(items)}", flush=True)


def launch(args) -> None:
    import subprocess
    selection = json.loads(args.selection.read_text())
    if not selection["motions"]:
        raise ValueError("Empty selection")
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) < 1 or any(not x.isdecimal() for x in gpus):
        raise ValueError("--gpus must be comma-separated numeric GPU ids")
    if args.batch_size_total % len(gpus):
        raise ValueError("Total batch must divide evenly across GPUs")
    per_gpu = args.batch_size_total // len(gpus)
    args.csv_root.mkdir(parents=True, exist_ok=True)
    logs = args.out / "logs"
    logs.mkdir(exist_ok=True)
    children = []
    for shard, gpu in enumerate(gpus):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        log = (logs / f"retarget_gpu{gpu}.log").open("a")
        cmd = [
            os.environ.get("PYTHON", os.sys.executable), str(Path(__file__).resolve()), "worker",
            "--selection", str(args.selection), "--bvh-root", str(args.bvh_root),
            "--csv-root", str(args.csv_root), "--retarget-config", str(args.retarget_config),
            "--shard-id", str(shard), "--num-shards", str(len(gpus)), "--batch-size", str(per_gpu),
        ]
        children.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, gpu))
    failed = []
    for proc, log, gpu in children:
        rc = proc.wait()
        log.close()
        if rc:
            failed.append((gpu, rc))
    if failed:
        raise RuntimeError(f"Retarget shard failure: {failed}")
    print(f"COMPLETE paired={len(selection['motions'])} total_batch={args.batch_size_total} gpus={gpus}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "launch", "worker"])
    p.add_argument("--out", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--bvh-root", type=Path, required=True)
    p.add_argument("--smpl-root", type=Path)
    p.add_argument("--csv-root", type=Path)
    p.add_argument("--retarget-config", type=Path)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--batch-size-total", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256, help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    a = p.parse_args()
    for name in ("out", "selection", "bvh_root", "smpl_root", "csv_root", "retarget_config"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.resolve())
    if a.mode == "prepare":
        if a.out is None or a.smpl_root is None:
            p.error("prepare requires --out and --smpl-root")
        prepare(a)
    elif a.mode == "launch":
        if None in (a.out, a.selection, a.csv_root, a.retarget_config):
            p.error("launch requires --out --selection --csv-root --retarget-config")
        launch(a)
    else:
        if None in (a.selection, a.csv_root, a.retarget_config):
            p.error("worker requires --selection --csv-root --retarget-config")
        worker(a)


if __name__ == "__main__":
    main()
