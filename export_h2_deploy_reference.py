#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export gear_sonic_deploy reference bundles from H2 retarget CSVs.

Input:
  A directory tree containing per-motion H2 retarget CSVs (root tx, root rpy, 31 DOF).

Output (per motion):
  <output>/<motion_name>/
    metadata.txt
    info.txt
    joint_pos.csv
    joint_vel.csv
    body_pos.csv
    body_quat.csv
    body_lin_vel.csv
    body_ang_vel.csv

Notes:
  - body_pos/body_quat come from Newton FK (state.body_q)
  - velocities are finite differences using dt = 1/fps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Heavy deps (numpy/warp/newton) are imported lazily in main()
# so `--help` works even if the environment isn't set up.
np = None  # type: ignore
wp = None  # type: ignore
newton = None  # type: ignore
csv_utils = None  # type: ignore
pipeline_utils = None  # type: ignore
newton_utils = None  # type: ignore


def _finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """Forward difference with first frame copied from first delta.

    x: (T, D)
    returns: (T, D)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    T = x.shape[0]
    if T == 0:
        return x.copy()
    if T == 1:
        return np.zeros_like(x)

    v = np.empty_like(x)
    v[1:] = (x[1:] - x[:-1]) / dt
    v[0] = v[1]
    return v


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    # q: (..., 4) xyzw
    out = q.copy()
    out[..., 0:3] *= -1.0
    return out


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a,b: (..., 4) xyzw
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def _quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, eps)


def _quat_to_ang_vel(q: np.ndarray, dt: float) -> np.ndarray:
    """Approximate world angular velocity from quaternion trajectory.

    q: (T, B, 4) xyzw, assumed world orientation.
    returns: (T, B, 3)

    Uses relative rotation dq = q[t+1] * conj(q[t]) and maps to axis-angle.
    Angular velocity is axis * angle / dt.
    """
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError(f"Expected (T,B,4), got shape={q.shape}")

    T, B, _ = q.shape
    if T == 0:
        return np.zeros((0, B, 3), dtype=q.dtype)
    if T == 1:
        return np.zeros((1, B, 3), dtype=q.dtype)

    qn = _quat_normalize(q)
    dq = _quat_multiply(qn[1:], _quat_conjugate(qn[:-1]))
    dq = _quat_normalize(dq)

    # Ensure shortest path (avoid jumps at w < 0)
    flip = (dq[..., 3] < 0.0)[..., None]
    dq = np.where(flip, -dq, dq)

    # axis-angle from quaternion
    xyz = dq[..., 0:3]
    w = np.clip(dq[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))

    axis = np.zeros_like(xyz)
    small = sin_half < 1e-8
    axis[~small] = xyz[~small] / sin_half[~small][..., None]
    # For very small angles, axis is arbitrary; xyz ~= axis*sin(theta/2)
    axis[small] = np.where(sin_half[small][..., None] > 0, xyz[small] / sin_half[small][..., None], 0.0)

    w_body = axis * (angle[..., None] / dt)

    out = np.empty((T, B, 3), dtype=q.dtype)
    out[1:] = w_body
    out[0] = out[1]
    return out


def _write_csv(path: Path, data: np.ndarray, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        str(path),
        data,
        delimiter=",",
        fmt="%.9g",
        header=",".join(headers),
        comments="",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _discover_csvs(input_root: Path) -> list[Path]:
    return sorted([p for p in input_root.rglob("*.csv") if p.is_file()])


def _build_h2_fk_model() -> tuple[newton.Model, newton.State, list[str], int]:
    robot_builder = newton.ModelBuilder()
    robot_builder.add_mjcf(pipeline_utils.get_robot_mjcf_path(pipeline_utils.TargetType.H2))

    builder = newton.ModelBuilder()
    builder.add_builder(robot_builder, wp.transform_identity())
    model = builder.finalize()
    state = model.state()

    body_names = [newton_utils.get_name_from_label(label) for label in robot_builder.body_label]
    return model, state, body_names, robot_builder.body_count


def export_motion(
    *,
    model: newton.Model,
    state: newton.State,
    body_names: list[str],
    num_bodies: int,
    csv_path: Path,
    motion_out_dir: Path,
    fps_src: float,
    fps_dst: float,
) -> None:
    if fps_src <= 0:
        raise ValueError(f"fps_src must be > 0, got {fps_src}")
    if fps_dst <= 0:
        raise ValueError(f"fps_dst must be > 0, got {fps_dst}")
    dt_dst = 1.0 / fps_dst

    # Treat input frame index as sampled at fps_src.
    buffer = csv_utils.load_csv(str(csv_path), fps=fps_src, csv_config=csv_utils.get_csv_config("h2"))
    T = buffer.num_frames
    if T <= 0:
        raise RuntimeError(f"Empty CSV animation buffer: {csv_path}")

    # Extract joint-level data (31 DOF)
    first = buffer.get_data(0)
    dof_dim = int(first.shape[0] - 7)
    joint_pos = np.zeros((T, dof_dim), dtype=np.float32)
    root_q = np.zeros((T, 7 + dof_dim), dtype=np.float32)  # for FK upload
    for t in range(T):
        row = buffer.get_data(t).astype(np.float32, copy=False)
        root_q[t, :] = row
        joint_pos[t, :] = row[7:]

    # Optional resampling to match deploy's 50 Hz control tick.
    if fps_dst != fps_src and T > 1:
        t_src = np.arange(T, dtype=np.float64) / float(fps_src)
        duration = float(t_src[-1])
        T_dst = int(round(duration * float(fps_dst))) + 1
        t_dst = np.arange(T_dst, dtype=np.float64) / float(fps_dst)

        # Linear interpolate translation + joints.
        root_pos_src = root_q[:, 0:3].astype(np.float64)
        root_pos_dst = np.zeros((T_dst, 3), dtype=np.float32)
        for d in range(3):
            root_pos_dst[:, d] = np.interp(t_dst, t_src, root_pos_src[:, d]).astype(np.float32)

        joint_pos_src = joint_pos.astype(np.float64)
        joint_pos_dst = np.zeros((T_dst, dof_dim), dtype=np.float32)
        for j in range(dof_dim):
            joint_pos_dst[:, j] = np.interp(t_dst, t_src, joint_pos_src[:, j]).astype(np.float32)

        # Slerp quaternions (xyzw).
        quat_src = _quat_normalize(root_q[:, 3:7].astype(np.float64))
        quat_dst = np.zeros((T_dst, 4), dtype=np.float64)
        for i, t in enumerate(t_dst):
            if t <= 0.0:
                quat_dst[i] = quat_src[0]
                continue
            if t >= duration:
                quat_dst[i] = quat_src[-1]
                continue
            u = t * float(fps_src)
            k0 = int(np.floor(u))
            k1 = min(k0 + 1, T - 1)
            a = float(u - k0)
            q0 = quat_src[k0]
            q1 = quat_src[k1]
            if float(np.dot(q0, q1)) < 0.0:
                q1 = -q1
            dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
            if dot > 0.9995:
                q = q0 + a * (q1 - q0)
                quat_dst[i] = q / max(float(np.linalg.norm(q)), 1e-12)
            else:
                theta_0 = math.acos(dot)
                sin_0 = math.sin(theta_0)
                theta = theta_0 * a
                s0 = math.sin(theta_0 - theta) / sin_0
                s1 = math.sin(theta) / sin_0
                quat_dst[i] = s0 * q0 + s1 * q1
        quat_dst = _quat_normalize(quat_dst).astype(np.float32)

        root_q = np.concatenate([root_pos_dst, quat_dst, joint_pos_dst], axis=1).astype(np.float32)
        joint_pos = joint_pos_dst
        T = int(root_q.shape[0])

    joint_vel = _finite_difference(joint_pos, dt_dst)

    # FK for body-level trajectories
    # state.body_q layout: (num_bodies, 7) per articulation (we use one).
    body_pos = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_quat_xyzw = np.zeros((T, num_bodies, 4), dtype=np.float32)

    for t in range(T):
        q = root_q[t]
        wp.copy(model.joint_q, wp.array(q, dtype=wp.float32), 0, 0, int(q.shape[0]))
        # joint_qd is not required for FK; keep zeros.
        newton.eval_fk(model, model.joint_q, model.joint_qd, state, None)
        bq = state.body_q.numpy().reshape(num_bodies, 7)
        body_pos[t, :, :] = bq[:, 0:3]
        # Newton body_q stores xyzw.
        body_quat_xyzw[t, :, :] = bq[:, 3:7]

    body_lin_vel = _finite_difference(body_pos.reshape(T, -1), dt_dst).reshape(T, num_bodies, 3)
    body_ang_vel = _quat_to_ang_vel(body_quat_xyzw, dt_dst)

    # Flatten body arrays to T x (B*D)
    body_pos_flat = body_pos.reshape(T, num_bodies * 3)

    # Export quaternion as wxyz (common in deploy stacks)
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]].reshape(T, num_bodies * 4)
    body_lin_vel_flat = body_lin_vel.reshape(T, num_bodies * 3)
    body_ang_vel_flat = body_ang_vel.reshape(T, num_bodies * 3)

    _write_csv(
    motion_out_dir / "joint_pos.csv",
    joint_pos,
    [f"joint_{i}" for i in range(dof_dim)],
    )

    _write_csv(
        motion_out_dir / "joint_vel.csv",
        joint_vel,
        [f"joint_vel_{i}" for i in range(dof_dim)],
    )

    _write_csv(
        motion_out_dir / "body_pos.csv",
        body_pos_flat,
        [f"body_{i//3}_{'xyz'[i%3]}" for i in range(num_bodies * 3)],
    )

    _write_csv(
        motion_out_dir / "body_quat.csv",
        body_quat_wxyz,
        [f"body_{i//4}_{'wxyz'[i%4]}" for i in range(num_bodies * 4)],
    )

    _write_csv(
        motion_out_dir / "body_lin_vel.csv",
        body_lin_vel_flat,
        [f"body_{i//3}_vel_{'xyz'[i%3]}" for i in range(num_bodies * 3)],
    )

    _write_csv(
        motion_out_dir / "body_ang_vel.csv",
        body_ang_vel_flat,
        [f"body_{i//3}_angvel_{'xyz'[i%3]}" for i in range(num_bodies * 3)],
    )

    body_indexes = list(range(num_bodies))

    metadata = (
        f"Metadata for: {csv_path.stem}\n"
        + "=" * 30
        + "\n\n"
        + "Body part indexes:\n"
        + "[" + " ".join(str(i) for i in body_indexes) + "]"
        + "\n\n"
        + f"Total timesteps: {T}\n\n"
        + f"FPS: {fps_dst}\n"
        + f"dt: {dt_dst}\n"
        + f"DOF: {dof_dim}\n"
        + f"Body parts: {num_bodies}\n\n"
        + "Body names:\n"
        + ",".join(body_names)
        + "\n\n"
        + "Data arrays summary:\n"
        + f"  joint_pos: ({T}, {dof_dim}) (float32)\n"
        + f"  joint_vel: ({T}, {dof_dim}) (float32)\n"
        + f"  body_pos_w: ({T}, {num_bodies}, 3) (float32)\n"
        + f"  body_quat_w: ({T}, {num_bodies}, 4) (float32)\n"
        + f"  body_lin_vel_w: ({T}, {num_bodies}, 3) (float32)\n"
        + f"  body_ang_vel_w: ({T}, {num_bodies}, 3) (float32)\n"
    )

    _write_text(motion_out_dir / "metadata.txt", metadata)

    info = "\n".join(
        [
            f"joint_pos shape: {joint_pos.shape} (T x DOF)",
            f"joint_vel shape: {joint_vel.shape} (T x DOF)",
            f"body_pos shape: {body_pos_flat.shape} (T x (B*3))",
            f"body_quat shape: {body_quat_wxyz.shape} (T x (B*4)), wxyz",
            f"body_lin_vel shape: {body_lin_vel_flat.shape} (T x (B*3))",
            f"body_ang_vel shape: {body_ang_vel_flat.shape} (T x (B*3))",
            f"joint_pos range: [{float(np.min(joint_pos)):.6g}, {float(np.max(joint_pos)):.6g}]",
            f"body_pos range: [{float(np.min(body_pos_flat)):.6g}, {float(np.max(body_pos_flat)):.6g}]",
        ]
    ) + "\n"
    _write_text(motion_out_dir / "info.txt", info)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export H2 deploy reference bundles from retarget CSVs")
    parser.add_argument("--input", required=True, type=str, help="Input directory containing H2 retarget CSVs")
    parser.add_argument("--output", required=True, type=str, help="Output directory for deploy reference bundles")
    parser.add_argument("--fps", default=30.0, type=float, help="Source FPS of input CSVs")
    parser.add_argument("--dst-fps", default=50.0, type=float, help="Export FPS for deploy bundles (default 50)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing motion export folders")
    parser.add_argument(
        "--preserve-subdirs",
        action="store_true",
        help="Preserve input subdirectory structure under output (motion_name may be nested).",
    )
    args = parser.parse_args()

    # Lazy imports
    global np, wp, newton, csv_utils, pipeline_utils, newton_utils
    import numpy as np  # type: ignore
    import warp as wp  # type: ignore
    import newton  # type: ignore
    import soma_retargeter.assets.csv as csv_utils  # type: ignore
    import soma_retargeter.pipelines.utils as pipeline_utils  # type: ignore
    import soma_retargeter.utils.newton_utils as newton_utils  # type: ignore

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    csv_paths = _discover_csvs(input_root)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {input_root}")

    model, state, body_names, num_bodies = _build_h2_fk_model()

    exported = 0
    skipped_existing = 0
    for csv_path in csv_paths:
        rel = csv_path.relative_to(input_root)
        if args.preserve_subdirs:
            motion_name = rel.with_suffix("")
        else:
            motion_name = Path(rel.stem)

        motion_out_dir = output_root / motion_name
        if not args.force:
            # Heuristic: if already exported, skip.
            if (motion_out_dir / "metadata.txt").is_file() and (motion_out_dir / "body_pos.csv").is_file():
                skipped_existing += 1
                continue
        else:
            if motion_out_dir.exists():
                import shutil
                shutil.rmtree(motion_out_dir)

        export_motion(
            model=model,
            state=state,
            body_names=body_names,
            num_bodies=num_bodies,
            csv_path=csv_path,
            motion_out_dir=motion_out_dir,
            fps_src=float(args.fps),
            fps_dst=float(args.dst_fps),
        )
        exported += 1

    print(
        f"[INFO]: Export complete. input_csvs={len(csv_paths)} exported={exported} skipped_existing={skipped_existing} output={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
