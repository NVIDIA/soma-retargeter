# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IK weight optimization for human-to-robot motion retargeting.

This optimizer tunes the per-effector IK weights in a target retargeter config.
The human-to-robot scaler offsets stay fixed; the output is a retargeter config
with optimized IK weights.

A robot candidate is evaluated by retargeting the input motions, running FK on the
solved IK joint motion (``reference_q``), and comparing the resulting effector
poses against its human-to-robot scaler targets. The line search accepts on
solved-IK tracking loss plus optional smoothness and clip-regression penalties.
The next update is driven by per-effector solved-IK tracking errors, with
optional reachability decay and hard-clip emphasis.

The clip policy has three active modes. Gate mode is strict and rejects steps
that make individual clips worse beyond configured limits. Penalty mode is softer:
it adds those clip regressions to the objective, which is useful for low-DOF
robots that need bounded tradeoffs. Auto mode starts in gate mode and switches to
penalty mode if strict gating stalls. All active modes still reject severe spikes;
off mode disables the clip policy entirely.

The setup is inspired by Disney Research's ReActor work, but it has diverged from
ReActor's bilevel formulation. ReActor optimizes correspondence offsets and a
per-motion vertical offset while keeping IK weights fixed; this optimizer keeps
the scaler offsets fixed and optimizes the IK weights instead.

This file is self-contained: config loading, retargeter config generation, error
model construction, batched retarget/FK evaluation, metric aggregation, dashboard
rendering, and output artifact writing all live here.

Optimization loop:
1. Build the per-effector tracking/error model from the target robot geometry.
2. Retarget every motion with the current IK weights.
3. Score the solved robot motion against the scaler targets.
4. Use the tracking errors to propose one global IK-weight update.
5. Try that update at smaller step sizes until the objective and clip policy accept it.
6. Write the optimized config plus CSV/JSON/dashboard artifacts for review.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

_TOOL_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TOOL_DIR.parents[2]
_DEFAULT_CONFIG_PATH = _TOOL_DIR / "assets" / "ik_optimizer" / "ik_weight_optimizer_config.json"
# This tool is commonly run as a script from VS Code; add the repo root so local
# packages like soma_retargeter import reliably from any working directory.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import newton

import soma_retargeter.io.bvh as bvh_utils
from soma_retargeter.io.utils import load_json
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.robotics import robot_registry
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.utils.space_conversion_utils import (
    FacingDirectionType,
    SpaceConverter,
)


def _resolve_path(base_dir: pathlib.Path, value: str) -> pathlib.Path:
    """Resolve *value* as an absolute path, using *base_dir* as the base for relative inputs."""
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _validate_config(config: dict[str, Any]) -> None:
    """Fail fast on config shape errors before loading motions or robot assets."""
    for key in ("input_folder", "output_folder", "target_type"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"config.{key} must be a non-empty string")
    extra_robot_paths = config.get("extra_robot_paths", [])
    if isinstance(extra_robot_paths, str):
        extra_robot_paths = [extra_robot_paths]
    if not isinstance(extra_robot_paths, list) or not all(isinstance(p, str) for p in extra_robot_paths):
        raise ValueError("config.extra_robot_paths must be a string or a list of strings")
    for key in (
        "ik_weight_update", "ik_weight_line_search", "clip_guard", "error_model",
        "objective", "effector_priorities", "retargeter_overrides", "plot",
    ):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"config.{key} must be an object")
    try:
        outer_iterations = int(config.get("outer_iterations", 20))
    except (TypeError, ValueError) as exc:
        raise ValueError("config.outer_iterations must be an integer") from exc
    if outer_iterations < 1:
        raise ValueError("config.outer_iterations must be >= 1")


def _log(message: str) -> None:
    """Emit a prefixed log message to stdout."""
    print(f"[Optimizer] {message}")


def _log_block(*lines: str) -> None:
    """Emit a blank prefix line followed by each entry in *lines* indented four spaces."""
    print("[Optimizer]")
    for line in lines:
        print(f"    {line}")


def _elapsed(start_time: float) -> str:
    """Return elapsed seconds since *start_time* as a formatted string."""
    return f"{time.perf_counter() - start_time:.2f}s"


def _elapsed_hms(start_time: float) -> str:
    """Return elapsed time since *start_time* formatted as HH:MM:SS."""
    total_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _gather_bvh_files(config: dict[str, Any], config_dir: pathlib.Path) -> list[pathlib.Path]:
    """Discover and return all .bvh files under the configured input folder."""
    if "input_folder" not in config:
        raise ValueError("Config must include 'input_folder' (root path to discover .bvh motions).")

    path = _resolve_path(config_dir, str(config["input_folder"]))
    if path.is_dir():
        bvh_paths = sorted(path.rglob("*.bvh"))
    elif path.is_file() and path.suffix.lower() == ".bvh":
        bvh_paths = [path]
    else:
        raise FileNotFoundError(f"input_folder does not exist: {path}")

    deduped = sorted({p.resolve() for p in bvh_paths})
    if not deduped:
        raise ValueError(f"No .bvh files found under input_folder: {path}")

    return deduped


def _register_extra_robot_paths(config: dict[str, Any], config_dir: pathlib.Path) -> None:
    """Register any extra robot manifest search paths listed in the config."""
    paths = config.get("extra_robot_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    for value in paths:
        path = _resolve_path(config_dir, value)
        robot_registry.register_robots_path(str(path))
        _log(f"registered robot manifest search path: {path}")


EffectorPoseFrames = tuple[np.ndarray, np.ndarray]


def _clamp01(value: Any) -> float:
    """Clamp *value* to [0.0, 1.0]."""
    return min(1.0, max(0.0, float(value)))


@dataclass
class OptimizerParameters:
    """Per-effector IK match-table weight scales. This is currently the
    only optimized quantity."""

    position_weight_scales: np.ndarray
    rotation_weight_scales: np.ndarray

    @classmethod
    def identity(cls, num_effectors: int) -> OptimizerParameters:
        """Return a parameter set with all weight scales initialised to one."""
        return cls(
            position_weight_scales=np.ones((num_effectors,), dtype=np.float32),
            rotation_weight_scales=np.ones((num_effectors,), dtype=np.float32),
        )

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the current weight scales."""
        return {
            "position_weight_scales": self.position_weight_scales.tolist(),
            "rotation_weight_scales": self.rotation_weight_scales.tolist(),
        }


@dataclass(frozen=True)
class SymmetryPair:
    """Pairing of a Left/Right effector so their IK weights stay mirrored during optimization."""

    left_name: str
    right_name: str
    left_idx: int
    right_idx: int
    base_left_position_weight: float
    base_right_position_weight: float
    base_left_rotation_weight: float
    base_right_rotation_weight: float

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the pair's effector names and indices."""
        return {
            "left_name": self.left_name,
            "right_name": self.right_name,
            "left_idx": self.left_idx,
            "right_idx": self.right_idx,
        }


@dataclass
class SourceAgnosticErrorModel:
    """Per-effector geometry-derived scale and weight arrays used to normalise IK tracking errors."""

    enabled: bool
    parent_indices: np.ndarray
    position_scales: np.ndarray
    rotation_scales: np.ndarray
    position_weights: np.ndarray
    rotation_weights: np.ndarray
    normalized_rotation_weight: float
    reachability_threshold: float
    unreachable_weight_decay: float

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of all error-model fields."""
        return {
            "enabled": self.enabled,
            "parent_indices": self.parent_indices.tolist(),
            "position_scales": self.position_scales.tolist(),
            "rotation_scales": self.rotation_scales.tolist(),
            "position_weights": self.position_weights.tolist(),
            "rotation_weights": self.rotation_weights.tolist(),
            "normalized_rotation_weight": self.normalized_rotation_weight,
            "reachability_threshold": self.reachability_threshold,
            "unreachable_weight_decay": self.unreachable_weight_decay,
        }


class OptimizerConfigWorkspace:
    """Holds a robot target's base retargeter config for the outer loop.

    The per-eval pipeline is built from an in-memory retargeter dict: the base
    config with the current IK-weight scales (and eval-time overrides) applied by
    ``_build_retargeter_config``. Nothing is copied to disk during the run. The
    model and scaler are read from the target's original registry location unchanged;
    only the IK match-table weights are tuned. The final optimized retargeter config
    is the sole on-disk output, written to the output folder.
    """

    def __init__(
        self,
        source_type: str,
        target_type: str,
        output_dir: pathlib.Path,
    ):
        self.source_type = source_type
        self.target_type = target_type

        source_entry = robot_registry.registry.get(target_type)
        self.source_asset_root = pathlib.Path(source_entry.manifest_dir)
        self.base_retargeter_rel = source_entry.retarget_configs[source_type]
        self.base_retargeter = load_json(self.source_asset_root / self.base_retargeter_rel)
        self.base_scaler_rel = self.base_retargeter["human_robot_scaler_config"]
        self.output_retargeter_path = (
            output_dir / f"optimized_{pathlib.Path(self.base_retargeter_rel).name}"
        )
        self.effector_names = list(self.base_retargeter["ik_match_table"].keys())
        self.base_ik_weights = {
            name: (
                float(data["t_weight"]),
                float(data["r_weight"]),
            )
            for name, data in self.base_retargeter["ik_match_table"].items()
        }

    def build_symmetry_pairs(self) -> list[SymmetryPair]:
        """Pair Left*/Right* effectors so their IK weights stay mirrored."""
        index_by_name = {name: idx for idx, name in enumerate(self.effector_names)}
        pairs: list[SymmetryPair] = []
        for left_name, left_idx in index_by_name.items():
            if not left_name.startswith("Left"):
                continue
            right_name = "Right" + left_name[len("Left") :]
            right_idx = index_by_name.get(right_name)
            if right_idx is None:
                continue

            left_t_weight, left_r_weight = self.base_ik_weights[left_name]
            right_t_weight, right_r_weight = self.base_ik_weights[right_name]
            pairs.append(
                SymmetryPair(
                    left_name=left_name,
                    right_name=right_name,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    base_left_position_weight=left_t_weight,
                    base_right_position_weight=right_t_weight,
                    base_left_rotation_weight=left_r_weight,
                    base_right_rotation_weight=right_r_weight,
                )
            )
        return pairs


def _make_pipeline(
    skeleton,
    source_type: str,
    target_type: str,
    retarget_config: dict[str, Any],
) -> SomaRetargetingPipeline:
    """Construct a SomaRetargetingPipeline for the given source/target and retarget config."""
    return SomaRetargetingPipeline(skeleton, source_type, target_type, retarget_config)


def _prepare_pipeline_targets(
    pipeline: SomaRetargetingPipeline,
    animations: list,
    root_tx: wp.transform,
) -> None:
    """Clear the pipeline and load *animations* as input motions with *root_tx*."""
    pipeline.clear()
    pipeline.add_input_motions(animations, [root_tx] * len(animations), scale_animation=True)


def _targets_after_pipeline_init(pipeline: SomaRetargetingPipeline) -> list[np.ndarray]:
    """Return target frames with initialization and stabilization frames stripped."""
    remove = pipeline.num_initialization_frames + pipeline.num_stabilization_frames
    return [np.asarray(targets[remove:], dtype=np.float32) for targets in pipeline.input_targets]


def _apply_symmetry_constraints(params: OptimizerParameters, symmetry_pairs: list[SymmetryPair]) -> None:
    """Project Left*/Right* weight scale pairs to their mean so the two sides stay mirrored."""
    for pair in symmetry_pairs:
        left_idx = pair.left_idx
        right_idx = pair.right_idx

        left_position_weight = pair.base_left_position_weight * float(params.position_weight_scales[left_idx])
        right_position_weight = pair.base_right_position_weight * float(params.position_weight_scales[right_idx])
        projected_position_weight = 0.5 * (left_position_weight + right_position_weight)
        if pair.base_left_position_weight > 1e-8:
            params.position_weight_scales[left_idx] = projected_position_weight / pair.base_left_position_weight
        if pair.base_right_position_weight > 1e-8:
            params.position_weight_scales[right_idx] = projected_position_weight / pair.base_right_position_weight

        left_rotation_weight = pair.base_left_rotation_weight * float(params.rotation_weight_scales[left_idx])
        right_rotation_weight = pair.base_right_rotation_weight * float(params.rotation_weight_scales[right_idx])
        projected_rotation_weight = 0.5 * (left_rotation_weight + right_rotation_weight)
        if pair.base_left_rotation_weight > 1e-8:
            params.rotation_weight_scales[left_idx] = projected_rotation_weight / pair.base_left_rotation_weight
        if pair.base_right_rotation_weight > 1e-8:
            params.rotation_weight_scales[right_idx] = projected_rotation_weight / pair.base_right_rotation_weight


def _buffers_to_arrays(buffers: list[CSVAnimationBuffer]) -> list[np.ndarray]:
    """Convert a list of CSVAnimationBuffers to a list of (num_frames, coord_count) numpy arrays."""
    arrays = []
    for buffer in buffers:
        frames = [np.asarray(frame, dtype=np.float32) for frame in buffer.data if frame is not None]
        if not frames:
            raise RuntimeError("Retargeting produced an empty CSVAnimationBuffer.")
        arrays.append(np.stack(frames, axis=0))
    return arrays


def _aggregate_metrics(per_motion: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-motion metric dicts into mean, std, min, and max across all motions."""
    keys = sorted({key for metrics in per_motion for key in metrics})
    aggregate: dict[str, float] = {}
    for key in keys:
        values = np.asarray([metrics[key] for metrics in per_motion if key in metrics], dtype=np.float32)
        if len(values) == 0:
            continue
        aggregate[key] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_min"] = float(np.min(values))
        aggregate[f"{key}_max"] = float(np.max(values))
    return aggregate


def _metric_delta(baseline: dict[str, float], final: dict[str, float], key: str) -> float | None:
    """Return the reduction (baseline minus final) for *key*, or None if either value is non-finite."""
    baseline_value = _metric_value(baseline, key)
    final_value = _metric_value(final, key)
    if not np.isfinite(baseline_value) or not np.isfinite(final_value):
        return None
    return baseline_value - final_value


def _metric_value(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    """Safely extract a float metric from *metrics*, returning *default* if missing or non-finite."""
    if key in metrics:
        return _finite_or_nan(metrics[key])
    return default


def _rank_motions_by_tracking_loss_improvement(
    history: list[dict[str, Any]],
    bvh_paths: list[pathlib.Path],
    *,
    tracking_metric: str = "tracking_loss",
) -> list[dict[str, Any]]:
    """Rank every motion by tracking-loss improvement from the first to the last history entry."""
    if len(history) < 2:
        return []

    baseline_metrics = history[0].get("per_motion_metrics", [])
    final_metrics = history[-1].get("per_motion_metrics", [])
    count = min(len(baseline_metrics), len(final_metrics), len(bvh_paths))
    if count == 0:
        return []

    comparison_metrics = [
        (tracking_metric, "tracking_loss"),
        ("ik_frame_loss_p95", "ik_frame_loss_p95"),
        ("ik_worst_frame_loss", "ik_worst_frame_loss"),
        ("ik_position_rmse_m", "ik_position_rmse_m"),
        ("ik_orientation_rmse_rad", "ik_orientation_rmse_rad"),
        ("reachability_mean", "reachability_mean"),
    ]
    rows: list[dict[str, Any]] = []
    for motion_idx in range(count):
        baseline = baseline_metrics[motion_idx]
        final = final_metrics[motion_idx]
        row: dict[str, Any] = {
            "rank": 0,
            "motion": str(bvh_paths[motion_idx]),
            "motion_name": bvh_paths[motion_idx].stem,
            "baseline_iteration": history[0].get("plot_iteration", int(history[0]["iteration"]) + 1),
            "final_iteration": history[-1].get("plot_iteration", int(history[-1]["iteration"]) + 1),
        }
        for data_key, label in comparison_metrics:
            if data_key in baseline:
                row[f"baseline_{label}"] = float(baseline[data_key])
            if data_key in final:
                row[f"final_{label}"] = float(final[data_key])
            delta = _metric_delta(baseline, final, data_key)
            if delta is not None:
                row[f"improvement_{label}"] = delta
                baseline_value = float(baseline[data_key])
                if abs(baseline_value) > 1e-8:
                    row[f"improvement_{label}_percent"] = 100.0 * delta / abs(baseline_value)

        # Reachability is higher-is-better (unlike the loss metrics); flip its sign.
        if "reachability_mean" in baseline and "reachability_mean" in final:
            row["improvement_reachability_mean"] = float(final["reachability_mean"]) - float(
                baseline["reachability_mean"]
            )
            baseline_reachability = float(baseline["reachability_mean"])
            if abs(baseline_reachability) > 1e-8:
                row["improvement_reachability_mean_percent"] = (
                    100.0 * row["improvement_reachability_mean"] / abs(baseline_reachability)
                )
        row["_sort_value"] = float(row.get("improvement_tracking_loss", float("-inf")))
        if row["_sort_value"] > 0.0:
            row["tracking_loss_status"] = "improved"
        elif row["_sort_value"] < 0.0:
            row["tracking_loss_status"] = "regressed"
        else:
            row["tracking_loss_status"] = "unchanged"
        rows.append(row)

    rows.sort(key=lambda item: item["_sort_value"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row.pop("_sort_value", None)
    return rows


def _write_motion_improvement_ranking_csv(
    rows: list[dict[str, Any]],
    output_dir: pathlib.Path,
) -> pathlib.Path | None:
    """Write the motion improvement ranking to a CSV and return the path, or None if empty."""
    if not rows:
        return None

    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["rank", "motion_name", "motion", "baseline_iteration", "final_iteration"]
    ordered_fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    path = output_dir / "optimizer_improvement_ranking.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Forward kinematics evaluation
# ---------------------------------------------------------------------------


@wp.kernel
def _extract_body_poses_kernel(
    body_q: wp.array(dtype=wp.transform),
    pos_link_indices: wp.array(dtype=wp.int32),
    rot_link_indices: wp.array(dtype=wp.int32),
    body_count: wp.int32,
    num_pos_links: wp.int32,
    num_rot_links: wp.int32,
    dst_offset: wp.int32,
    out_pos: wp.array2d(dtype=wp.vec3),
    out_rot: wp.array2d(dtype=wp.quat),
):
    env = wp.tid()
    base = env * body_count
    row = dst_offset + env

    for i in range(num_pos_links):
        tx = body_q[base + pos_link_indices[i]]
        out_pos[row, i] = tx.p

    for i in range(num_rot_links):
        tx = body_q[base + rot_link_indices[i]]
        out_rot[row, i] = wp.normalize(tx.q)


class BodyEvaluator:
    """Runs FK on GPU and downloads solved effector poses for CPU-side metrics."""

    def __init__(self, pipeline: SomaRetargetingPipeline, fk_frame_batch_size: int = 100):
        self.pipeline = pipeline
        self.max_batch_size = max(1, int(fk_frame_batch_size))
        self.body_count = pipeline.num_body_count
        self.coord_count = pipeline.ik_model.joint_coord_count

        self.pos_link_indices_np = np.array(
            [idx for idx, _ in pipeline.mapped_body_link_pos_data], dtype=np.int32
        )
        self.rot_link_indices_np = np.array(
            [idx for idx, _ in pipeline.mapped_body_link_rot_data], dtype=np.int32
        )
        self.num_pos_links = len(self.pos_link_indices_np)
        self.num_rot_links = len(self.rot_link_indices_np)

        self._wp_pos_idx = wp.array(self.pos_link_indices_np, dtype=wp.int32)
        self._wp_rot_idx = wp.array(self.rot_link_indices_np, dtype=wp.int32)

        self._model = None
        self._state = None
        self._model_batch_size = 0

    def _ensure_model(self, required: int) -> None:
        """Lazily (re)build the FK model when the required batch size exceeds the current one."""
        batch = max(1, min(self.max_batch_size, required))
        if self._model is not None and self._model_batch_size >= batch:
            return
        self._model = self.pipeline._build_model(batch)
        self._state = self._model.state()
        self._model_batch_size = batch

    def evaluate(self, q_frames_list: list[np.ndarray]) -> list[EffectorPoseFrames]:
        """Run FK for one or more motions; returns one (positions, rotations) per motion.

        Each entry of ``q_frames_list`` is a motion's solved coords (shape
        ``(num_frames, coord_count)``, the solved ``reference_q``). All motions are
        packed into one flat array and FK'd in sub-batches of ``fk_frame_batch_size`` on
        a single reused Newton model. It works for a single motion (a list of one) too.
        """
        frame_counts = [int(np.asarray(q).shape[0]) for q in q_frames_list]
        flat = np.concatenate(
            [np.asarray(q, dtype=np.float32) for q in q_frames_list], axis=0
        )
        if flat.shape[1] != self.coord_count:
            raise ValueError(
                f"Expected q frames with {self.coord_count} coordinates, got {flat.shape[1]}."
            )
        wp_q = wp.array(flat, dtype=float, ndim=2)
        total_frames = flat.shape[0]
        self._ensure_model(min(total_frames, self.max_batch_size))

        out_pos = wp.zeros(shape=(total_frames, self.num_pos_links), dtype=wp.vec3)
        out_rot = wp.zeros(shape=(total_frames, self.num_rot_links), dtype=wp.quat)

        bs = self._model_batch_size
        pad_q = wp.zeros(shape=(bs, self.coord_count), dtype=float)

        for start in range(0, total_frames, bs):
            end = min(start + bs, total_frames)
            actual = end - start

            src_slice = wp_q[start:end]
            if actual < bs:
                wp.copy(pad_q, src_slice, count=actual * self.coord_count)
                last_row = wp_q[end - 1 : end]
                for fill in range(actual, bs):
                    wp.copy(
                        pad_q, last_row,
                        dest_offset=fill * self.coord_count,
                        count=self.coord_count,
                    )
                self._model.joint_q.assign(pad_q.reshape(bs * self.coord_count))
            else:
                self._model.joint_q.assign(src_slice.reshape(bs * self.coord_count))

            newton.eval_fk(
                self._model, self._model.joint_q, self._model.joint_qd, self._state
            )

            wp.launch(
                _extract_body_poses_kernel,
                dim=actual,
                inputs=[
                    self._state.body_q,
                    self._wp_pos_idx,
                    self._wp_rot_idx,
                    self.body_count,
                    self.num_pos_links,
                    self.num_rot_links,
                    start,
                    out_pos,
                    out_rot,
                ],
            )

        wp.synchronize()

        pos_np = out_pos.numpy()
        rot_np = out_rot.numpy()

        results: list[EffectorPoseFrames] = []
        offset = 0
        for n in frame_counts:
            results.append((pos_np[offset : offset + n], rot_np[offset : offset + n]))
            offset += n
        return results


def _normalize_quats(q: np.ndarray) -> np.ndarray:
    """L2-normalise an array of quaternions along the last axis."""
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of (x, y, z, w) quaternion arrays (matches ``wp.mul``)."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=-1)


def _rotvec_error(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation-vector error: axis * angle of (target^-1 * actual), per quaternion.

    Pure numpy: the poses are already host arrays at the metric stage, so this
    avoids bouncing them back to the GPU just for elementwise quaternion math.
    """
    shape = actual.shape[:-1] + (3,)
    if actual.size == 0:
        return np.zeros(shape, dtype=np.float32)
    a = _normalize_quats(actual.reshape(-1, 4).astype(np.float32))
    t = _normalize_quats(target.reshape(-1, 4).astype(np.float32))
    t_inv = t.copy()
    t_inv[:, :3] = -t_inv[:, :3]            # inverse of a unit quat == conjugate
    q = _quat_mul(t_inv, a)
    flip = q[:, 3] < 0.0                    # shortest-arc sign convention
    q[flip] = -q[flip]
    vec = q[:, :3]
    vec_norm = np.linalg.norm(vec, axis=-1)
    # angle via atan2 (well-conditioned near 0, matches wp.quat_to_axis_angle);
    # rotvec = axis*angle, with the small-angle limit angle/vec_norm -> 2.
    angle = 2.0 * np.arctan2(vec_norm, q[:, 3])
    scale = np.where(vec_norm > 1e-8, angle / np.maximum(vec_norm, 1e-12), 2.0)
    return (vec * scale[:, None]).astype(np.float32).reshape(shape)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Geodesic angle between two quaternions: 2 * acos(|dot(a, b)|) (pure numpy)."""
    shape = q_a.shape[:-1]
    if q_a.size == 0:
        return np.zeros(shape, dtype=np.float32)
    a = _normalize_quats(q_a.reshape(-1, 4).astype(np.float32))
    b = _normalize_quats(q_b.reshape(-1, 4).astype(np.float32))
    d = np.minimum(np.abs(np.sum(a * b, axis=-1)), 1.0)
    return (2.0 * np.arccos(d)).astype(np.float32).reshape(shape)


@dataclass
class _TrackingErrorAccumulators:
    """Streaming accumulators for the IK-weight update.

    The update rebalances per-effector IK weights from errors accumulated across
    all clips. ``count`` is the sum of clip weights; with no hard-clip emphasis it
    is simply the number of clips.
    """

    pos_err_norm: np.ndarray
    rot_err_norm: np.ndarray
    count: float = 0.0

    @staticmethod
    def zeros(num_effectors: int) -> _TrackingErrorAccumulators:
        """Return a zeroed accumulator sized for *num_effectors* effectors."""
        return _TrackingErrorAccumulators(
            pos_err_norm=np.zeros(num_effectors, dtype=np.float32),
            rot_err_norm=np.zeros(num_effectors, dtype=np.float32),
            count=0.0,
        )


@dataclass(frozen=True)
class IkWeightUpdateConfig:
    """Scalar controls for one IK-weight rebalance step (see ``_updated_params``).

    Root protection is per-component (``protect_root_position`` / ``protect_root_rotation``,
    both default True).  Freeing root rotation (for robots with limited torso DoF)
    floors it at ``root_rotation_min_scale``.
    """

    learning_rate: float = 0.15
    min_scale: float = 0.35
    max_scale: float = 3.0
    protect_root_position: bool = True
    protect_root_rotation: bool = True
    root_rotation_min_scale: float = 0.5
    # Steer the update toward higher-loss clips. 0 is uniform; 1 is proportional
    # to the clip loss.
    hard_clip_emphasis: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> IkWeightUpdateConfig:
        """Construct from the ``ik_weight_update`` sub-dict of the optimizer config."""
        g = dict(config.get("ik_weight_update", {}))
        return cls(
            learning_rate=float(g.get("learning_rate", 0.15)),
            min_scale=float(g.get("min_scale", 0.35)),
            max_scale=float(g.get("max_scale", 3.0)),
            protect_root_position=bool(g.get("protect_root_position", True)),
            protect_root_rotation=bool(g.get("protect_root_rotation", True)),
            root_rotation_min_scale=float(g.get("root_rotation_min_scale", 0.5)),
            hard_clip_emphasis=max(0.0, float(g.get("hard_clip_emphasis", 0.0))),
        )


@dataclass(frozen=True)
class LineSearchConfig:
    """Backtracking line-search controls for the IK-weight update (config key
    ``ik_weight_line_search``)."""

    enabled: bool = True
    max_backtracks: int = 5
    shrink: float = 0.5
    min_improvement: float = 0.00005

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LineSearchConfig:
        """Construct from the ``ik_weight_line_search`` sub-dict of the optimizer config."""
        ls = dict(config.get("ik_weight_line_search", {}))
        return cls(
            enabled=bool(ls.get("enabled", True)),
            max_backtracks=max(0, int(ls.get("max_backtracks", 5))),
            shrink=float(ls.get("shrink", 0.5)),
            min_improvement=float(ls.get("min_improvement", 0.00005)),
        )


_CLIP_GUARD_MODES = ("off", "gate", "penalty", "auto")


@dataclass(frozen=True)
class ClipGuardConfig:
    """Per-clip regression policy for line-search acceptance.

    ``gate`` is the strict mode: a probe is rejected when a guarded clip metric
    regresses beyond its configured limit. This is the default for flexible
    embodiments where those limits are usually feasible.

    ``penalty`` keeps the same robust clip metrics in the objective instead of
    rejecting them outright. It still hard-rejects non-finite solves and large
    frame or acceleration spikes, which makes it more useful for low-DOF targets
    that must trade off one clip against another.

    ``auto`` starts with ``gate`` and moves once to ``penalty`` when the gated
    phase would otherwise stop on convergence patience. ``off`` disables the clip
    checks and penalty term.

    ``penalty_balance`` is a 0..1 scale on the clip-regression penalty, measured
    against the initial tracking loss:
    ``penalty_balance * initial_tracking_loss * raw_clip_excess``.
    """

    mode: str = "gate"
    penalty_balance: float = 1.0
    penalty_tail_fraction: float = 0.2  # 0 disables the tail term
    penalty_tail_weight: float = 1.0
    # Penalty mode keeps only severe spikes on the hard-reject path.
    catastrophic_frame_loss_max_regression: float = 2.0
    catastrophic_accel_spike_regression: float = 2.0
    max_tracking_loss_regression: float = 0.05
    # Worst-frame loss is jumpy; p95 is the more stable frame-loss signal.
    max_worst_frame_loss_regression: float = 0.05
    max_clip_frame_loss_p95_regression: float = 0.05
    max_aligned_frame_loss_p95_regression: float = 0.05
    max_linear_accel_p95_regression: float = 0.35
    max_angular_accel_p95_regression: float = 0.35
    max_linear_accel_spike_regression: float = 0.25
    max_angular_accel_spike_regression: float = 0.25
    max_logged_violations: int = 3
    guard_aligned_max: bool = False  # strict option; p95 is the default aligned-frame gate

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ClipGuardConfig:
        """Construct from the ``clip_guard`` sub-dict of the optimizer config."""
        g = dict(config.get("clip_guard", {}))
        mode = str(g.get("mode", "auto"))
        if mode not in _CLIP_GUARD_MODES:
            valid = ", ".join(repr(m) for m in _CLIP_GUARD_MODES)
            raise ValueError(
                f"clip_guard.mode must be one of {valid}; got {mode!r}"
            )
        return cls(
            mode=mode,
            penalty_balance=_clamp01(g.get("penalty_balance", 1.0)),
            penalty_tail_fraction=_clamp01(g.get("penalty_tail_fraction", 0.2)),
            penalty_tail_weight=_clamp01(g.get("penalty_tail_weight", 1.0)),
            catastrophic_frame_loss_max_regression=max(0.0, float(g.get("catastrophic_frame_loss_max_regression", 2.0))),
            catastrophic_accel_spike_regression=max(0.0, float(g.get("catastrophic_accel_spike_regression", 2.0))),
            max_tracking_loss_regression=float(g.get("max_tracking_loss_regression", 0.05)),
            max_worst_frame_loss_regression=float(g.get("max_worst_frame_loss_regression", 0.05)),
            max_clip_frame_loss_p95_regression=float(g.get("max_clip_frame_loss_p95_regression", 0.05)),
            max_aligned_frame_loss_p95_regression=float(g.get("max_aligned_frame_loss_p95_regression", 0.05)),
            max_linear_accel_p95_regression=float(g.get("max_linear_accel_p95_regression", 0.35)),
            max_angular_accel_p95_regression=float(g.get("max_angular_accel_p95_regression", 0.35)),
            max_linear_accel_spike_regression=float(g.get("max_linear_accel_spike_regression", 0.25)),
            max_angular_accel_spike_regression=float(g.get("max_angular_accel_spike_regression", 0.25)),
            max_logged_violations=max(1, int(g.get("max_logged_violations", 3))),
            guard_aligned_max=bool(g.get("guard_aligned_max", False)),
        )


@dataclass(frozen=True)
class PlotConfig:
    """Dashboard rendering controls (config key ``plot``)."""

    enabled: bool = True
    live: bool = False
    dpi: int = 140

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PlotConfig:
        """Construct from the ``plot`` sub-dict of the optimizer config."""
        p = dict(config.get("plot", {}))
        return cls(
            enabled=bool(p.get("enabled", True)),
            live=bool(p.get("live", False)),
            dpi=int(p.get("dpi", 140)),
        )


@dataclass(frozen=True)
class ObjectiveConfig:
    """Line-search objective controls (config key ``objective``).

    ``smoothness_balance`` is a 0..1 fraction of the initial tracking loss. It
    controls how much the line search values smoother solved robot motion:
    ``smoothness_balance * initial_tracking_loss * normalized_smoothness``.
    """

    smoothness_balance: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ObjectiveConfig:
        """Construct from the ``objective`` sub-dict of the optimizer config."""
        o = dict(config.get("objective", {}))
        return cls(smoothness_balance=_clamp01(o.get("smoothness_balance", 0.0)))


def _compute_metrics_from_poses(
    reference_body_poses: EffectorPoseFrames,
    target_frames: np.ndarray,
    ik_mask: np.ndarray,
    error_model: SourceAgnosticErrorModel,
    *,
    accumulators: _TrackingErrorAccumulators | None = None,
    unreachable_weight_decay: float = 0.3,
    hard_clip_emphasis: float = 0.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate tracking metrics for a single motion, scored on the solved IK motion.

    The tracking loss and per-effector IK-weight rebalance signals come from
    ``reference_pos``/``reference_rot`` (FK of ``reference_q``, the solved retarget)
    vs the kinematic targets. That is the deployed artifact and what the IK weights
    directly produce. Acceleration/smoothness metrics are likewise on the solved motion.
    When *accumulators* is provided, per-effector rebalancing signals are added in
    place for the global update. The returned frame-loss array is transient input
    for the aligned-frame line-search guard.
    """
    reference_pos, reference_rot = reference_body_poses

    num_frames = min(len(reference_pos), len(target_frames), len(ik_mask))

    reference_pos = reference_pos[:num_frames]
    reference_rot = reference_rot[:num_frames]
    target_frames = target_frames[:num_frames]

    mask = ik_mask[:num_frames].copy()
    if not np.any(mask):
        mask[:] = True

    target_pos = target_frames[:, :, 0:3]
    target_rot = target_frames[:, :, 3:7]

    position_scales = error_model.position_scales.reshape(1, -1, 1)
    rotation_scales = error_model.rotation_scales.reshape(1, -1, 1)
    pw_3d = error_model.position_weights.reshape(1, -1, 1)
    rw_3d = error_model.rotation_weights.reshape(1, -1, 1)

    ik_pos_error = reference_pos - target_pos
    ik_rot_error_vec = _rotvec_error(reference_rot, target_rot)
    normalized_ik_pos_error = ik_pos_error / np.maximum(position_scales, 1e-8)
    normalized_ik_rot_error = ik_rot_error_vec / np.maximum(rotation_scales, 1e-8)

    ik_pos_norm = np.linalg.norm(normalized_ik_pos_error[mask], axis=-1)
    reachability = np.exp(-np.mean(ik_pos_norm, axis=0) / error_model.reachability_threshold)

    ik_pos_rmse = float(np.sqrt(np.mean(np.square(ik_pos_error[mask]))))
    ik_root_pos_rmse = float(np.sqrt(np.mean(np.square(ik_pos_error[mask, 0]))))
    norm_ik_pos_rmse = float(np.sqrt(np.mean(np.square(normalized_ik_pos_error[mask]))))
    ik_rot_rmse = float(np.sqrt(np.mean(np.square(
        _quat_angle_error(reference_rot[mask], target_rot[mask])))))
    ik_root_rot_rmse = float(np.sqrt(np.mean(np.square(
        _quat_angle_error(reference_rot[mask, 0], target_rot[mask, 0])))))

    # Keep per-frame loss alongside the mean so isolated bad frames stay visible.
    ik_frame_loss = (
        np.mean(pw_3d * np.square(normalized_ik_pos_error), axis=(1, 2))
        + error_model.normalized_rotation_weight
        * np.mean(rw_3d * np.square(normalized_ik_rot_error), axis=(1, 2))
    )
    masked_ik_frame_loss = ik_frame_loss[mask]
    tracking_loss = float(np.mean(masked_ik_frame_loss))
    ik_frame_loss_p95 = float(np.percentile(masked_ik_frame_loss, 95.0))
    ik_worst_frame_loss = float(np.max(masked_ik_frame_loss))

    # Second-difference acceleration on the solved motion. Rotational acceleration
    # uses the exp-map angular increment.
    accel_rmse = 0.0
    accel_max = 0.0
    accel_p95 = 0.0
    ang_accel_max = 0.0
    ang_accel_p95 = 0.0
    smoothness_loss = 0.0
    accel_mask = mask[2:] & mask[1:-1] & mask[:-2]
    if np.any(accel_mask):
        pos_accel = reference_pos[2:] - 2.0 * reference_pos[1:-1] + reference_pos[:-2]
        norm_pos_accel = pos_accel / np.maximum(position_scales, 1e-8)
        omega = _rotvec_error(reference_rot[1:], reference_rot[:-1])
        rot_accel = omega[1:] - omega[:-1]
        norm_rot_accel = rot_accel / np.maximum(rotation_scales, 1e-8)

        accel_mag = np.linalg.norm(pos_accel[accel_mask], axis=-1)
        accel_rmse = float(np.sqrt(np.mean(np.square(pos_accel[accel_mask]))))
        accel_max = float(np.max(accel_mag))
        accel_p95 = float(np.percentile(accel_mag, 95.0))
        ang_mag = np.linalg.norm(rot_accel[accel_mask], axis=-1)
        ang_accel_max = float(np.max(ang_mag))
        ang_accel_p95 = float(np.percentile(ang_mag, 95.0))

        accel_loss_pf = (
            np.mean(pw_3d * np.square(norm_pos_accel), axis=(1, 2))
            + error_model.normalized_rotation_weight
            * np.mean(rw_3d * np.square(norm_rot_accel), axis=(1, 2))
        )
        smoothness_loss = float(np.mean(accel_loss_pf[accel_mask]))

    metrics = {
        "tracking_loss": tracking_loss,
        "ik_frame_loss_p95": ik_frame_loss_p95,
        "ik_worst_frame_loss": ik_worst_frame_loss,
        "ik_position_rmse_m": ik_pos_rmse,
        "ik_root_position_rmse_m": ik_root_pos_rmse,
        "ik_orientation_rmse_rad": ik_rot_rmse,
        "ik_root_orientation_rmse_rad": ik_root_rot_rmse,
        "ik_position_rmse_normalized": norm_ik_pos_rmse,
        "reachability_mean": float(np.mean(reachability)),
        "reachability_min": float(np.min(reachability)),
        "ik_linear_accel_rmse": accel_rmse,
        "ik_linear_accel_max": accel_max,
        "ik_linear_accel_p95": accel_p95,
        "ik_angular_accel_max": ang_accel_max,
        "ik_angular_accel_p95": ang_accel_p95,
        "ik_smoothness_loss": smoothness_loss,
    }

    if accumulators is not None:
        masked_norm_pos = normalized_ik_pos_error[mask]
        masked_norm_rot = normalized_ik_rot_error[mask]
        if len(masked_norm_pos) > 0:
            pw_1d = error_model.position_weights
            rw_1d = error_model.rotation_weights

            # A decay of 1.0 leaves all effectors equally weighted.
            reach_factor = unreachable_weight_decay + (1.0 - unreachable_weight_decay) * reachability

            clip_weight = float(tracking_loss) ** hard_clip_emphasis if hard_clip_emphasis > 0.0 else 1.0

            accumulators.pos_err_norm += clip_weight * (
                np.mean(np.linalg.norm(masked_norm_pos, axis=-1), axis=0)
                * pw_1d
                * reach_factor
            )
            accumulators.rot_err_norm += clip_weight * (
                np.mean(np.linalg.norm(masked_norm_rot, axis=-1), axis=0)
                * rw_1d
                * reach_factor
            )
            accumulators.count += clip_weight

    return metrics, masked_ik_frame_loss


def _updated_params(
    params: OptimizerParameters,
    acc: _TrackingErrorAccumulators,
    *,
    step_scale: float,
    root_mask: np.ndarray | None,
    symmetry_pairs: list[SymmetryPair],
    ik: IkWeightUpdateConfig,
) -> tuple[OptimizerParameters, dict[str, float]]:
    """Return a *new* parameter set with one scaled IK-weight rebalance applied.

    Non-mutating, so the line search can re-apply the same direction from the
    same base at different ``step_scale`` values.

    Root protection is per-component.  Root tracking propagates down the whole
    kinematic tree, so by default the root effector (``root_mask``) is excluded
    from IK-weight rebalancing.  But a robot may need tight root *translation*
    while tolerating some root *orientation* freedom (e.g. limited torso DoF):
    set ``protect_root_rotation`` False to let the root rotation weight adapt,
    floored at ``root_rotation_min_scale`` so it cannot collapse.  The loss-gated
    line search is the primary safeguard; this floor is extra insurance because
    the loss down-weights rotation.
    """
    new = copy.deepcopy(params)
    count = max(1, acc.count)

    pos_err_norm = acc.pos_err_norm / count
    rot_err_norm = acc.rot_err_norm / count
    pos_mean = float(np.mean(pos_err_norm)) + 1e-8
    rot_mean = float(np.mean(rot_err_norm)) + 1e-8
    pos_expo = step_scale * ik.learning_rate * ((pos_err_norm / pos_mean) - 1.0)
    rot_expo = step_scale * ik.learning_rate * ((rot_err_norm / rot_mean) - 1.0)
    rot_min = np.full_like(rot_expo, ik.min_scale)
    if root_mask is not None:
        if ik.protect_root_position:
            pos_expo = np.where(root_mask, 0.0, pos_expo)
        if ik.protect_root_rotation:
            rot_expo = np.where(root_mask, 0.0, rot_expo)
        else:
            # Freed root rotation gets a higher floor than distal effectors
            # so it can adapt without collapsing root orientation.
            rot_min = np.where(root_mask, ik.root_rotation_min_scale, ik.min_scale)
    new.position_weight_scales = np.clip(
        new.position_weight_scales * np.exp(pos_expo), ik.min_scale, ik.max_scale
    )
    new.rotation_weight_scales = np.clip(
        new.rotation_weight_scales * np.exp(rot_expo), rot_min, ik.max_scale
    )

    _apply_symmetry_constraints(new, symmetry_pairs)

    return new, {
        "position_weight_update_norm": float(np.linalg.norm(new.position_weight_scales - params.position_weight_scales)),
        "rotation_weight_update_norm": float(np.linalg.norm(new.rotation_weight_scales - params.rotation_weight_scales)),
        "symmetry_pair_count": float(len(symmetry_pairs)),
    }


# ---------------------------------------------------------------------------
# Clip guard and penalty policy
# ---------------------------------------------------------------------------


def _clip_guard_violations(
    anchor_metrics: list[dict[str, float] | None],
    trial_metrics: list[dict[str, float] | None],
    anchor_frame_losses: list[np.ndarray | None],
    trial_frame_losses: list[np.ndarray | None],
    motion_names: list[str],
    guard: ClipGuardConfig,
) -> list[dict[str, Any]]:
    """Return solved-IK regressions for one line-search probe.

    Scalar metrics compare clip summaries against the current anchor. Aligned
    frame-loss checks compare the same frame index in anchor and trial, which catches
    a local regression even when the clip's absolute worst frame moved elsewhere.

    Every entry has a ``gates`` flag. Gate mode uses it to split blocking and
    advisory entries; penalty mode can reuse the same regressions as advisory data.
    Results are sorted worst-first by relative regression.
    """
    specs = (
        ("tracking_loss", guard.max_tracking_loss_regression, 1e-6),
        ("ik_frame_loss_p95", guard.max_clip_frame_loss_p95_regression, 1e-6),
        ("ik_worst_frame_loss", guard.max_worst_frame_loss_regression, 1e-6),
        ("ik_linear_accel_p95", guard.max_linear_accel_p95_regression, 1e-4),
        ("ik_angular_accel_p95", guard.max_angular_accel_p95_regression, 1e-4),
    )
    frame_floor = 1e-6  # denominator floor so near-zero baseline frames don't blow up the ratio
    violations: list[dict[str, Any]] = []
    for idx, (anchor_m, trial_m) in enumerate(zip(anchor_metrics, trial_metrics)):
        if anchor_m is None or trial_m is None:
            continue
        for metric, limit, floor in specs:
            anchor_val = float(anchor_m.get(metric, 0.0))
            trial_val = float(trial_m.get(metric, 0.0))
            relative = (trial_val - anchor_val) / max(abs(anchor_val), floor)
            if relative > limit:
                violations.append({
                    "guard": "clip_guard",
                    "motion": motion_names[idx],
                    "metric": metric,
                    "anchor": anchor_val,
                    "trial": trial_val,
                    "relative": relative,
                    "limit": limit,
                    "gates": True,
                })

        anchor_fl = anchor_frame_losses[idx] if idx < len(anchor_frame_losses) else None
        trial_fl = trial_frame_losses[idx] if idx < len(trial_frame_losses) else None
        if anchor_fl is None or trial_fl is None:
            continue
        n = min(len(anchor_fl), len(trial_fl))
        if n == 0:
            continue
        a = np.asarray(anchor_fl[:n], dtype=np.float64)
        t = np.asarray(trial_fl[:n], dtype=np.float64)
        rel = (t - a) / np.maximum(np.abs(a), frame_floor)
        # Attach a representative frame for both aligned-frame summaries.
        p95_val = float(np.percentile(rel, 95.0))
        max_idx = int(np.argmax(rel))
        p95_idx = int(np.argmin(np.abs(rel - p95_val)))
        for metric, value, fidx, gates, limit in (
            (
                "ik_frame_loss_regression_p95",
                p95_val,
                p95_idx,
                True,
                guard.max_aligned_frame_loss_p95_regression,
            ),
            (
                "ik_frame_loss_regression_max",
                float(rel[max_idx]),
                max_idx,
                guard.guard_aligned_max,
                guard.max_tracking_loss_regression,
            ),
        ):
            if value > limit:
                violations.append({
                    "guard": "clip_guard",
                    "motion": motion_names[idx],
                    "metric": metric,
                    "frame": int(fidx),
                    "anchor": float(a[fidx]),
                    "trial": float(t[fidx]),
                    "relative": value,
                    "limit": limit,
                    "gates": gates,
                })
    violations.sort(key=lambda v: v["relative"], reverse=True)
    return violations


def _best_clip_guard_violations(
    best_metrics: list[dict[str, float] | None],
    trial_metrics: list[dict[str, float] | None],
    motion_names: list[str],
    guard: ClipGuardConfig,
) -> list[dict[str, Any]]:
    """Return regressions against each clip's best accepted spike metrics."""
    specs = (
        ("best_clip_guard", "ik_worst_frame_loss", guard.max_worst_frame_loss_regression, 1e-6),
        ("best_smoothness_guard", "ik_linear_accel_max", guard.max_linear_accel_spike_regression, 1e-4),
        ("best_smoothness_guard", "ik_angular_accel_max", guard.max_angular_accel_spike_regression, 1e-4),
    )
    violations: list[dict[str, Any]] = []
    for idx, (best_m, trial_m) in enumerate(zip(best_metrics, trial_metrics)):
        if best_m is None or trial_m is None:
            continue
        for guard_name, metric, limit, floor in specs:
            best_val = float(best_m.get(metric, 0.0))
            trial_val = float(trial_m.get(metric, 0.0))
            relative = (trial_val - best_val) / max(abs(best_val), floor)
            if relative > limit:
                violations.append({
                    "guard": guard_name,
                    "motion": motion_names[idx],
                    "metric": metric,
                    "baseline": best_val,
                    "trial": trial_val,
                    "relative": relative,
                    "limit": limit,
                    "gates": True,
                })
    violations.sort(key=lambda v: v["relative"], reverse=True)
    return violations


_BEST_GUARD_METRICS = ("ik_worst_frame_loss", "ik_linear_accel_max", "ik_angular_accel_max")


def _update_best_guard_metrics(
    best_metrics: list[dict[str, float] | None],
    current_metrics: list[dict[str, float] | None],
) -> None:
    """Update the per-clip best accepted values for spike metrics in place."""
    for idx, metrics in enumerate(current_metrics):
        if metrics is None:
            continue
        current_best = best_metrics[idx]
        if current_best is None:
            best_metrics[idx] = dict(metrics)
            continue
        for metric in _BEST_GUARD_METRICS:
            if float(metrics.get(metric, float("inf"))) < float(current_best.get(metric, float("inf"))):
                current_best[metric] = float(metrics[metric])


# Robust scalar clip metrics used by penalty mode. Single-frame maxima stay on the
# hard-reject path; aligned p95 is computed from frame-loss arrays below.
_PENALTY_SPECS = (
    ("tracking_loss", "max_tracking_loss_regression", 1e-6),
    ("ik_frame_loss_p95", "max_clip_frame_loss_p95_regression", 1e-6),
    ("ik_linear_accel_p95", "max_linear_accel_p95_regression", 1e-4),
    ("ik_angular_accel_p95", "max_angular_accel_p95_regression", 1e-4),
)

_FRAME_REGRESSION_FLOOR = 1e-6


def _aligned_frame_regression(
    base_fl: np.ndarray | None, trial_fl: np.ndarray | None
) -> tuple[float, float, int, float, float]:
    """Return p95/max aligned frame-loss regression plus the max frame details."""
    if base_fl is None or trial_fl is None:
        return 0.0, 0.0, 0, 0.0, 0.0
    n = min(len(base_fl), len(trial_fl))
    if n == 0:
        return 0.0, 0.0, 0, 0.0, 0.0
    a = np.asarray(base_fl[:n], dtype=np.float64)
    t = np.asarray(trial_fl[:n], dtype=np.float64)
    rel = (t - a) / np.maximum(np.abs(a), _FRAME_REGRESSION_FLOOR)
    max_idx = int(np.argmax(rel))
    return float(np.percentile(rel, 95.0)), float(rel[max_idx]), max_idx, float(a[max_idx]), float(t[max_idx])


def _clip_excess_by_motion(
    trial_metrics: list[dict[str, float] | None],
    baseline_metrics: list[dict[str, float] | None],
    trial_frame_losses: list[np.ndarray | None] | None,
    baseline_frame_losses: list[np.ndarray | None] | None,
    guard: ClipGuardConfig,
) -> tuple[list[float], list[int], dict[str, float]]:
    """Return per-clip hinge excesses and aggregate metric contributors."""
    trial_frame_losses = trial_frame_losses or []
    baseline_frame_losses = baseline_frame_losses or []
    excesses: list[float] = []
    indices: list[int] = []
    metric_excess: dict[str, float] = {}
    for idx, (base_m, trial_m) in enumerate(zip(baseline_metrics, trial_metrics)):
        if base_m is None or trial_m is None:
            continue
        total = 0.0
        for metric, limit_attr, floor in _PENALTY_SPECS:
            base_val = float(base_m.get(metric, 0.0))
            trial_val = float(trial_m.get(metric, 0.0))
            rel = (trial_val - base_val) / max(abs(base_val), floor)
            e = max(0.0, rel - getattr(guard, limit_attr))
            total += e
            if e > 0.0:
                metric_excess[metric] = metric_excess.get(metric, 0.0) + e
        base_fl = baseline_frame_losses[idx] if idx < len(baseline_frame_losses) else None
        trial_fl = trial_frame_losses[idx] if idx < len(trial_frame_losses) else None
        e = max(0.0, _aligned_frame_regression(base_fl, trial_fl)[0] - guard.max_aligned_frame_loss_p95_regression)
        total += e
        if e > 0.0:
            metric_excess["aligned_frame_loss_p95"] = metric_excess.get("aligned_frame_loss_p95", 0.0) + e
        excesses.append(total)
        indices.append(idx)
    return excesses, indices, metric_excess


def _penalty_terms(excesses: list[float], tail_fraction: float) -> tuple[float, float, np.ndarray]:
    """Return (mean_excess, tail_excess, sorted_order) for a list of per-clip hinge excesses."""
    if not excesses:
        return 0.0, 0.0, np.asarray([], dtype=np.int64)
    arr = np.asarray(excesses, dtype=np.float64)
    order = np.argsort(arr)
    mean_term = float(arr.mean())
    tail_term = 0.0
    if tail_fraction > 0.0:
        k = max(1, int(np.ceil(tail_fraction * len(arr))))
        tail_term = float(arr[order[-k:]].mean())
    return mean_term, tail_term, order


def _clip_penalty(
    trial_metrics: list[dict[str, float] | None],
    baseline_metrics: list[dict[str, float] | None] | None,
    trial_frame_losses: list[np.ndarray | None] | None,
    baseline_frame_losses: list[np.ndarray | None] | None,
    guard: ClipGuardConfig,
) -> float:
    """Soft per-clip regression penalty vs the fixed iter-0 baseline.

    Each clip gets a hinge penalty for robust scalar regressions and aligned p95
    frame-loss regression beyond their deadbands. The objective uses the mean
    excess, plus an optional tail term over the worst clips so difficult clips
    still affect acceptance.
    """
    if baseline_metrics is None:
        return 0.0
    excesses, _, _ = _clip_excess_by_motion(
        trial_metrics, baseline_metrics, trial_frame_losses, baseline_frame_losses, guard
    )
    mean_term, tail_term, _ = _penalty_terms(excesses, guard.penalty_tail_fraction)
    return mean_term + guard.penalty_tail_weight * tail_term


def _clip_penalty_breakdown(
    trial_metrics: list[dict[str, float] | None],
    baseline_metrics: list[dict[str, float] | None] | None,
    trial_frame_losses: list[np.ndarray | None] | None,
    baseline_frame_losses: list[np.ndarray | None] | None,
    motion_names: list[str],
    guard: ClipGuardConfig,
) -> dict[str, Any]:
    """Return raw penalty terms and largest contributors for probe diagnostics."""
    empty = {
        "penalty_mean_term": 0.0, "penalty_tail_term": 0.0,
        "penalty_top_clip": None, "penalty_top_clip_excess": 0.0,
        "penalty_top_metric": None, "penalty_top_metric_excess": 0.0,
    }
    if baseline_metrics is None:
        return empty
    excesses, indices, metric_excess = _clip_excess_by_motion(
        trial_metrics, baseline_metrics, trial_frame_losses, baseline_frame_losses, guard
    )
    if not excesses:
        return empty
    arr = np.asarray(excesses, dtype=np.float64)
    mean_term, tail_term, order = _penalty_terms(excesses, guard.penalty_tail_fraction)
    worst_idx = indices[int(order[-1])]
    top_metric, top_metric_excess = (
        max(metric_excess.items(), key=lambda kv: kv[1]) if metric_excess else (None, 0.0)
    )
    return {
        "penalty_mean_term": mean_term,
        "penalty_tail_term": tail_term,
        "penalty_top_clip": motion_names[worst_idx] if worst_idx < len(motion_names) else str(worst_idx),
        "penalty_top_clip_excess": float(arr[int(order[-1])]),
        "penalty_top_metric": top_metric,
        "penalty_top_metric_excess": float(top_metric_excess),
    }


# Accel spike checks use a dataset-scale floor so unusually smooth baseline clips do
# not reject a probe for a small rise that is still normal for this motion set.
_CATASTROPHIC_ACCEL_FLOOR_FRACTION = 0.25

# A non-finite value here is treated as a failed solve, not a tradeoff.
_CATASTROPHIC_FINITE_METRICS = (
    "tracking_loss", "ik_frame_loss_p95", "ik_worst_frame_loss",
    "ik_linear_accel_p95", "ik_angular_accel_p95",
    "ik_linear_accel_max", "ik_angular_accel_max",
)


def _catastrophic_violations(
    baseline_metrics: list[dict[str, float] | None] | None,
    baseline_frame_losses: list[np.ndarray | None] | None,
    trial_metrics: list[dict[str, float] | None],
    trial_frame_losses: list[np.ndarray | None] | None,
    motion_names: list[str],
    guard: ClipGuardConfig,
) -> list[dict[str, Any]]:
    """Return penalty-mode hard rejects measured against the iter-0 baseline."""
    if baseline_metrics is None:
        return []
    trial_frame_losses = trial_frame_losses or []
    baseline_frame_losses = baseline_frame_losses or []
    accel_floor = {
        metric: _CATASTROPHIC_ACCEL_FLOOR_FRACTION
        * max((float(bm.get(metric, 0.0)) for bm in baseline_metrics if bm), default=0.0)
        for metric in ("ik_linear_accel_max", "ik_angular_accel_max")
    }
    violations: list[dict[str, Any]] = []
    for idx, (base_m, trial_m) in enumerate(zip(baseline_metrics, trial_metrics)):
        if trial_m is None:
            violations.append({
                "guard": "catastrophic_guard", "motion": motion_names[idx],
                "metric": "missing_metrics", "anchor": 0.0, "trial": float("nan"),
                "relative": float("inf"), "limit": 0.0, "gates": True,
            })
            continue
        for metric in _CATASTROPHIC_FINITE_METRICS:
            val = trial_m.get(metric)
            if val is None or not np.isfinite(float(val)):
                violations.append({
                    "guard": "catastrophic_guard", "motion": motion_names[idx],
                    "metric": f"{metric}_nonfinite", "anchor": 0.0,
                    "trial": float(val) if val is not None else float("nan"),
                    "relative": float("inf"), "limit": 0.0, "gates": True,
                })
        if base_m is None:
            continue
        for metric in ("ik_linear_accel_max", "ik_angular_accel_max"):
            base_val = float(base_m.get(metric, 0.0))
            trial_val = float(trial_m.get(metric, 0.0))
            rel = (trial_val - base_val) / max(abs(base_val), accel_floor[metric], 1e-4)
            if rel > guard.catastrophic_accel_spike_regression:
                violations.append({
                    "guard": "catastrophic_guard", "motion": motion_names[idx],
                    "metric": metric, "anchor": base_val, "trial": trial_val,
                    "relative": rel, "limit": guard.catastrophic_accel_spike_regression,
                    "gates": True,
                })
        base_fl = baseline_frame_losses[idx] if idx < len(baseline_frame_losses) else None
        trial_fl = trial_frame_losses[idx] if idx < len(trial_frame_losses) else None
        _, max_reg, max_idx, base_at_max, trial_at_max = _aligned_frame_regression(base_fl, trial_fl)
        if max_reg > guard.catastrophic_frame_loss_max_regression:
            violations.append({
                "guard": "catastrophic_guard", "motion": motion_names[idx],
                "metric": "ik_frame_loss_regression_max", "frame": max_idx,
                "anchor": base_at_max, "trial": trial_at_max, "relative": max_reg,
                "limit": guard.catastrophic_frame_loss_max_regression, "gates": True,
            })
    violations.sort(key=lambda v: v["relative"], reverse=True)
    return violations


@dataclass(frozen=True)
class _ClipGuardDecision:
    """Outcome of a single clip-policy check: blocking violations, advisory notes, and active mode."""

    active_mode: str
    blocking: list[dict[str, Any]]
    advisory: list[dict[str, Any]]

    @property
    def blocked(self) -> bool:
        """Return True when there is at least one blocking violation."""
        return bool(self.blocking)


class _ClipGuardPolicy:
    """Stateful clip policy used by the line search.

    The configured mode is fixed for diagnostics. ``active_mode`` changes only for
    ``auto``, after the gated phase has exhausted convergence patience.
    """

    def __init__(
        self,
        guard: ClipGuardConfig,
        motion_names: list[str],
        baseline_metrics: list[dict[str, float] | None],
        baseline_frame_losses: list[np.ndarray | None],
    ) -> None:
        self.guard = guard
        self.active_mode = "gate" if guard.mode == "auto" else guard.mode
        self.motion_names = motion_names
        self.baseline_metrics = [dict(m) if m is not None else None for m in baseline_metrics]
        self.baseline_frame_losses = [
            None if fl is None else np.asarray(fl, dtype=np.float64).copy()
            for fl in baseline_frame_losses
        ]
        # Gate mode also guards against drift from each clip's best accepted values.
        self.best_metrics = [dict(m) if m is not None else None for m in baseline_metrics]

    def try_enter_penalty_phase(self) -> bool:
        """Switch from gate to penalty mode if configured for auto; return True if switched."""
        if self.guard.mode != "auto" or self.active_mode != "gate":
            return False
        self.active_mode = "penalty"
        return True

    def penalty(self, outcome: _EvaluationResult, tracking_loss_scale: float) -> float:
        """Return the penalty-mode clip-regression term for the line-search objective."""
        if self.active_mode != "penalty":
            return 0.0
        return self.guard.penalty_balance * tracking_loss_scale * _clip_penalty(
            outcome.per_motion_metrics, self.baseline_metrics,
            outcome.ik_frame_loss_by_motion, self.baseline_frame_losses, self.guard,
        )

    def penalty_breakdown(self, outcome: _EvaluationResult) -> dict[str, Any]:
        """Raw mean/tail terms and largest contributors for CSV diagnostics."""
        if self.active_mode != "penalty":
            return {}
        return _clip_penalty_breakdown(
            outcome.per_motion_metrics, self.baseline_metrics,
            outcome.ik_frame_loss_by_motion, self.baseline_frame_losses,
            self.motion_names, self.guard,
        )

    def probe(self, anchor: _EvaluationResult, trial: _EvaluationResult) -> _ClipGuardDecision:
        """Return the clip-policy decision for a line-search probe."""
        if self.active_mode == "off":
            return _ClipGuardDecision(self.active_mode, [], [])
        if self.active_mode == "penalty":
            blocking = _catastrophic_violations(
                self.baseline_metrics, self.baseline_frame_losses,
                trial.per_motion_metrics, trial.ik_frame_loss_by_motion,
                self.motion_names, self.guard,
            )
            advisory = _clip_guard_violations(
                anchor.per_motion_metrics, trial.per_motion_metrics,
                anchor.ik_frame_loss_by_motion, trial.ik_frame_loss_by_motion,
                self.motion_names, self.guard,
            )
            return _ClipGuardDecision(self.active_mode, blocking, advisory)
        violations = _clip_guard_violations(
            anchor.per_motion_metrics, trial.per_motion_metrics,
            anchor.ik_frame_loss_by_motion, trial.ik_frame_loss_by_motion,
            self.motion_names, self.guard,
        )
        violations.extend(_best_clip_guard_violations(
            self.best_metrics, trial.per_motion_metrics, self.motion_names, self.guard,
        ))
        violations.sort(key=lambda v: v["relative"], reverse=True)
        return _ClipGuardDecision(
            self.active_mode,
            [v for v in violations if v["gates"]],
            [v for v in violations if not v["gates"]],
        )

    def note_accepted(self, outcome: _EvaluationResult) -> None:
        """Update per-clip best-metric tracking when a step is accepted in gate mode."""
        if self.active_mode == "gate":
            _update_best_guard_metrics(self.best_metrics, outcome.per_motion_metrics)


def _pipeline_parent_indices(pipeline: SomaRetargetingPipeline) -> np.ndarray:
    """Return the per-effector parent indices aligned to the pipeline's mapped-joints order."""
    scaler = pipeline.human_robot_scaler
    scaler_names = scaler.effector_names()
    parent_name_by_name = {
        name: scaler_names[parent_idx] if parent_idx >= 0 else None
        for name, parent_idx in zip(scaler_names, scaler.mapped_joint_parents)
    }
    pipeline_index_by_name = {name: idx for idx, name in enumerate(pipeline.mapped_joints)}
    return np.asarray(
        [
            pipeline_index_by_name.get(parent_name_by_name.get(name), -1)
            for name in pipeline.mapped_joints
        ],
        dtype=np.int32,
    )


def _apply_effector_priorities(
    effector_names: list[str],
    position_weights: np.ndarray,
    rotation_weights: np.ndarray,
    config: dict[str, Any],
) -> None:
    """Multiply per-effector position and rotation weights by config-specified multipliers."""
    priority_cfg = config.get("effector_priorities", {})
    if not priority_cfg or not bool(priority_cfg.get("enabled", True)):
        return

    index_by_name = {name: idx for idx, name in enumerate(effector_names)}
    index_by_lower_name = {name.lower(): idx for idx, name in enumerate(effector_names)}

    def apply_named_multipliers(weights: np.ndarray, key: str) -> None:
        multipliers = priority_cfg.get(key, {})
        if not isinstance(multipliers, dict):
            raise TypeError(f"effector_priorities.{key} must be an object mapping effector names to multipliers.")
        for raw_name, raw_multiplier in multipliers.items():
            name = str(raw_name)
            idx = index_by_name.get(name, index_by_lower_name.get(name.lower()))
            if idx is None:
                continue
            weights[idx] *= max(float(raw_multiplier), 0.0)

    apply_named_multipliers(position_weights, "position_weights")
    apply_named_multipliers(rotation_weights, "rotation_weights")


def _build_error_model_streamed(
    pipeline: SomaRetargetingPipeline,
    segment_lengths_per_effector: list[list[np.ndarray]],
    config: dict[str, Any],
) -> SourceAgnosticErrorModel:
    """Build error model from per-effector segment lengths collected across batches.

    Stores only scalar segment lengths (1 float per frame per non-root
    effector) rather than full target frames (7 floats per frame per effector),
    taking the median segment length per effector as its position scale.
    """
    cfg = config.get("error_model", {})
    enabled = bool(cfg.get("enabled", True))
    num_effectors = len(pipeline.mapped_joints)
    parent_indices = _pipeline_parent_indices(pipeline)

    min_position_scale = float(cfg.get("min_position_scale_m", 0.08))
    root_position_scale = float(cfg.get("root_position_scale_m", 1.0))
    rotation_scale = max(float(cfg.get("rotation_scale_rad", 0.75)), 1e-6)

    if not enabled:
        return SourceAgnosticErrorModel(
            enabled=False,
            parent_indices=parent_indices,
            position_scales=np.ones(num_effectors, dtype=np.float32),
            rotation_scales=np.ones(num_effectors, dtype=np.float32),
            position_weights=np.ones(num_effectors, dtype=np.float32),
            rotation_weights=np.ones(num_effectors, dtype=np.float32),
            normalized_rotation_weight=0.25,
            reachability_threshold=1.0,
            unreachable_weight_decay=1.0,
        )

    position_scales = np.full(num_effectors, min_position_scale, dtype=np.float32)
    for eff_idx in range(num_effectors):
        if parent_indices[eff_idx] < 0:
            continue
        parts = segment_lengths_per_effector[eff_idx]
        if parts:
            lengths = np.concatenate(parts)
            position_scales[eff_idx] = max(min_position_scale, float(np.median(lengths)))

    root_mask = parent_indices < 0
    if np.any(root_mask):
        position_scales[root_mask] = max(min_position_scale, root_position_scale)

    position_weights = np.ones(num_effectors, dtype=np.float32)
    rotation_weights = np.ones(num_effectors, dtype=np.float32)

    _apply_effector_priorities(pipeline.mapped_joints, position_weights, rotation_weights, config)

    return SourceAgnosticErrorModel(
        enabled=True,
        parent_indices=parent_indices,
        position_scales=position_scales,
        rotation_scales=np.full(num_effectors, rotation_scale, dtype=np.float32),
        position_weights=position_weights,
        rotation_weights=rotation_weights,
        normalized_rotation_weight=float(cfg.get("normalized_rotation_weight", 0.25)),
        reachability_threshold=max(float(cfg.get("reachability_threshold", 0.35)), 1e-6),
        unreachable_weight_decay=float(np.clip(cfg.get("unreachable_weight_decay", 0.2), 0.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# Config writing with rounding
# ---------------------------------------------------------------------------


def _round3(x: float) -> float:
    return round(float(x), 3)


def _build_retargeter_config(
    workspace: OptimizerConfigWorkspace,
    params: OptimizerParameters,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the retargeter config for ``params`` in memory.

    Only the IK match-table weights are tuned; everything else (incl. the scaler
    reference, kept as a path relative to the robot asset root) is taken verbatim
    from the base config, so the result is directly deployable. ``overrides`` (e.g.
    post-processing off during optimization) are applied for the per-eval pipeline
    only. The deployable output is built without them.
    """
    retargeter = copy.deepcopy(workspace.base_retargeter)
    for eff_idx, joint_name in enumerate(workspace.effector_names):
        base_t_weight, base_r_weight = workspace.base_ik_weights[joint_name]
        retargeter["ik_match_table"][joint_name]["t_weight"] = _round3(
            base_t_weight * float(params.position_weight_scales[eff_idx])
        )
        retargeter["ik_match_table"][joint_name]["r_weight"] = _round3(
            base_r_weight * float(params.rotation_weight_scales[eff_idx])
        )
    retargeter["human_robot_scaler_config"] = pathlib.PurePosixPath(
        workspace.base_scaler_rel
    ).as_posix()
    for key, value in (overrides or {}).items():
        retargeter[key] = value
    return retargeter


# ---------------------------------------------------------------------------
# Plotting and CSV diagnostics
# ---------------------------------------------------------------------------


def _finite_or_nan(value: Any) -> float:
    """Convert *value* to float, returning NaN if conversion fails or the result is non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _norm_from_values(values: list[float]) -> float:
    """Return the L2 norm of the finite values in *values*, or NaN if none are finite."""
    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return float("nan")
    return float(np.linalg.norm(np.asarray(finite_values, dtype=np.float32)))


def _first_finite_positive(values: list[float]) -> float:
    """Return the first finite positive value in *values*, or NaN if none exists."""
    for v in values:
        if np.isfinite(v) and v > 1e-8:
            return v
    return float("nan")


def _clip_guard_mode_transitions(history: list[dict[str, Any]]) -> list[tuple[float, str, str]]:
    """Extract (iteration, from_mode, to_mode) tuples where the clip-guard mode changed."""
    transitions: list[tuple[float, str, str]] = []
    previous_mode: str | None = None
    for entry in history:
        update = entry.get("parameter_update", {})
        mode = update.get("clip_guard_active_mode")
        if not isinstance(mode, str):
            continue
        if previous_mode is not None and mode != previous_mode:
            iteration = float(entry.get("plot_iteration", int(entry["iteration"]) + 1))
            transitions.append((iteration, previous_mode, mode))
        previous_mode = mode
    return transitions


def _add_convergence_metrics(history: list[dict[str, Any]]) -> None:
    """Annotate each history entry in-place with loss-improvement and weight-update convergence metrics."""
    if not history:
        return

    ik_weight_update_values: list[float] = []

    for entry in history:
        update = entry.setdefault("parameter_update", {})
        position_weight_update = _finite_or_nan(update.get("position_weight_update_norm"))
        rotation_weight_update = _finite_or_nan(update.get("rotation_weight_update_norm"))

        ik_weight_update = _norm_from_values([position_weight_update, rotation_weight_update])
        update["ik_weight_update_norm"] = ik_weight_update
        ik_weight_update_values.append(ik_weight_update)

    first_ik = _first_finite_positive(ik_weight_update_values)

    for entry, iku in zip(history, ik_weight_update_values):
        update = entry["parameter_update"]
        update["ik_weight_update_relative_to_first"] = (
            iku / first_ik if np.isfinite(first_ik) and np.isfinite(iku) else float("nan")
        )

    previous_loss = float("nan")
    previous_objective = float("nan")
    previous_ik_weight_update = float("nan")
    previous_clip_guard_mode: str | None = None
    for entry in history:
        aggregate = entry.setdefault("aggregate_metrics", {})
        update = entry.get("parameter_update", {})
        current_clip_guard_mode = update.get("clip_guard_active_mode")
        mode_changed = (
            isinstance(previous_clip_guard_mode, str)
            and isinstance(current_clip_guard_mode, str)
            and current_clip_guard_mode != previous_clip_guard_mode
        )
        current_loss = _metric_value(aggregate, "tracking_loss")
        if np.isfinite(previous_loss) and np.isfinite(current_loss):
            improvement = previous_loss - current_loss
            aggregate["tracking_loss_improvement"] = improvement
            aggregate["tracking_loss_improvement_percent"] = (
                100.0 * improvement / abs(previous_loss) if abs(previous_loss) > 1e-8 else float("nan")
            )
            aggregate["tracking_loss_improvement_per_previous_ik_weight_update_norm"] = (
                improvement / previous_ik_weight_update
                if np.isfinite(previous_ik_weight_update) and previous_ik_weight_update > 1e-8 else float("nan")
            )
        else:
            aggregate["tracking_loss_improvement"] = float("nan")
            aggregate["tracking_loss_improvement_percent"] = float("nan")
            aggregate["tracking_loss_improvement_per_previous_ik_weight_update_norm"] = float("nan")

        # Objective improvement is meaningful only while the objective definition is stable.
        current_objective = _finite_or_nan(aggregate.get("objective_loss"))
        if mode_changed:
            aggregate["objective_loss_improvement"] = float("nan")
            aggregate["objective_loss_improvement_percent"] = float("nan")
            aggregate["objective_loss_improvement_suppressed"] = 1.0
        elif np.isfinite(previous_objective) and np.isfinite(current_objective):
            obj_improvement = previous_objective - current_objective
            aggregate["objective_loss_improvement"] = obj_improvement
            aggregate["objective_loss_improvement_percent"] = (
                100.0 * obj_improvement / abs(previous_objective) if abs(previous_objective) > 1e-8 else float("nan")
            )
            aggregate["objective_loss_improvement_suppressed"] = 0.0
        else:
            aggregate["objective_loss_improvement"] = float("nan")
            aggregate["objective_loss_improvement_percent"] = float("nan")
            aggregate["objective_loss_improvement_suppressed"] = 0.0

        previous_loss = current_loss
        previous_objective = current_objective
        previous_ik_weight_update = _finite_or_nan(update.get("ik_weight_update_norm"))
        if isinstance(current_clip_guard_mode, str):
            previous_clip_guard_mode = current_clip_guard_mode


def _write_iteration_metrics_csv(
    history: list[dict[str, Any]], output_dir: pathlib.Path
) -> pathlib.Path | None:
    """Write per-iteration aggregate and parameter-update metrics to a flat CSV."""
    if not history:
        return None
    _add_convergence_metrics(history)
    aggregate_keys = sorted({k for e in history for k in e.get("aggregate_metrics", {})})
    update_keys = sorted({k for e in history for k in e.get("parameter_update", {})})
    fieldnames = ["iteration", *aggregate_keys, *[f"parameter_update.{k}" for k in update_keys]]
    path = output_dir / "optimizer_iteration_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in history:
            row: dict[str, Any] = {
                "iteration": entry.get("plot_iteration", int(entry["iteration"]) + 1),
            }
            row.update(entry.get("aggregate_metrics", {}))
            for k, v in entry.get("parameter_update", {}).items():
                row[f"parameter_update.{k}"] = v
            writer.writerow(row)
    return path


def _draw_dashboard(
    history: list[dict[str, Any]],
    fig: Any,
    axes: Any,
    title: str,
) -> None:
    """Draw the 6-panel optimization dashboard onto a 3x2 matplotlib grid.

    The first panel shows the accepted objective and its tracking/smoothness terms.
    The remaining panels show IK weight updates, solved-IK tracking, acceleration,
    and per-frame loss signals used by the clip guard.
    """
    _add_convergence_metrics(history)
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    iterations = np.asarray(
        [e.get("plot_iteration", int(e["iteration"]) + 1) for e in history], dtype=float
    )
    ax = axes.ravel()
    clip_guard_transitions = _clip_guard_mode_transitions(history)
    clip_guard_transition_style = {
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 1.2,
        "alpha": 0.65,
    }

    def series(key, section="aggregate_metrics"):
        if section == "aggregate_metrics":
            return np.asarray([_metric_value(e.get(section, {}), key) for e in history], dtype=float)
        return np.asarray([e.get(section, {}).get(key, np.nan) for e in history], dtype=float)

    def dlabel(label, ys, allow=True):
        if not allow:
            return label
        finite = ys[np.isfinite(ys)]
        if len(finite) < 2 or abs(float(finite[0])) < 1e-8:
            return label
        return f"{label} ({100.0 * (float(finite[-1]) - float(finite[0])) / abs(float(finite[0])):+.1f}%)"

    def line(a, key, label, *, section="aggregate_metrics", allow=True, color=None):
        ys = series(key, section)
        if np.all(np.isnan(ys)):
            return
        a.plot(iterations, ys, marker="o", linewidth=2.0, color=color, label=dlabel(label, ys, allow))

    def twin_legend(a_primary, a_twin):
        h1, l1 = a_primary.get_legend_handles_labels()
        h2, l2 = a_twin.get_legend_handles_labels()
        if h1 or h2:
            a_primary.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    def format_iteration_tick(value, _position):
        rounded = round(value)
        if abs(value - rounded) > 1e-6:
            return ""
        return str(int(rounded))

    a0 = ax[0]
    a0.set_title("Optimization Loss and Tracking")
    a0.set_ylabel("loss")
    a0t = a0.twinx()
    improvement = series("objective_loss_improvement_percent")
    if not np.all(np.isnan(improvement)):
        a0t.bar(iterations, np.nan_to_num(improvement, nan=0.0), width=0.6,
                color="tab:blue", alpha=0.35, label="optimization loss improvement %/iter")
        a0t.axhline(0.0, color="black", linewidth=0.8, alpha=0.3)
        a0t.set_ylabel("optimization loss improvement %")
    line(a0, "objective_loss", "optimization loss", color="tab:orange")
    line(a0, "tracking_loss", "tracking loss", color="tab:green")
    line(a0, "objective_smoothness_term", "smoothness penalty", color="tab:purple")
    if np.any(series("objective_clip_penalty_term") > 0.0):
        line(a0, "objective_clip_penalty_term", "clip penalty", color="tab:red")
    for transition_idx, from_mode, to_mode in clip_guard_transitions:
        label = f"clip guard {from_mode}->{to_mode}"
        a0.axvline(
            transition_idx,
            label=label,
            **clip_guard_transition_style,
        )
    a0.set_zorder(a0t.get_zorder() + 1)  # draw the lines above the bars...
    a0.patch.set_visible(False)          # ...without an opaque background hiding them
    twin_legend(a0, a0t)

    ax[1].set_title("IK Weight Update")
    line(ax[1], "ik_weight_update_norm", "IK weight update", section="parameter_update", allow=False)
    ax[1].set_ylabel("update norm")

    ax[2].set_title("Position Error")
    line(ax[2], "ik_position_rmse_m", "all effectors RMSE")
    line(ax[2], "ik_root_position_rmse_m", "root RMSE")
    ax[2].set_ylabel("meters")

    ax[3].set_title("Rotation Error")
    line(ax[3], "ik_orientation_rmse_rad", "all effectors RMSE")
    line(ax[3], "ik_root_orientation_rmse_rad", "root RMSE")
    ax[3].set_ylabel("radians")

    a4 = ax[4]
    a4.set_title("Motion Smoothness")
    line(a4, "ik_linear_accel_max_max", "linear accel max", color="tab:blue")
    a4.set_ylabel("linear frame accel")
    a4t = a4.twinx()
    line(a4t, "ik_angular_accel_max_max", "angular accel max", color="tab:red")
    a4t.set_ylabel("angular frame accel")

    ax[5].set_title("Per-Frame Tracking Loss")
    line(ax[5], "tracking_loss", "mean")
    line(ax[5], "ik_frame_loss_p95", "p95 frame (clip mean)")
    line(ax[5], "ik_worst_frame_loss_max", "worst frame (global)")
    ax[5].set_ylabel("frame loss")

    for a in ax:
        a.set_xlabel("iteration")
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.xaxis.set_major_formatter(FuncFormatter(format_iteration_tick))
        a.grid(True, alpha=0.25)
    for transition_idx, from_mode, to_mode in clip_guard_transitions:
        label = f"clip guard {from_mode}->{to_mode}"
        for a in ax[1:]:
            a.axvline(
                transition_idx,
                label=label,
                **clip_guard_transition_style,
            )
    for a in (ax[1], ax[2], ax[3], ax[5]):
        handles, _labels = a.get_legend_handles_labels()
        if handles:
            a.legend(loc="best", fontsize=8)
    twin_legend(a4, a4t)
    fig.suptitle(title, fontsize=14)


def _render_dashboard_png(
    history: list[dict[str, Any]],
    png_path: pathlib.Path,
    *,
    title: str,
    plot: PlotConfig | None = None,
) -> pathlib.Path | None:
    """Render the dashboard to ``png_path`` with matplotlib (headless Agg backend, so
    it works on a server with no display).  Returns the path, or None if matplotlib
    is unavailable (a warning is logged) or there is no history.
    """
    if not history:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - matplotlib is an optional render-time dependency
        _log(
            "matplotlib not found; skipping plot rendering (CSV/JSON diagnostics are "
            "still written, plot later with --plot once matplotlib is installed). "
            "Install with: pip install matplotlib"
        )
        return None
    plot = plot or PlotConfig()
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.0), constrained_layout=True)
    _draw_dashboard(history, fig, axes, title)
    fig.savefig(png_path, dpi=plot.dpi)
    plt.close(fig)
    return png_path


def _history_from_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    """Rebuild the iteration history from the flattened metrics CSV.

    Inverse of ``_write_iteration_metrics_csv``: ``parameter_update.*`` columns go
    back under ``parameter_update``, everything else under ``aggregate_metrics``.
    """
    history: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aggregate: dict[str, float] = {}
            update: dict[str, Any] = {}
            for key, value in row.items():
                if key in (None, "iteration"):
                    continue
                if key.startswith("parameter_update.") and key.endswith("_mode"):
                    update[key[len("parameter_update."):]] = value
                    continue
                try:
                    fval = float(value) if value not in (None, "") else float("nan")
                except (TypeError, ValueError):
                    continue
                if key.startswith("parameter_update."):
                    update[key[len("parameter_update."):]] = fval
                else:
                    aggregate[key] = fval
            try:
                iteration = int(float(row.get("iteration", len(history) + 1)))
            except (TypeError, ValueError):
                iteration = len(history) + 1
            history.append({
                "iteration": iteration - 1,
                "plot_iteration": iteration,
                "aggregate_metrics": aggregate,
                "parameter_update": update,
            })
    return history


def render_dashboard_from_file(
    source: str | pathlib.Path,
    png_path: str | pathlib.Path | None = None,
    *,
    title: str | None = None,
    plot: PlotConfig | None = None,
) -> pathlib.Path | None:
    """Render the optimization dashboard PNG from a serialized run.

    ``source`` is either a ``optimizer_metrics.json`` (preferred because it
    holds the exact per-iteration ``history`` at full precision) or a
    ``optimizer_iteration_metrics.csv``.  This decouples plotting from the
    run, so a job that didn't render (e.g. matplotlib not installed at run time)
    can be plotted afterwards.  Returns the PNG path, or None if there is no
    history or matplotlib is unavailable.
    """
    source = pathlib.Path(source)
    if source.suffix.lower() == ".csv":
        history = _history_from_csv(source)
    else:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            history = data.get("history", [])
            if title is None:
                target = data.get("base_target_type") or data.get("target_type")
                if target:
                    title = f"{str(target).replace('_', ' ').title()} IK Weight Optimization"
        else:
            history = data
    if not history:
        _log(f"no iteration history found in {source}")
        return None
    if png_path is None:
        png_path = source.with_name("optimizer_dashboard.png")
    return _render_dashboard_png(
        history, pathlib.Path(png_path), title=title or "IK Weight Optimization",
        plot=plot,
    )


# ---------------------------------------------------------------------------
# IK weight optimization loop
# ---------------------------------------------------------------------------


def _load_bvh_batch(
    bvh_paths: list[pathlib.Path],
    skeleton,
    batch_indices: list[int],
    cache: dict[int, Any] | None = None,
) -> list:
    """Load (or return cached) animations for the given global motion indices."""
    result = []
    for idx in batch_indices:
        if cache is not None and idx in cache:
            result.append(cache[idx])
        else:
            anim = bvh_utils.load_bvh(str(bvh_paths[idx]), skeleton)[1]
            if cache is not None:
                cache[idx] = anim
            result.append(anim)
    return result


def _setup_pipeline_and_evaluators(
    skeleton,
    source_type: str,
    target_type: str,
    retarget_config: dict[str, Any],
    fk_frame_batch_size: int,
) -> tuple[SomaRetargetingPipeline, BodyEvaluator, int]:
    """Build a retargeting pipeline and its FK body evaluator for one evaluation pass."""
    pipeline = _make_pipeline(skeleton, source_type, target_type, retarget_config)
    evaluator = BodyEvaluator(pipeline, fk_frame_batch_size=fk_frame_batch_size)
    num_effectors = len(pipeline.mapped_joints)
    return pipeline, evaluator, num_effectors


def _retarget_batch(
    pipeline: SomaRetargetingPipeline,
    bvh_paths: list[pathlib.Path],
    skeleton,
    batch_gi: list[int],
    root_tx,
    anim_cache: dict[int, Any] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Retarget a batch of motions and return solved coords and trimmed target frames."""
    batch_anims = _load_bvh_batch(bvh_paths, skeleton, batch_gi, cache=anim_cache)
    _prepare_pipeline_targets(pipeline, batch_anims, root_tx)
    buffers = pipeline.execute()
    reference_q_batch = _buffers_to_arrays(buffers)
    target_frames_batch = _targets_after_pipeline_init(pipeline)
    return reference_q_batch, target_frames_batch


def _build_multi_batch_error_model(
    pipeline: SomaRetargetingPipeline,
    bvh_paths: list[pathlib.Path],
    skeleton,
    batch_ranges: list[list[int]],
    root_tx,
    config: dict[str, Any],
    anim_cache: dict[int, Any] | None = None,
) -> SourceAgnosticErrorModel:
    """Build the source-agnostic error model by streaming segment lengths over all motion batches."""
    num_effectors = len(pipeline.mapped_joints)
    parent_indices = _pipeline_parent_indices(pipeline)
    seg_per_eff: list[list[np.ndarray]] = [[] for _ in range(num_effectors)]
    for batch_gi in batch_ranges:
        em_anims = _load_bvh_batch(bvh_paths, skeleton, batch_gi, cache=anim_cache)
        _prepare_pipeline_targets(pipeline, em_anims, root_tx)
        for tf in _targets_after_pipeline_init(pipeline):
            tpos = np.asarray(tf[:, :, 0:3], dtype=np.float32)
            for eff_idx in range(num_effectors):
                pi = int(parent_indices[eff_idx])
                if pi < 0:
                    continue
                seg_per_eff[eff_idx].append(
                    np.linalg.norm(tpos[:, eff_idx] - tpos[:, pi], axis=-1)
                )
        del em_anims
    return _build_error_model_streamed(pipeline, seg_per_eff, config)


@dataclass(frozen=True)
class _ObjectiveBreakdown:
    """Decomposed optimization objective for one parameter evaluation."""

    objective_loss: float
    normalized_smoothness_loss: float
    smoothness_term: float
    clip_penalty_term: float


@dataclass
class _EvaluationResult:
    """Everything produced by evaluating one parameter set.

    Holds the scalar tracking loss used by the line search plus the artifacts
    needed to record history and propose the next step.  Returned both for the
    per-iteration anchor and for every line-search probe.
    """

    tracking_loss: float
    aggregate: dict[str, float]
    per_motion_metrics: list[dict[str, float] | None]
    # Transient aligned-frame guard input; not serialized into history.
    ik_frame_loss_by_motion: list[np.ndarray | None]
    acc: _TrackingErrorAccumulators
    error_model: SourceAgnosticErrorModel
    mapped_joints: list[str]


def _format_guard_violation(v: dict[str, Any]) -> str:
    """Format a single guard violation dict into a human-readable one-line string."""
    frame = f" frame={int(v['frame'])}" if "frame" in v else ""
    baseline = float(v["anchor"] if "anchor" in v else v["baseline"])
    return (
        f"{v['motion']} {v['metric']}{frame} "
        f"{baseline:.4f}->{v['trial']:.4f} "
        f"(+{100.0 * v['relative']:.1f}% > +{100.0 * v['limit']:.1f}%)"
    )


def _guard_note_log_lines(
    advisory: list[dict[str, Any]],
    *,
    scale: float,
    clip_guard: ClipGuardConfig,
) -> list[str]:
    """Format advisory (non-blocking) guard violations into log lines."""
    if not advisory:
        return []
    return [
        "GUARD NOTE(S):",
        *[
            f"    {_format_guard_violation(v)} scale={scale:.4f}"
            for v in advisory[:clip_guard.max_logged_violations]
        ],
    ]


def _log_initial_state(outcome: _EvaluationResult, elapsed: str) -> None:
    """Log the baseline metrics before any weight updates."""
    _log_block(
        "INITIAL STATE",
        f"    tracking_loss={outcome.tracking_loss:.6f}",
        f"    objective_loss={outcome.aggregate['objective_loss']:.6f}",
        f"    ik_smoothness_loss={outcome.aggregate['ik_smoothness_loss']:.6f}",
        f"    worst_frame_loss={outcome.aggregate['ik_worst_frame_loss_max']:.4f}",
        f"    ik_pos_rmse={outcome.aggregate['ik_position_rmse_m']:.4f}m",
        f"    ik_rot_rmse={outcome.aggregate['ik_orientation_rmse_rad']:.4f}rad",
        f"    reach={outcome.aggregate['reachability_mean']:.3f}",
        f"    elapsed={elapsed}",
    )


def _log_accept(
    iter_idx: int,
    total_iters: int,
    outcome: _EvaluationResult,
    stats: dict[str, float],
    *,
    scale: float,
    elapsed: str,
) -> None:
    """Log a summary block when an iteration's proposed update is accepted."""
    _log_block(
        f"ITERATION [{iter_idx}/{total_iters}] ACCEPT",
        f"    tracking_loss={outcome.tracking_loss:.6f}",
        f"    objective_loss={outcome.aggregate['objective_loss']:.6f}",
        f"    ik_smoothness_loss={outcome.aggregate['ik_smoothness_loss']:.6f}",
        f"    ik_pos_rmse={outcome.aggregate['ik_position_rmse_m']:.4f}m",
        f"    ik_rot_rmse={outcome.aggregate['ik_orientation_rmse_rad']:.4f}rad",
        f"    reach={outcome.aggregate['reachability_mean']:.3f}",
        f"    d_w={stats['position_weight_update_norm']:.5f}",
        f"    scale={scale:.4f}",
        f"    elapsed={elapsed}",
    )


def _log_accept_guard_notes(
    iter_idx: int,
    total_iters: int,
    advisory: list[dict[str, Any]],
    *,
    scale: float,
    clip_guard: ClipGuardConfig,
) -> None:
    """Log advisory guard notes that accompanied an accepted update."""
    if advisory:
        _log_block(
            f"ITERATION [{iter_idx}/{total_iters}] ACCEPT with non-blocking guard note(s)",
            *_guard_note_log_lines(advisory, scale=scale, clip_guard=clip_guard),
        )


def _log_reject_probe(
    iter_idx: int,
    total_iters: int,
    probe_idx: int,
    probe_count: int,
    *,
    scale: float,
    clip_guard: ClipGuardConfig,
    guard_label: str | None = None,
    blocking: list[dict[str, Any]] | None = None,
    objective_message: str | None = None,
    advisory: list[dict[str, Any]] | None = None,
) -> None:
    """Log a summary block when a line-search probe is rejected."""
    lines = [f"ITERATION [{iter_idx}/{total_iters}] REJECT - PROBE [{probe_idx}/{probe_count}]:"]
    advisory = advisory or []
    if blocking:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for violation in blocking:
            grouped.setdefault(str(violation.get("guard", guard_label or "clip_guard")), []).append(violation)
        for label, violations in grouped.items():
            lines.append(f"    VIOLATION(S) OF '{label}':")
            lines.extend(
                f"        {rank} - {_format_guard_violation(v)}"
                for rank, v in enumerate(violations[:clip_guard.max_logged_violations], start=1)
            )
        if not advisory:
            lines.append(f"    scale={scale:.4f}")
    elif objective_message is not None:
        lines.append(f"    {objective_message} scale={scale:.4f}")
    lines.extend(_guard_note_log_lines(advisory, scale=scale, clip_guard=clip_guard))
    _log_block(*lines)


def _log_hold(
    iter_idx: int,
    total_iters: int,
    outcome: _EvaluationResult,
    *,
    no_improve: int,
    violation_count: int,
    elapsed: str,
) -> None:
    """Log a summary block when all line-search probes were rejected and the anchor is held."""
    _log_block(
        f"ITERATION [{iter_idx}/{total_iters}] HOLD",
        f"    no_improve={no_improve}",
        f"    clip_guard_violations={violation_count}",
        f"    tracking_loss={outcome.tracking_loss:.6f}",
        f"    objective_loss={outcome.aggregate['objective_loss']:.6f}",
        f"    ik_smoothness_loss={outcome.aggregate['ik_smoothness_loss']:.6f}",
        f"    ik_pos_rmse={outcome.aggregate['ik_position_rmse_m']:.4f}m",
        f"    ik_rot_rmse={outcome.aggregate['ik_orientation_rmse_rad']:.4f}rad",
        f"    elapsed={elapsed}",
    )


def _probe_record(
    iteration: int,
    probe: int,
    scale: float,
    clip_guard_mode: str,
    clip_guard_active_mode: str,
    trial: _EvaluationResult,
    trial_obj: float,
    anchor_obj: float,
    smoothness_term: float,
    clip_penalty_term: float,
    accepted: bool,
    blocking: list[dict[str, Any]],
    penalty_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one CSV row for a line-search probe.

    The row keeps the configured and active guard mode, objective components,
    rejection reason, and strongest blocking guard violation when one exists.
    Penalty-mode rows also include the raw penalty breakdown.
    """
    top = blocking[0] if blocking else {}
    record = {
        "iteration": iteration,
        "probe": probe,
        "scale": float(scale),
        "clip_guard_mode": clip_guard_mode,
        "clip_guard_active_mode": clip_guard_active_mode,
        "accepted": int(accepted),
        "reject_reason": "accepted" if accepted else ("guard" if blocking else "objective_not_improved"),
        "objective_loss": float(trial_obj),
        "anchor_objective": float(anchor_obj),
        "tracking_loss": float(trial.tracking_loss),
        "smoothness_term": float(smoothness_term),
        "clip_penalty_term": float(clip_penalty_term),
        "top_guard": top.get("guard"),
        "top_metric": top.get("metric"),
        "top_motion": top.get("motion"),
        "top_frame": top.get("frame"),
        "top_anchor": top.get("anchor", top.get("baseline")),
        "top_trial": top.get("trial"),
        "top_relative": top.get("relative"),
        "top_limit": top.get("limit"),
    }
    if penalty_breakdown:
        record.update(penalty_breakdown)
    return record


def _write_probe_diagnostics_csv(
    records: list[dict[str, Any]], output_dir: pathlib.Path
) -> pathlib.Path | None:
    """Write every line-search probe (across all iterations) to a flat CSV."""
    if not records:
        return None
    fieldnames = [
        "iteration", "probe", "scale", "clip_guard_mode", "clip_guard_active_mode",
        "accepted", "reject_reason", "objective_loss", "anchor_objective",
        "tracking_loss", "smoothness_term", "clip_penalty_term",
        "penalty_mean_term", "penalty_tail_term", "penalty_top_clip",
        "penalty_top_clip_excess", "penalty_top_metric",
        "penalty_top_metric_excess", "top_guard", "top_metric",
        "top_motion", "top_frame", "top_anchor", "top_trial",
        "top_relative", "top_limit",
    ]
    path = output_dir / "optimizer_probe_diagnostics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def _finalize_run(
    *,
    workspace: OptimizerConfigWorkspace,
    params: OptimizerParameters,
    history: list[dict[str, Any]],
    artifact_dir: pathlib.Path,
    bvh_paths: list[pathlib.Path],
    source_type: str,
    target_type: str,
    final_effectors: list[str],
    symmetry_pairs: list[SymmetryPair],
    plot_cfg: PlotConfig,
    plot_title: str,
    dashboard_png: pathlib.Path,
) -> dict[str, Any]:
    """Write the run artifacts (optimized retargeter config, iteration CSV, dashboard
    PNG, motion-ranking CSV, metrics JSON) and return the result dict."""
    # The deployable config excludes eval-time overrides.
    deployable = _build_retargeter_config(workspace, params)
    with open(workspace.output_retargeter_path, "w") as f:
        json.dump(deployable, f, indent=4)

    artifact_paths: dict[str, str] = {}
    if plot_cfg.enabled:
        csv_path = _write_iteration_metrics_csv(history, artifact_dir)
        if csv_path is not None:
            artifact_paths["iteration_metrics_csv"] = str(csv_path)
        plot_path = _render_dashboard_png(
            history,
            dashboard_png,
            title=plot_title,
            plot=plot_cfg,
        )
        if plot_path is not None:
            artifact_paths["dashboard_plot"] = str(plot_path)
        if artifact_paths:
            _log_block(
                "SAVED DIAGNOSTICS:",
                *[f"    {key}={value}" for key, value in artifact_paths.items()],
            )

    motion_improvement_ranking = _rank_motions_by_tracking_loss_improvement(
        history, bvh_paths
    )
    ranking_path = _write_motion_improvement_ranking_csv(
        motion_improvement_ranking, artifact_dir
    )
    if ranking_path is not None:
        artifact_paths["motion_improvement_ranking_csv"] = str(ranking_path)
        improved = sum(
            r.get("tracking_loss_status") == "improved"
            for r in motion_improvement_ranking
        )
        regressed = sum(
            r.get("tracking_loss_status") == "regressed"
            for r in motion_improvement_ranking
        )
        unchanged = len(motion_improvement_ranking) - improved - regressed
        _log_block(
            "SAVED MOTION RANKING:",
            f"    {ranking_path}",
        )
        _log_block(
            "TRACKING LOSS:",
            f"    improved={improved}",
            f"    regressed={regressed}",
            f"    unchanged={unchanged}",
        )
        if improved:
            top_improvements = [
                row for row in motion_improvement_ranking
                if row.get("tracking_loss_status") == "improved"
            ][:5]
            _log_block(
                "TOP 5 IMPROVEMENTS:",
                *[
                    f"    {rank} - {row['motion_name']} "
                    f"d_tracking={row.get('improvement_tracking_loss', 0.0):.6f} "
                    f"d_ik_pos={row.get('improvement_ik_position_rmse_m', 0.0):.4f}m"
                    for rank, row in enumerate(top_improvements, start=1)
                ],
            )
        if regressed:
            bottom_regressions = [
                row for row in motion_improvement_ranking
                if row.get("tracking_loss_status") == "regressed"
            ][-5:]
            _log_block(
                "BOTTOM 5 REGRESSIONS:",
                *[
                    f"    {int(row['rank'])} - {row['motion_name']} "
                    f"d_tracking={row.get('improvement_tracking_loss', 0.0):.6f} "
                    f"d_ik_pos={row.get('improvement_ik_position_rmse_m', 0.0):.4f}m"
                    for row in bottom_regressions
                ],
            )

    result = {
        "source_type": source_type,
        "target_type": workspace.target_type,
        "base_target_type": target_type,
        "optimized_retargeter_config": str(workspace.output_retargeter_path),
        "motions": [str(p) for p in bvh_paths],
        "effectors": final_effectors,
        "symmetry": {
            "enabled": bool(symmetry_pairs),
            "pairs": [p.to_jsonable() for p in symmetry_pairs],
        },
        "history": history,
        "motion_improvement_ranking": motion_improvement_ranking,
        "final_parameters": params.to_jsonable(),
        "artifacts": artifact_paths,
    }
    with open(artifact_dir / "optimizer_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    _log_block(
        "SAVED RESULTS:",
        f"    {artifact_dir}",
    )

    return result



def run_optimizer(config: dict[str, Any], config_dir: pathlib.Path) -> dict[str, Any]:
    """Run the full IK weight optimization loop and return the result dict.

    Loads motions, builds the error model, then iterates outer_iterations times:
    propose a weight update, run a line-search with clip-guard checking, accept or hold,
    and write diagnostics on convergence or early stop.
    """
    _validate_config(config)
    _register_extra_robot_paths(config, config_dir)
    run_t0 = time.perf_counter()
    source_type = config.get("source_type", "soma")
    target_type = config["target_type"]
    plot_title = f"{target_type.replace('_', ' ').title()} IK Weight Optimization"
    output_folder = _resolve_path(config_dir, config["output_folder"])
    output_dir = output_folder / f"optimized_{target_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    bvh_paths = _gather_bvh_files(config, config_dir)
    num_motions = len(bvh_paths)
    _log(f"using {num_motions} BVH motion(s)")

    load_t0 = time.perf_counter()
    skeleton, _first_anim = bvh_utils.load_bvh(str(bvh_paths[0]))
    del _first_anim
    _log(f"loaded skeleton in {_elapsed(load_t0)}")

    root_tx = SpaceConverter(FacingDirectionType.MAYA).transform(wp.transform_identity())
    workspace = OptimizerConfigWorkspace(
        source_type,
        target_type,
        output_dir,
    )
    artifact_dir = output_dir
    params = OptimizerParameters.identity(len(workspace.effector_names))
    target_symmetric = bool(config.get("is_target_symmetric", True))
    symmetry_pairs = workspace.build_symmetry_pairs() if target_symmetric else []
    if target_symmetric:
        pair_labels = ", ".join(
            f"{p.left_name}/{p.right_name}" for p in symmetry_pairs
        )
        _log(
            f"target symmetry constraint enabled: {len(symmetry_pairs)} mirrored pair(s)"
            + (f" ({pair_labels})" if pair_labels else "")
        )

    anim_cache: dict[int, Any] = {}
    fk_frame_batch_size = max(1, int(config.get("fk_frame_batch_size", 100)))
    motion_batch_size = max(1, int(config.get("motion_batch_size", 1000)))
    batch_ranges = [
        list(range(b, min(b + motion_batch_size, num_motions)))
        for b in range(0, num_motions, motion_batch_size)
    ]

    plot_cfg = PlotConfig.from_config(config)
    dashboard_png = artifact_dir / "optimizer_dashboard.png"

    outer_iterations = max(1, int(config.get("outer_iterations", 20)))
    history: list[dict[str, Any]] = []

    ik_update = IkWeightUpdateConfig.from_config(config)
    line_search = LineSearchConfig.from_config(config)
    clip_guard = ClipGuardConfig.from_config(config)
    objective = ObjectiveConfig.from_config(config)
    convergence_patience = int(config.get("convergence_patience", 1))
    # Optional per-probe diagnostics explain line-search accept/reject decisions.
    save_probe_diagnostics = bool(config.get("save_probe_diagnostics", False))
    motion_names = [p.name for p in bvh_paths]
    initial_tracking_loss_scale = 1.0
    smoothness_denom = 1.0
    clip_policy: _ClipGuardPolicy

    if len(batch_ranges) > 1:
        _log(f"motion batching: {len(batch_ranges)} batches of up to {motion_batch_size} motions")

    def _build_error_model() -> SourceAgnosticErrorModel:
        """Build the static source-agnostic error model (once).

        It depends only on target geometry, specifically per-effector segment lengths, which
        is independent of the IK weights, so it never changes during optimization
        and is built a single time before the loop.
        """
        retarget_config = _build_retargeter_config(
            workspace, params, config.get("retargeter_overrides")
        )
        em_pipeline = _make_pipeline(
            skeleton, source_type, workspace.target_type, retarget_config
        )
        return _build_multi_batch_error_model(
            em_pipeline, bvh_paths, skeleton, batch_ranges,
            root_tx, config,
            anim_cache=anim_cache,
        )

    def _evaluate(
        eval_params: OptimizerParameters,
        error_model: SourceAgnosticErrorModel,
    ) -> _EvaluationResult:
        """Retarget, FK(reference_q), then compute solved-IK metrics and rebalance accumulators.

        Builds the retargeter config for ``eval_params``, retargets each motion batch,
        FKs the solved ``reference_q``, and scores the solved-IK objective. Used for
        both the anchor and every line-search probe.
        """
        retarget_config = _build_retargeter_config(
            workspace, eval_params, config.get("retargeter_overrides")
        )
        pipeline, evaluator, num_effectors = _setup_pipeline_and_evaluators(
            skeleton, source_type, workspace.target_type, retarget_config, fk_frame_batch_size
        )
        acc = _TrackingErrorAccumulators.zeros(num_effectors)
        per_motion_metrics: list[dict[str, float] | None] = [None] * num_motions
        frame_loss_by_motion: list[np.ndarray | None] = [None] * num_motions

        for batch_gi in batch_ranges:
            reference_q_batch, target_frames_batch = _retarget_batch(
                pipeline, bvh_paths, skeleton, batch_gi,
                root_tx,
                anim_cache=anim_cache,
            )
            ref_poses_batch = evaluator.evaluate(reference_q_batch)
            for local_idx, global_m in enumerate(batch_gi):
                ref_poses = ref_poses_batch[local_idx]
                ik_mask = np.ones(len(reference_q_batch[local_idx]), dtype=bool)
                metrics, frame_loss = _compute_metrics_from_poses(
                    ref_poses,
                    target_frames_batch[local_idx],
                    ik_mask,
                    error_model,
                    accumulators=acc,
                    unreachable_weight_decay=error_model.unreachable_weight_decay,
                    hard_clip_emphasis=ik_update.hard_clip_emphasis,
                )
                per_motion_metrics[global_m] = metrics
                frame_loss_by_motion[global_m] = frame_loss

        aggregate = _aggregate_metrics(per_motion_metrics)
        mean_loss = _metric_value(aggregate, "tracking_loss")
        return _EvaluationResult(
            tracking_loss=mean_loss,
            aggregate=aggregate,
            per_motion_metrics=per_motion_metrics,
            ik_frame_loss_by_motion=frame_loss_by_motion,
            acc=acc,
            error_model=error_model,
            mapped_joints=list(pipeline.mapped_joints),
        )

    def _objective_breakdown(outcome: _EvaluationResult) -> _ObjectiveBreakdown:
        """Return the objective and its smoothness/clip-penalty components."""
        norm_smoothness = float(outcome.aggregate.get("ik_smoothness_loss", 0.0)) / max(smoothness_denom, 1e-8)
        objective_scale = max(initial_tracking_loss_scale, 1e-8)
        smoothness_term = objective.smoothness_balance * objective_scale * norm_smoothness
        clip_penalty_term = clip_policy.penalty(outcome, objective_scale)
        objective_loss = outcome.tracking_loss + smoothness_term + clip_penalty_term
        return _ObjectiveBreakdown(
            objective_loss=objective_loss,
            normalized_smoothness_loss=norm_smoothness,
            smoothness_term=smoothness_term,
            clip_penalty_term=clip_penalty_term,
        )

    def _objective_loss(outcome: _EvaluationResult) -> float:
        return _objective_breakdown(outcome).objective_loss

    def _stamp_objective_metrics(outcome: _EvaluationResult) -> None:
        """Write the objective breakdown fields into outcome.aggregate in-place."""
        breakdown = _objective_breakdown(outcome)
        outcome.aggregate["normalized_smoothness_loss"] = breakdown.normalized_smoothness_loss
        outcome.aggregate["objective_smoothness_term"] = breakdown.smoothness_term
        outcome.aggregate["objective_clip_penalty_term"] = breakdown.clip_penalty_term
        outcome.aggregate["objective_loss"] = breakdown.objective_loss

    _zero_update = {
        "position_weight_update_norm": 0.0,
        "rotation_weight_update_norm": 0.0,
        "symmetry_pair_count": float(len(symmetry_pairs)),
    }

    def _history_entry_snapshot(
        plot_iteration: int,
        outcome: _EvaluationResult,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a serializable history entry dict from an evaluation result."""
        per_motion_snapshot = [
            dict(metrics) if metrics is not None else None
            for metrics in outcome.per_motion_metrics
        ]
        return {
            "iteration": plot_iteration,
            "plot_iteration": plot_iteration,
            "stage": "pre_update",
            "aggregate_metrics": dict(outcome.aggregate),
            "per_motion_metrics": per_motion_snapshot,
            "parameter_update": dict(update),
            "error_model": outcome.error_model.to_jsonable(),
        }

    def _record(plot_iteration: int, outcome: _EvaluationResult, update_stats: dict[str, Any]) -> None:
        """Append a history snapshot and optionally refresh the live dashboard."""
        _stamp_objective_metrics(outcome)
        update = dict(update_stats)
        update.setdefault("clip_guard_mode", clip_guard.mode)
        update.setdefault("clip_guard_active_mode", clip_policy.active_mode)
        history.append(
            _history_entry_snapshot(plot_iteration, outcome, update)
        )
        if plot_cfg.enabled and plot_cfg.live:
            _render_dashboard_png(history, dashboard_png, title=plot_title, plot=plot_cfg)

    error_model = _build_error_model()
    root_mask = error_model.parent_indices < 0
    anchor_t0 = time.perf_counter()
    anchor = _evaluate(params, error_model)
    initial_tracking_loss_scale = max(anchor.tracking_loss, 1e-8)
    smoothness_denom = max(float(anchor.aggregate.get("ik_smoothness_loss", 0.0)), 1e-8)
    # Penalty mode measures total clip drift from the original config, not just
    # drift from the current accepted step.
    clip_policy = _ClipGuardPolicy(
        clip_guard, motion_names, anchor.per_motion_metrics, anchor.ik_frame_loss_by_motion
    )
    final_effectors = anchor.mapped_joints
    _record(0, anchor, dict(_zero_update))
    _log_initial_state(anchor, _elapsed(anchor_t0))

    no_improve = 0
    plot_idx = 1
    probe_records: list[dict[str, Any]] = []
    for outer_idx in range(outer_iterations):
        iter_t0 = time.perf_counter()

        accepted = None
        scale = 1.0
        anchor_obj = _objective_loss(anchor)
        # HOLD rows reuse the anchor metrics, but keep the strongest rejection
        # reason so the dashboard/CSV show what stopped progress.
        last_nonempty_violations: list[dict[str, Any]] = []
        n_probes = (line_search.max_backtracks + 1) if line_search.enabled else 1
        for probe in range(n_probes):
            trial_params, stats = _updated_params(
                params, anchor.acc,
                step_scale=scale,
                root_mask=root_mask,
                symmetry_pairs=symmetry_pairs,
                ik=ik_update,
            )
            trial = _evaluate(trial_params, error_model)
            trial_objective = _objective_breakdown(trial)
            decision = clip_policy.probe(anchor, trial)
            if decision.blocked:
                last_nonempty_violations = decision.blocking

            # A line-search probe must both improve the objective and pass the
            # active clip policy. In penalty mode, ordinary clip regressions are
            # already priced into the objective; only catastrophic issues block.
            improved = (
                (not line_search.enabled)
                or (trial_objective.objective_loss < anchor_obj - line_search.min_improvement)
            )
            if save_probe_diagnostics:
                probe_records.append(_probe_record(
                    outer_idx + 1, probe + 1, scale,
                    clip_guard.mode, clip_policy.active_mode,
                    trial, trial_objective.objective_loss, anchor_obj,
                    trial_objective.smoothness_term, trial_objective.clip_penalty_term,
                    accepted=bool(improved and not decision.blocked),
                    blocking=decision.blocking,
                    penalty_breakdown=clip_policy.penalty_breakdown(trial),
                ))
            if improved and not decision.blocked:
                accepted = (trial_params, trial, stats, scale)
                _log_accept_guard_notes(
                    outer_idx + 1,
                    outer_iterations,
                    decision.advisory,
                    scale=scale,
                    clip_guard=clip_guard,
                )
                break
            if decision.blocked:
                _log_reject_probe(
                    outer_idx + 1,
                    outer_iterations,
                    probe + 1,
                    n_probes,
                    scale=scale,
                    clip_guard=clip_guard,
                    guard_label=str(decision.blocking[0].get("guard", "clip_guard")),
                    blocking=decision.blocking,
                    advisory=decision.advisory,
                )
            else:
                objective_improvement = anchor_obj - trial_objective.objective_loss
                if objective_improvement > 0.0:
                    objective_message = (
                        f"objective improved by {objective_improvement:.6f}, "
                        f"below acceptance margin (anchor={anchor_obj:.6f}, "
                        f"trial={trial_objective.objective_loss:.6f})"
                    )
                elif objective_improvement < 0.0:
                    objective_message = (
                        f"objective worsened by {-objective_improvement:.6f} "
                        f"(anchor={anchor_obj:.6f}, trial={trial_objective.objective_loss:.6f})"
                    )
                else:
                    objective_message = (
                        f"objective unchanged (anchor={anchor_obj:.6f}, "
                        f"trial={trial_objective.objective_loss:.6f})"
                    )
                _log_reject_probe(
                    outer_idx + 1,
                    outer_iterations,
                    probe + 1,
                    n_probes,
                    scale=scale,
                    clip_guard=clip_guard,
                    objective_message=objective_message,
                    advisory=decision.advisory,
                )
            scale *= line_search.shrink

        if accepted is not None:
            params, anchor, stats, scale = accepted
            final_effectors = anchor.mapped_joints
            stats["line_search_scale"] = float(scale)
            stats["line_search_accepted"] = 1.0
            no_improve = 0
            clip_policy.note_accepted(anchor)
            _record(plot_idx, anchor, stats)
            _log_accept(
                outer_idx + 1,
                outer_iterations,
                anchor,
                stats,
                scale=scale,
                elapsed=_elapsed(iter_t0),
            )
        else:
            no_improve += 1
            hold_stats = dict(_zero_update)
            hold_stats["line_search_scale"] = 0.0
            hold_stats["line_search_accepted"] = 0.0
            hold_stats["clip_guard_violation_count"] = float(len(last_nonempty_violations))
            if last_nonempty_violations:
                top = last_nonempty_violations[0]
                hold_stats["clip_guard_top_motion"] = top["motion"]
                hold_stats["clip_guard_top_metric"] = top["metric"]
                hold_stats["clip_guard_top_regression"] = float(top["relative"])
                if "guard" in top:
                    hold_stats["clip_guard_top_guard"] = top["guard"]
                if "frame" in top:
                    hold_stats["clip_guard_top_frame"] = float(top["frame"])
            _record(plot_idx, anchor, hold_stats)
            _log_hold(
                outer_idx + 1,
                outer_iterations,
                anchor,
                no_improve=no_improve,
                violation_count=len(last_nonempty_violations),
                elapsed=_elapsed(iter_t0),
            )

        plot_idx += 1
        if accepted is None and convergence_patience >= 0 and no_improve > convergence_patience:
            if clip_policy.try_enter_penalty_phase():
                _log(
                    "clip_guard auto transition: gate -> penalty after "
                    f"{no_improve} hold(s)"
                )
                no_improve = 0
                continue
            _log(
                f"CONVERGED: no improvement for {no_improve} consecutive iteration(s). "
                f"Stopping early after {plot_idx - 1} step(s) in {_elapsed_hms(run_t0)}"
            )
            break

    if save_probe_diagnostics:
        probe_csv = _write_probe_diagnostics_csv(probe_records, artifact_dir)
        if probe_csv is not None:
            _log_block("SAVED PROBE DIAGNOSTICS:", f"    {probe_csv}")

    return _finalize_run(
        workspace=workspace,
        params=params,
        history=history,
        artifact_dir=artifact_dir,
        bvh_paths=bvh_paths,
        source_type=source_type,
        target_type=target_type,
        final_effectors=final_effectors,
        symmetry_pairs=symmetry_pairs,
        plot_cfg=plot_cfg,
        plot_title=plot_title,
        dashboard_png=dashboard_png,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for the optimizer entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize IK match-table weights for a robot retargeter config."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help=f"Input config JSON file. Defaults to {_DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        metavar="PATH",
        help=(
            "Render a dashboard from optimizer_metrics.json or iteration_metrics.csv; "
            "optional second PATH is the output PNG."
        ),
    )
    args = parser.parse_args(argv)
    if args.plot is not None and len(args.plot) > 2:
        parser.error(
            "--plot accepts <optimizer_metrics.json|...iteration_metrics.csv> [out.png]"
        )
    return args


def _resolve_config_path(config_arg: pathlib.Path | None) -> pathlib.Path:
    """Resolve the config path from the CLI argument, falling back to the default."""
    config_path = config_arg or _DEFAULT_CONFIG_PATH
    return pathlib.Path(config_path).expanduser().resolve()


def main() -> int:
    """CLI entry point: parse args, run optimizer or render a dashboard, return exit code."""
    args = _parse_args(sys.argv[1:])
    if args.plot is not None:
        out = pathlib.Path(args.plot[1]) if len(args.plot) > 1 else None
        path = render_dashboard_from_file(args.plot[0], out)
        if path is not None:
            _log(f"plotted {path}")
            return 0
        _log("plot produced no output (no history or plotting unavailable)")
        return 1

    config_path = _resolve_config_path(args.config)
    if not config_path.is_file():
        _log(f"config file not found: {config_path}")
        return 2

    config = load_json(config_path)
    wp.init()
    run_optimizer(config, config_path.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
