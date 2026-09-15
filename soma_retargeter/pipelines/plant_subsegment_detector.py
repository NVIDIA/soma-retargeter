# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Detect high-confidence flat foot plant subsegments inside coarse contact segments.

Operates on the output of the velocity/jerk contact detector and uses foot
landmark geometry to identify intervals where the foot is reliably flat on a
support surface.  Only those intervals are promoted to correction targets by
the downstream :class:`PlantCorrectionBlender`.
"""


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.animation.contact_phase import (
    FootLandmarkModel,
    build_heuristic_foot_landmark_model,
    load_authored_foot_landmark_model,
    project_foot_landmarks,
)
from soma_retargeter.utils.math_utils import (
    contiguous_true_runs,
    select_stable_frame_in_segment,
    _STABLE_FRAME_TOLERANCE,
    _STABLE_FRAME_EPSILON,
)

if TYPE_CHECKING:
    from soma_retargeter.animation.contact_detection import BaselineAResult
    from soma_retargeter.animation.skeleton import Skeleton
    from soma_retargeter.animation.animation_buffer import AnimationBuffer

logger = logging.getLogger(__name__)

_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_EPS = 1e-8

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlantSubsegmentConfig:
    max_flatness_deg: float = 25.0
    plant_speed_threshold: float = 0.15
    max_height_spread: float = 0.05
    max_position_delta: float = 0.01
    min_plant_frames: int = 3
    min_acceptance_confidence: float = 0.5
    ground_height_tolerance: float = 0.03
    ground_confidence_boost: float = 1.15
    elevated_confidence_scale: float = 1.0
    max_gap_frames: int = 2

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlantSubsegmentConfig":
        """Construct a ``PlantSubsegmentConfig`` from a dictionary, using defaults for missing keys.

        Args:
            data: Mapping of field names to values, or ``None`` to get all defaults.

        Returns:
            A ``PlantSubsegmentConfig`` with values from *data* overlaid on the defaults.
        """
        if data is None:
            return cls()
        kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in data:
                kwargs[f] = type(cls.__dataclass_fields__[f].default)(data[f])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class PlantSubsegment:
    foot_index: int
    start_frame: int
    end_frame: int
    stable_frame: int
    confidence: float
    support_height: float
    mean_flatness_deg: float


@dataclass
class PlantSubsegmentResult:
    subsegments: list[PlantSubsegment] = field(default_factory=list)
    landmark_model: FootLandmarkModel | None = None
    heel_local_offsets: np.ndarray | None = None
    sole_normal_local: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two 3-vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    cos_angle = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _pick_stable_frame(speed_slice: np.ndarray, global_start: int) -> int:
    """Lowest-speed frame within a run, biased toward the midpoint."""
    return select_stable_frame_in_segment(
        speed_slice, global_start, global_start + len(speed_slice),
        tolerance=_STABLE_FRAME_TOLERANCE,
        epsilon=_STABLE_FRAME_EPSILON,
        context="_pick_stable_frame",
    )


def _build_default_landmark_model(skeleton, skeleton_type: str, root_tx):
    """Load authored landmarks if available, otherwise build heuristically."""
    try:
        authored = load_authored_foot_landmark_model(skeleton, skeleton_type)
        if authored is not None:
            return authored
    except Exception:
        pass
    return build_heuristic_foot_landmark_model(skeleton, skeleton_type, root_tx)


def _bridge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False-gaps of up to *max_gap* frames with True.

    Prevents single-frame noise blips from fragmenting an otherwise
    contiguous plant run.  A no-op when *max_gap* <= 0.
    """
    if max_gap <= 0:
        return mask
    result = mask.copy()
    false_runs = contiguous_true_runs(~mask, 0)
    for s, e in false_runs:
        if (e - s) <= max_gap:
            result[s:e] = True
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_plant_subsegments(
    result: "BaselineAResult",
    skeleton: "Skeleton",
    animation: "AnimationBuffer",
    root_tx: wp.transform,
    skeleton_type: str,
    contact_joint_names: list[str],
    foot_channel_map: dict[int, int],
    config: PlantSubsegmentConfig | None = None,
    all_global_tx: np.ndarray | None = None,
    landmark_model: FootLandmarkModel | None = None,
) -> PlantSubsegmentResult:
    """Identify high-confidence flat-plant subsegments within coarse contacts.

    Parameters
    ----------
    result : BaselineAResult
        Output of ``detect_contacts_velocity_jerk``.
    skeleton, animation, root_tx :
        Source animation data for landmark projection.
    skeleton_type : str
        ``"soma"``.
    contact_joint_names : list[str]
        Joint names tracked by the contact detector (e.g.
        ``["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]``).
    foot_channel_map : dict[int, int]
        Maps detector channel index to foot index (0=left, 1=right).
    config : PlantSubsegmentConfig, optional
    all_global_tx : (T, J, 7) float32, optional
        Pre-computed global transforms.  Computed if not supplied.
    landmark_model : FootLandmarkModel, optional
        Reuse an existing model instead of loading from disk.

    Returns
    -------
    PlantSubsegmentResult
    """
    if config is None:
        config = PlantSubsegmentConfig()

    if landmark_model is None:
        landmark_model = _build_default_landmark_model(skeleton, skeleton_type, root_tx)

    if all_global_tx is None:
        all_global_tx = pose_utils.compute_global_poses_batch(
            skeleton, animation.local_transforms, root_tx,
        )
    else:
        all_global_tx = np.asarray(all_global_tx, dtype=np.float32)

    landmarks = project_foot_landmarks(all_global_tx, landmark_model, use_warp=True)
    heel_pos = np.asarray(landmarks.heel_positions, dtype=np.float64)
    toe_pos = np.asarray(landmarks.toe_pivot_positions, dtype=np.float64)
    sole_normals = np.asarray(landmarks.sole_normals, dtype=np.float64)

    num_frames = heel_pos.shape[0]
    max_flatness_rad = np.deg2rad(config.max_flatness_deg)

    # Per-frame foot center and speed of the foot center
    foot_center = 0.5 * (heel_pos + toe_pos)  # (T, 2, 3)
    foot_center_delta = np.zeros((num_frames, 2), dtype=np.float64)
    if num_frames > 1:
        foot_center_delta[1:] = np.linalg.norm(
            np.diff(foot_center, axis=0), axis=-1,
        )

    # Precompute per-frame, per-foot quality metrics (vectorized)
    norms = np.linalg.norm(sole_normals, axis=-1, keepdims=True)
    norms = np.where(norms < _EPS, 1.0, norms)
    dot_up = (sole_normals / norms)[:, :, 2]  # dot with [0,0,1] = z component
    flatness_angle = np.arccos(np.clip(dot_up, -1.0, 1.0))

    height_spread = np.abs(heel_pos[:, :, 2] - toe_pos[:, :, 2])  # (T, 2)

    segments = result.contact_segments or []
    speed = result.speed  # (T, C) from contact detector

    subsegments: list[PlantSubsegment] = []

    for seg in segments:
        ch = seg.joint_channel
        if ch not in foot_channel_map:
            continue
        foot = foot_channel_map[ch]
        s, e = seg.start_frame, seg.end_frame

        if e - s < config.min_plant_frames:
            continue

        # Boolean mask of plant-candidate frames within this segment
        frames = np.arange(s, e)
        mask_flat = flatness_angle[frames, foot] < max_flatness_rad
        mask_speed = speed[frames, ch] < config.plant_speed_threshold
        mask_spread = height_spread[frames, foot] < config.max_height_spread
        mask_stable = foot_center_delta[frames, foot] < config.max_position_delta
        combined = mask_flat & mask_speed & mask_spread & mask_stable

        if config.max_gap_frames > 0:
            combined = _bridge_small_gaps(combined, config.max_gap_frames)

        # Extract contiguous runs of True values
        runs = contiguous_true_runs(combined, s)
        for run_start, run_end in runs:
            run_len = run_end - run_start
            if run_len < config.min_plant_frames:
                continue

            run_frames = np.arange(run_start, run_end)

            # Per-frame quality score
            flat_score = 1.0 - np.clip(
                flatness_angle[run_frames, foot] / max_flatness_rad, 0.0, 1.0,
            )
            speed_score = 1.0 - np.clip(
                speed[run_frames, ch].astype(np.float64) / config.plant_speed_threshold,
                0.0, 1.0,
            )
            base_conf = float(np.mean(flat_score * speed_score))

            mean_height = float(np.mean(foot_center[run_frames, foot, 2]))
            if abs(mean_height) <= config.ground_height_tolerance:
                conf = min(base_conf * config.ground_confidence_boost, 1.0)
            else:
                conf = min(base_conf * config.elevated_confidence_scale, 1.0)

            if conf < config.min_acceptance_confidence:
                continue

            stable = _pick_stable_frame(speed[run_start:run_end, ch], run_start)
            mean_flat_deg = float(np.rad2deg(np.mean(flatness_angle[run_frames, foot])))

            subsegments.append(PlantSubsegment(
                foot_index=foot,
                start_frame=run_start,
                end_frame=run_end,
                stable_frame=stable,
                confidence=conf,
                support_height=mean_height,
                mean_flatness_deg=mean_flat_deg,
            ))

    # Cache landmark local offsets for the blender
    heel_local = np.zeros((2, 3), dtype=np.float32)
    sole_local = np.zeros((2, 3), dtype=np.float32)
    for foot_rt in landmark_model.feet:
        heel_local[foot_rt.foot_index] = np.asarray(foot_rt.heel_local_in_foot, dtype=np.float32)
        sole_local[foot_rt.foot_index] = np.asarray(foot_rt.sole_normal_local, dtype=np.float32)

    logger.info(
        "Plant subsegment detection: %d subsegments accepted out of %d coarse segments",
        len(subsegments), len(segments),
    )

    return PlantSubsegmentResult(
        subsegments=subsegments,
        landmark_model=landmark_model,
        heel_local_offsets=heel_local,
        sole_normal_local=sole_local,
    )


