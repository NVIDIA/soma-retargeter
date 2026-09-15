# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Foot-target blender that corrects only high-confidence flat plant subsegments.

For each accepted :class:`PlantSubsegment`, a stable effector target is
computed from the retargeted ``input_targets`` using the **median** position
and **averaged** rotation across all frames in the subsegment range.

The foot rotation is flattened so the sole is horizontal.  By default, the
heading (yaw about world-up) is preserved; an optional
``enable_flatten_foot_plant`` mode applies the stronger sole-normal-only
flattening that may shift heading.

Position correction propagates into adjacent free-swing spans via **decayed
boundary residuals**: the positional offset at the plant boundary is carried
forward/backward with a smootherstep decay envelope, so the swing trajectory
follows its original shape with a gentle, bounded nudge rather than being
pulled toward a fixed anchor.

Rotation correction is conservative: full anchor rotation is applied only
inside the plant core, with a smootherstep decay window sized by
``rotation_propagation_ratio`` of the adjacent free-swing span on each side
to avoid boundary pops.
"""


from typing import TYPE_CHECKING

import math

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from soma_retargeter.pipelines.plant_subsegment_detector import (
        PlantSubsegmentResult,
    )

_EPS = 1e-8
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

DEFAULT_CONTACT_JOINTS = ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
DEFAULT_FOOT_CHANNEL_MAP = {0: 0, 2: 1}

_FLATTEN_THRESHOLD_DEG = 5.0
_FLATTEN_MIN_CONFIDENCE = 0.5
_DEFAULT_PROPAGATION_RATIO = 0.4
_DEFAULT_ROTATION_PROPAGATION_RATIO = 0.15
_DEFAULT_MAX_PROPAGATION_FRAMES = 60
_SLERP_NLERP_THRESHOLD = 0.9995
_NUM_FEET = 2


# ---------------------------------------------------------------------------
# Quaternion / geometry helpers
#
# Scalar operations in _flatten_rotation / _flatten_rotation_preserve_heading
# use Warp math types (wp.quat, wp.vec3) as a math library -- no GPU kernels
# or device transfers.  Batch operations inside _precompute_targets use numpy
# vectorized helpers (_quat_average, _smootherstep01_batch, _slerp_batch).
#
# Convention: results stored per-frame or returned to callers are cast to
# float32 for memory efficiency and compatibility.
# ---------------------------------------------------------------------------

def _wp_quat_to_np(q: wp.quat) -> np.ndarray:
    """Convert a Warp quaternion to a float32 numpy array (xyzw)."""
    return np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)


def _quat_average(quats: np.ndarray) -> np.ndarray:
    """Average of unit quaternions (sign-corrected sum, normalized).

    Works well when the input quaternions are close to each other, as is
    the case for frames within a validated plant subsegment.
    """
    quats = np.asarray(quats, dtype=np.float64)
    q0 = quats[0]
    total = np.zeros(4, dtype=np.float64)
    for q in quats:
        if np.dot(q, q0) < 0:
            total -= q
        else:
            total += q
    n = np.linalg.norm(total)
    if n < _EPS:
        return q0.astype(np.float32)
    return (total / n).astype(np.float32)


def _smootherstep01_batch(values: np.ndarray) -> np.ndarray:
    """Vectorized quintic smootherstep over an array of values."""
    t = np.clip(values, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _slerp_batch(
    q0s: np.ndarray,
    q1: np.ndarray,
    ts: np.ndarray | float,
) -> np.ndarray:
    """Vectorized slerp: interpolate each row of *q0s* toward a single *q1*.

    Parameters
    ----------
    q0s : (N, 4) float64
    q1  : (4,) float64
    ts  : (N,) float64 or scalar — interpolation weights per row

    Returns
    -------
    (N, 4) float32
    """
    q0s = np.asarray(q0s, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64).ravel()
    ts = np.broadcast_to(np.asarray(ts, dtype=np.float64), q0s.shape[0])

    dots = q0s @ q1                              # (N,)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    q1_signed = signs[:, None] * q1[None, :]     # (N, 4) — sign-corrected
    dots_abs = np.abs(dots)

    theta = np.arccos(np.clip(dots_abs, -1.0, 1.0))
    sin_theta = np.sin(theta)

    # Full slerp path
    safe_sin = np.where(sin_theta < _EPS, 1.0, sin_theta)
    w0 = np.sin((1.0 - ts) * theta) / safe_sin
    w1 = np.sin(ts * theta) / safe_sin

    slerp_result = w0[:, None] * q0s + w1[:, None] * q1_signed

    # Nlerp fallback for near-parallel quaternions
    nlerp_result = (1.0 - ts[:, None]) * q0s + ts[:, None] * q1_signed
    nlerp_norms = np.linalg.norm(nlerp_result, axis=-1, keepdims=True)
    nlerp_norms = np.where(nlerp_norms < _EPS, 1.0, nlerp_norms)
    nlerp_result = nlerp_result / nlerp_norms

    use_nlerp = (dots_abs > _SLERP_NLERP_THRESHOLD) | (sin_theta < _EPS)
    result = np.where(use_nlerp[:, None], nlerp_result, slerp_result)

    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    norms = np.where(norms < _EPS, 1.0, norms)
    result = result / norms

    return result.astype(np.float32)


def _extract_twist_angle_z(q: wp.quat) -> float:
    """Extract the twist (heading) angle about world-up from a quaternion.

    Uses swing-twist decomposition about the Z axis.
    """
    twist_z = float(q[2])
    twist_w = float(q[3])
    n = math.sqrt(twist_z * twist_z + twist_w * twist_w)
    if n < _EPS:
        return 0.0
    return 2.0 * math.atan2(twist_z / n, twist_w / n)


# ---------------------------------------------------------------------------
# Anchor built per subsegment
# ---------------------------------------------------------------------------

class _PlantAnchor:
    """Frozen correction target for a single plant subsegment.

    Attributes
    ----------
    foot_index : 0 (left) or 1 (right).
    start_frame, end_frame : subsegment bounds in *original animation* frames.
    confidence : quality score from the plant subsegment detector (0..1).
    target_position : (3,) float32 — anchor world position (median).
    target_rotation : (4,) float32 — anchor world rotation (averaged quaternion).
    """

    __slots__ = (
        "foot_index", "start_frame", "end_frame",
        "confidence", "target_position", "target_rotation",
    )

    def __init__(
        self,
        foot_index: int,
        start_frame: int,
        end_frame: int,
        confidence: float,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ):
        self.foot_index = foot_index
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.confidence = confidence
        self.target_position = np.asarray(target_position, dtype=np.float32)
        self.target_rotation = np.asarray(target_rotation, dtype=np.float32)


# ---------------------------------------------------------------------------
# PlantCorrectionBlender
# ---------------------------------------------------------------------------

class PlantCorrectionBlender:
    """Blends retargeted foot targets toward stable plant anchors.

    Plant-core frames receive full anchor-based correction (position and
    rotation).  Free-swing frames receive only a **decayed boundary
    residual** for position -- the positional offset measured at the plant
    boundary is propagated with a smootherstep decay, so the swing follows
    its original trajectory with a gentle, bounded nudge.  Rotation outside
    the core decays via a shorter smootherstep window sized by
    ``rotation_propagation_ratio`` of the adjacent free-swing span.

    Parameters
    ----------
    transition_frames : int
        Minimum floor (frames) for both the position and rotation decay
        windows.
    propagation_ratio : float
        Fraction (0--1) of each adjacent free-swing span used as the
        position residual decay window.  Default 0.4 (40 %).
    rotation_propagation_ratio : float
        Fraction (0--1) of each adjacent free-swing span used as the
        rotation decay window.  Default 0.15 (15 %).  Kept shorter than
        position to avoid rotational drag during fast swing.
    enable_flatten_foot_plant : bool
        When True, apply sole-normal-only flattening that may shift
        heading.  When False (default), flatten while preserving the
        original heading (yaw about world-up).
    """

    def __init__(
        self,
        transition_frames: int = 4,
        propagation_ratio: float = _DEFAULT_PROPAGATION_RATIO,
        rotation_propagation_ratio: float = _DEFAULT_ROTATION_PROPAGATION_RATIO,
        enable_flatten_foot_plant: bool = False,
        sole_normal_local: np.ndarray | None = None,
    ):
        self.transition_frames = max(int(transition_frames), 0)
        self.propagation_ratio = float(np.clip(propagation_ratio, 0.0, 1.0))
        self.rotation_propagation_ratio = float(np.clip(rotation_propagation_ratio, 0.0, 1.0))
        self.enable_flatten_foot_plant = bool(enable_flatten_foot_plant)
        self._sole_normal_local = np.asarray(sole_normal_local, dtype=np.float64) if sole_normal_local is not None else None
        self._anchors: list[_PlantAnchor] = []
        self._precomputed_targets: np.ndarray | None = None
        self._precomputed_mask: np.ndarray | None = None
        self._original_feet: np.ndarray | None = None
        self._num_frames = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        plant_result: "PlantSubsegmentResult",
        input_targets: np.ndarray,
        feet_effector_indices: list[int],
        initialization_frames: int = 0,
        stabilization_frames: int = 0,
        foot_effector_scale: float = 1.0,
        foot_effector_ik_offsets: np.ndarray | None = None,
    ) -> None:
        """Precompute corrected foot targets from accepted plant subsegments.

        Parameters
        ----------
        plant_result : PlantSubsegmentResult
            Accepted subsegments and foot landmark data.
        input_targets : (T_buf, num_effectors, 7) float32
            Buffered retargeted effector targets (pos + quat per effector).
        feet_effector_indices : [left_idx, right_idx]
            Indices into the effector dimension of *input_targets*.
        initialization_frames : int
            Number of initialization frames prepended to the buffer.
        stabilization_frames : int
            Number of stabilization frames prepended to the buffer.
        foot_effector_scale : float32, optional
            Per-foot HumanToRobotScaler scale (needed for rotation flattening).
        foot_effector_ik_offsets : (2, 7) float32, optional
            Per-foot IK offsets (needed for rotation flattening).
        """
        T = input_targets.shape[0]
        buffer_offset = int(initialization_frames) + int(stabilization_frames)
        self._num_frames = T
        self._anchors = []
        self._precomputed_targets = None
        self._precomputed_mask = None
        self._original_feet = None

        sole_normal_eff = np.zeros((_NUM_FEET, 3), dtype=np.float64)
        can_flatten = (
            plant_result.sole_normal_local is not None
            and foot_effector_ik_offsets is not None
        )
        if can_flatten:
            # The sole normal in the foot effector's local frame: the direction
            # that points "up" (away from the ground) when the foot is flat.
            # Defaults to +Z which is correct for robots whose ankle link has
            # identity orientation at rest (e.g. Unitree G1).  Robots with a
            # rotated ankle frame must supply the correct vector via
            # sole_normal_local in the post-processing config.
            effector_up = self._sole_normal_local if self._sole_normal_local is not None else _WORLD_UP
            for foot in range(_NUM_FEET):
                sole_normal_eff[foot] = effector_up

        for sub in plant_result.subsegments:
            eff_idx = feet_effector_indices[sub.foot_index]
            start_buf = max(buffer_offset + sub.start_frame, 0)
            end_buf = min(buffer_offset + sub.end_frame, T)

            if end_buf <= start_buf:
                continue

            seg_rows = np.asarray(
                input_targets[start_buf:end_buf, eff_idx], dtype=np.float64,
            )
            target_pos = np.median(seg_rows[:, :3], axis=0).copy()
            target_rot = _quat_average(seg_rows[:, 3:7])

            if can_flatten and sub.confidence >= _FLATTEN_MIN_CONFIDENCE:
                if self.enable_flatten_foot_plant:
                    target_rot = self._flatten_rotation(
                        target_rot, sole_normal_eff[sub.foot_index],
                    )
                else:
                    target_rot = self._flatten_rotation_preserve_heading(
                        target_rot, sole_normal_eff[sub.foot_index],
                    )

            self._anchors.append(_PlantAnchor(
                foot_index=sub.foot_index,
                start_frame=sub.start_frame,
                end_frame=sub.end_frame,
                confidence=sub.confidence,
                target_position=target_pos,
                target_rotation=target_rot,
            ))

        if self._anchors:
            self._precompute_targets(input_targets, feet_effector_indices, buffer_offset)

    # ------------------------------------------------------------------
    # Rotation flattening
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_rotation(
        effector_rot: np.ndarray,
        sole_normal_eff_local: np.ndarray,
    ) -> np.ndarray:
        """Minimally adjust *effector_rot* so the sole normal aligns with world up.

        Applies the shortest-arc rotation from the current world-space sole
        normal to +Z, which may shift the foot's heading (yaw).
        """
        rot_q = wp.quat(*np.asarray(effector_rot, dtype=np.float64))
        sole_v = wp.vec3(*np.asarray(sole_normal_eff_local, dtype=np.float64))
        world_up = wp.vec3(0.0, 0.0, 1.0)

        sole_world_raw = wp.quat_rotate(rot_q, sole_v)
        if wp.length(sole_world_raw) < _EPS:
            return _wp_quat_to_np(rot_q)
        sole_world = wp.normalize(sole_world_raw)

        cos_angle = max(-1.0, min(1.0, float(wp.dot(sole_world, world_up))))
        angle = math.acos(cos_angle)
        if math.degrees(angle) < _FLATTEN_THRESHOLD_DEG:
            return _wp_quat_to_np(rot_q)

        axis = wp.cross(sole_world, world_up)
        if wp.length(axis) < _EPS:
            return _wp_quat_to_np(rot_q)
        axis = wp.normalize(axis)

        correction = wp.quat_from_axis_angle(axis, angle)
        corrected = wp.normalize(correction * rot_q)

        return _wp_quat_to_np(corrected)

    @staticmethod
    def _flatten_rotation_preserve_heading(
        effector_rot: np.ndarray,
        sole_normal_eff_local: np.ndarray,
    ) -> np.ndarray:
        """Flatten sole normal to world-up while preserving the original heading.

        Applies the same tilt correction as :meth:`_flatten_rotation` then
        counter-rotates about world-up to restore the original yaw, using a
        swing-twist decomposition about Z.
        """
        rot_arr = np.asarray(effector_rot, dtype=np.float64).ravel()
        flat_arr = PlantCorrectionBlender._flatten_rotation(
            effector_rot, sole_normal_eff_local,
        ).astype(np.float64)

        if np.dot(rot_arr, flat_arr) < 0:
            flat_arr = -flat_arr

        rot_q = wp.quat(*rot_arr)
        flat_q = wp.quat(*flat_arr)

        orig_yaw = _extract_twist_angle_z(rot_q)
        flat_yaw = _extract_twist_angle_z(flat_q)
        delta_yaw = flat_yaw - orig_yaw

        if delta_yaw > math.pi:
            delta_yaw -= 2.0 * math.pi
        elif delta_yaw < -math.pi:
            delta_yaw += 2.0 * math.pi

        if abs(delta_yaw) > _EPS:
            world_up = wp.vec3(0.0, 0.0, 1.0)
            counter = wp.quat_from_axis_angle(world_up, -delta_yaw)
            flat_q = wp.normalize(counter * flat_q)

        return _wp_quat_to_np(flat_q)

    # ------------------------------------------------------------------
    # Decay window sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_decay_window(
        free_span: int,
        ratio: float,
        transition_frames: int,
    ) -> int:
        """Compute a decay window (frames) for a free-swing span.

        Used for both position-residual and rotation decay windows.
        """
        if free_span <= 0:
            return 0
        prop = int(np.ceil(free_span * ratio))
        prop = min(prop, _DEFAULT_MAX_PROPAGATION_FRAMES)
        prop = max(prop, transition_frames)
        return min(prop, free_span)

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute_targets(
        self,
        input_targets: np.ndarray,
        feet_effector_indices: list[int],
        buffer_offset: int,
    ) -> None:
        """Precompute corrected foot targets for all frames.

        The algorithm runs in three phases per foot, entirely in numpy:

        1. **Core correction** -- plant-core frames get confidence-weighted
           position lerp and slerp rotation toward the anchor.
        2. **Position residual propagation** -- boundary residuals are decayed
           into adjacent free-swing spans via smootherstep, accumulated with
           ``np.add.at`` to handle overlapping windows.
        3. **Rotation decay** -- slerp toward the anchor rotation over the
           adjacent free-swing span prevents orientation discontinuities.
        """
        T = self._num_frames
        original_feet = np.asarray(
            input_targets[:, feet_effector_indices], dtype=np.float32,
        )
        self._original_feet = original_feet
        precomputed = original_feet.copy()
        mask = np.zeros((T, _NUM_FEET), dtype=bool)

        for foot in range(_NUM_FEET):
            foot_anchors = sorted(
                [a for a in self._anchors if a.foot_index == foot],
                key=lambda a: a.start_frame,
            )
            if not foot_anchors:
                continue

            # ── Phase 1: core correction ───────────────────────────────
            windows: list[tuple[int, int, _PlantAnchor]] = []
            constrained = np.zeros(T, dtype=bool)

            for anchor in foot_anchors:
                start_buf = max(buffer_offset + anchor.start_frame, 0)
                end_buf = min(buffer_offset + anchor.end_frame, T)
                if end_buf <= start_buf:
                    continue
                windows.append((start_buf, end_buf, anchor))

                frames = np.arange(start_buf, end_buf)
                conf = anchor.confidence
                precomputed[frames, foot, :3] = (
                    original_feet[frames, foot, :3] * (1.0 - conf)
                    + anchor.target_position * conf
                )
                precomputed[frames, foot, 3:7] = _slerp_batch(
                    original_feet[frames, foot, 3:7],
                    anchor.target_rotation,
                    conf,
                )
                constrained[frames] = True
                mask[frames, foot] = True

            if not windows:
                continue

            # ── Phase 2: position residual propagation ─────────────────
            propagated = np.zeros((T, 3), dtype=np.float64)
            weight_sum = np.zeros(T, dtype=np.float64)

            for idx, (start_buf, end_buf, anchor) in enumerate(windows):
                conf = anchor.confidence
                left_free_start = windows[idx - 1][1] if idx > 0 else 0
                right_free_end = (
                    windows[idx + 1][0] if idx + 1 < len(windows) else T
                )

                left_border_res = conf * (
                    anchor.target_position
                    - original_feet[start_buf, foot, :3]
                )
                right_border_res = conf * (
                    anchor.target_position
                    - original_feet[end_buf - 1, foot, :3]
                )

                left_free = max(start_buf - left_free_start, 0)
                if left_free > 0:
                    window = self._compute_decay_window(
                        left_free, self.propagation_ratio,
                        self.transition_frames,
                    )
                    if window > 0 and float(np.linalg.norm(left_border_res)) > _EPS:
                        left_start = max(start_buf - window, left_free_start)
                        frames_l = np.arange(left_start, start_buf)
                        d_l = (start_buf - frames_l).astype(np.float64)
                        decay_l = 1.0 - _smootherstep01_batch(
                            d_l / float(window + 1),
                        )
                        valid = decay_l > _EPS
                        if np.any(valid):
                            vf = frames_l[valid]
                            vw = decay_l[valid]
                            res = np.asarray(left_border_res, dtype=np.float64)
                            np.add.at(propagated, vf, res[None, :] * vw[:, None])
                            np.add.at(weight_sum, vf, vw)

                right_free = max(right_free_end - end_buf, 0)
                if right_free > 0:
                    window = self._compute_decay_window(
                        right_free, self.propagation_ratio,
                        self.transition_frames,
                    )
                    if window > 0 and float(np.linalg.norm(right_border_res)) > _EPS:
                        right_end = min(end_buf + window, right_free_end)
                        frames_r = np.arange(end_buf, right_end)
                        d_r = (frames_r - end_buf + 1).astype(np.float64)
                        decay_r = 1.0 - _smootherstep01_batch(
                            d_r / float(window + 1),
                        )
                        valid = decay_r > _EPS
                        if np.any(valid):
                            vf = frames_r[valid]
                            vw = decay_r[valid]
                            res = np.asarray(right_border_res, dtype=np.float64)
                            np.add.at(propagated, vf, res[None, :] * vw[:, None])
                            np.add.at(weight_sum, vf, vw)

            has_propagation = (~constrained) & (weight_sum > _EPS)
            if np.any(has_propagation):
                precomputed[has_propagation, foot, :3] = (
                    original_feet[has_propagation, foot, :3]
                    + propagated[has_propagation].astype(np.float32)
                )
                mask[has_propagation, foot] = True

            # ── Phase 3: rotation decay ────────────────────────────────
            # Deduplicate by frame: when two anchors' decay windows overlap,
            # keep only the entry with the highest slerp weight (nearest core).
            rot_frame_dict: dict[int, tuple[int, float]] = {}

            for idx, (start_buf, end_buf, anchor) in enumerate(windows):
                conf = anchor.confidence
                left_free_start = windows[idx - 1][1] if idx > 0 else 0
                right_free_end = (
                    windows[idx + 1][0] if idx + 1 < len(windows) else T
                )
                left_free = max(start_buf - left_free_start, 0)
                right_free = max(right_free_end - end_buf, 0)

                rot_left = self._compute_decay_window(
                    left_free, self.rotation_propagation_ratio,
                    self.transition_frames,
                )
                if rot_left > 0:
                    ease_start = max(start_buf - rot_left, 0)
                    frames_rl = np.arange(ease_start, start_buf)
                    if len(frames_rl) > 0:
                        free_mask = ~constrained[frames_rl]
                        if np.any(free_mask):
                            f_free = frames_rl[free_mask]
                            d_rl = (start_buf - f_free).astype(np.float64)
                            decay_rl = 1.0 - _smootherstep01_batch(
                                d_rl / float(rot_left + 1),
                            )
                            w_rl = conf * decay_rl
                            active = w_rl > _EPS
                            if np.any(active):
                                for f, w in zip(
                                    f_free[active], w_rl[active],
                                ):
                                    fi = int(f)
                                    wf = float(w)
                                    if fi not in rot_frame_dict or wf > rot_frame_dict[fi][1]:
                                        rot_frame_dict[fi] = (idx, wf)

                rot_right = self._compute_decay_window(
                    right_free, self.rotation_propagation_ratio,
                    self.transition_frames,
                )
                if rot_right > 0:
                    ease_end = min(end_buf + rot_right, T)
                    frames_rr = np.arange(end_buf, ease_end)
                    if len(frames_rr) > 0:
                        free_mask = ~constrained[frames_rr]
                        if np.any(free_mask):
                            f_free = frames_rr[free_mask]
                            d_rr = (f_free - end_buf + 1).astype(np.float64)
                            decay_rr = 1.0 - _smootherstep01_batch(
                                d_rr / float(rot_right + 1),
                            )
                            w_rr = conf * decay_rr
                            active = w_rr > _EPS
                            if np.any(active):
                                for f, w in zip(
                                    f_free[active], w_rr[active],
                                ):
                                    fi = int(f)
                                    wf = float(w)
                                    if fi not in rot_frame_dict or wf > rot_frame_dict[fi][1]:
                                        rot_frame_dict[fi] = (idx, wf)

            if rot_frame_dict:
                anchor_groups: dict[int, tuple[list[int], list[float]]] = {}
                for frame_idx, (win_idx, w) in rot_frame_dict.items():
                    if win_idx not in anchor_groups:
                        anchor_groups[win_idx] = ([], [])
                    anchor_groups[win_idx][0].append(frame_idx)
                    anchor_groups[win_idx][1].append(w)

                for win_idx, (frame_list, weight_list) in anchor_groups.items():
                    target_rot = windows[win_idx][2].target_rotation
                    f_arr = np.array(frame_list, dtype=np.int32)
                    w_arr = np.array(weight_list, dtype=np.float64)
                    precomputed[f_arr, foot, 3:7] = _slerp_batch(
                        original_feet[f_arr, foot, 3:7],
                        target_rot,
                        w_arr,
                    )
                    mask[f_arr, foot] = True

        self._precomputed_targets = precomputed
        self._precomputed_mask = mask

    # ------------------------------------------------------------------
    # Per-frame query (called from pipeline execute loop)
    # ------------------------------------------------------------------

    def blend_targets(
        self,
        frame: int,
        original_feet_tx: np.ndarray,
    ) -> np.ndarray:
        """Apply corrected foot effector targets for *frame* **in-place**.

        The correction is applied as an **additive delta** relative to the
        build-time input.  This preserves any runtime adjustments already
        present in *original_feet_tx* (e.g. collision offsets) while still
        steering the foot toward the precomputed plant anchor.  The delta
        decays to zero at the decay-window boundary via smootherstep.

        *original_feet_tx* is modified in-place and returned.
        """
        if (
            self._precomputed_targets is None
            or self._precomputed_mask is None
            or frame < 0
            or frame >= self._num_frames
        ):
            return original_feet_tx

        for foot in range(_NUM_FEET):
            if self._precomputed_mask[frame, foot]:
                delta = (self._precomputed_targets[frame, foot]
                         - self._original_feet[frame, foot])
                original_feet_tx[foot] = original_feet_tx[foot] + delta
        return original_feet_tx
