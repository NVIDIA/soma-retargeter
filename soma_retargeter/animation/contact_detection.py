# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Velocity / jerk state-machine foot-contact detector for skeletal animations.

Works on per-joint world-space positions and uses velocity, acceleration and
jerk thresholds combined with a simple state machine (idle -> contact -> lift-off).
Operates in Newton Z-up (metres) and returns per-frame float contact values in [0, 1].
"""


import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import warp as wp

import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.utils.math_utils import (
    contiguous_true_runs,
    select_stable_frame_in_segment,
    _STABLE_FRAME_TOLERANCE,
    _STABLE_FRAME_EPSILON,
)

logger = logging.getLogger(__name__)

# ======================================================================
# Helper utilities
# ======================================================================

def log_contact_segments(contacts: np.ndarray, joint_names: list[str]):
    """Print contiguous contact regions for each channel."""
    _, num_channels = contacts.shape
    for ch in range(num_channels):
        mask = contacts[:, ch] > 0.5
        segments = contiguous_true_runs(mask)
        name = joint_names[ch] if ch < len(joint_names) else f"ch{ch}"
        logger.info(f"  {name}: {len(segments)} segments  {segments[:10]}{'…' if len(segments) > 10 else ''}")


def _ramp_contacts_outward(contacts: np.ndarray, transition_frames: int) -> np.ndarray:
    """Apply an outward linear ramp to binary contacts, per channel.

    All originally detected frames stay at 1.0.  The ramp extends into
    neighbouring non-contact frames, fading linearly from 1.0 → 0.0 over
    *transition_frames* frames on each side of every contact region.

    Parameters
    ----------
    contacts : np.ndarray, shape (T, C), binary {0, 1}
    transition_frames : int – number of ramp frames on each side

    Returns
    -------
    smoothed : np.ndarray, shape (T, C), float32 in [0, 1]
    """
    if transition_frames <= 0:
        return contacts.astype(np.float32, copy=True)

    num_frames, num_channels = contacts.shape
    result = contacts.astype(np.float32).copy()
    ramp = (1.0 - np.arange(1, transition_frames + 1, dtype=np.float32) / (transition_frames + 1)).astype(np.float32)

    for ch in range(num_channels):
        mask = contacts[:, ch] > 0.5
        if not np.any(mask):
            continue
        runs = contiguous_true_runs(mask)

        for start, end in runs:
            before_len = min(transition_frames, start)
            if before_len > 0:
                result[start - before_len : start, ch] = np.maximum(
                    result[start - before_len : start, ch],
                    ramp[:before_len][::-1],
                )
            after_len = min(transition_frames, num_frames - end)
            if after_len > 0:
                result[end : end + after_len, ch] = np.maximum(
                    result[end : end + after_len, ch],
                    ramp[:after_len],
                )

    return result


# ======================================================================
# Baseline A – velocity / jerk state machine
# ======================================================================

@dataclass
class BaselineAThresholds:
    velocity_contact: float = 0.1          # m/s – speed below this → candidate
    jerk_contact: float = -0.05            # m/s³ – negative jerk triggers window
    velocity_uncontact: float = 0.2        # m/s – speed above this → lift off
    velocity_probably_lift_off: float = 0.05  # m/s
    post_contact_jerk_window: int = 5      # frames
    transition_frames: int = 4             # outward ramp frames for smoothing
    edge_propagation_frames: int = 8       # max frames to extend contacts at clip boundaries


@dataclass
class ContactSegment:
    """A single contiguous contact region for one joint."""
    joint_channel: int       # channel index (0..C-1)
    start_frame: int         # first binary-contact frame
    end_frame: int           # one-past-last binary-contact frame
    stable_frame: int        # representative frame (lowest speed, midpoint-biased)
    transform: wp.transform | None = None  # diagnostic only: world-space pose at stable_frame


@dataclass
class BaselineAResult:
    """Result bundle from :func:`detect_contacts_velocity_jerk`.

    **Core** — set by the detector, consumed by the plant subsegment
    detector and ``PlantCorrectionBlender``.

    **Diagnostic** — per-channel kinematics populated for debug viewers
    and NPZ export; not read by any production pipeline code.
    """

    # -- Core fields (detector → plant subsegment detector / blender) ------
    contacts: np.ndarray                              # (T, C) float32 [0,1] – ramped contact weight
    contact_segments: list[ContactSegment] | None = None  # binary segment intervals

    # -- Diagnostic: per-channel kinematics (detector, plots only) ---------
    speed: np.ndarray | None = None         # (T, C) float32 m/s
    jerk: np.ndarray | None = None          # (T, C) float32 m/s^3
    acceleration: np.ndarray | None = None  # (T, C) float32 m/s^2


def _find_contact_segments(
    binary_contacts: np.ndarray,
    speed: np.ndarray,
    all_global_tx: np.ndarray | None,
    joint_indices: list[int],
    compute_transforms: bool = True,
) -> list[ContactSegment]:
    """Find contiguous contact segments and pick a stable frame for each.

    The stable frame is the one with the lowest speed within the segment.
    Among near-ties (within 5 % of the minimum) the frame closest to the
    segment midpoint is preferred.

    Parameters
    ----------
    binary_contacts : (T, C) float32 {0, 1} – contacts *before* ramping
    speed : (T, C) float64
    all_global_tx : np.ndarray or None, shape (F, J, 7) – pre-computed global
        transforms.  Required when *compute_transforms* is True.
    joint_indices : list[int] – skeleton joint indices for each channel
    compute_transforms : bool – when True (default), populate
        ``ContactSegment.transform`` with the world-space pose at the
        stable frame.  Set to False to skip this (diagnostic-only) lookup.

    Returns
    -------
    list[ContactSegment]
    """
    _, num_channels = binary_contacts.shape
    segments: list[ContactSegment] = []

    for ch in range(num_channels):
        mask = binary_contacts[:, ch] > 0.5
        runs = contiguous_true_runs(mask)

        for s, e in runs:
            seg_speed = speed[s:e, ch]
            best = select_stable_frame_in_segment(
                seg_speed, s, e,
                tolerance=_STABLE_FRAME_TOLERANCE,
                epsilon=_STABLE_FRAME_EPSILON,
                context=f"_find_contact_segments ch={ch}",
            )

            tx = None
            if compute_transforms and all_global_tx is not None:
                jidx = joint_indices[ch]
                row = all_global_tx[best, jidx]
                tx = wp.transform(
                    wp.vec3(float(row[0]), float(row[1]), float(row[2])),
                    wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
                )

            segments.append(ContactSegment(
                joint_channel=ch,
                start_frame=s,
                end_frame=e,
                stable_frame=best,
                transform=tx,
            ))

    return segments


def _propagate_edge_contacts(
    contacts: np.ndarray,
    speed: np.ndarray,
    velocity_threshold: float,
    max_edge_frames: int,
) -> None:
    """Extend contacts at clip boundaries where the state machine has a blind spot.

    The forward state machine cannot detect contacts in the first few frames
    because jerk (3rd derivative) is zero there.  Similarly the last frame is
    never processed.  This pass checks whether a detected contact segment sits
    near a clip boundary and, if the speed signal is low enough, propagates the
    contact outward to the boundary.

    Operates **in-place** on *contacts*.

    Parameters
    ----------
    contacts : np.ndarray, shape (T, C), float32  – binary {0, 1}
    speed : np.ndarray, shape (T, C), float64
    velocity_threshold : float – speed must stay below this to propagate
    max_edge_frames : int – only consider segments within this many frames
        of the clip start/end
    """
    if max_edge_frames <= 0:
        return

    num_frames, num_channels = contacts.shape

    for j in range(num_channels):
        active = np.flatnonzero(contacts[:, j] > 0.5)
        if active.size == 0:
            continue

        first_contact = int(active[0])
        if 0 < first_contact <= max_edge_frames:
            blocked = np.flatnonzero(speed[:first_contact, j] >= velocity_threshold)
            start = int(blocked[-1] + 1) if blocked.size > 0 else 0
            contacts[start:first_contact, j] = 1.0

        last_contact = int(active[-1])
        if last_contact >= num_frames - 1 - max_edge_frames:
            trailing = speed[last_contact + 1 :, j] < velocity_threshold
            if trailing.size > 0:
                stop = np.flatnonzero(~trailing)
                end = last_contact + 1 + (int(stop[0]) if stop.size > 0 else trailing.size)
                contacts[last_contact + 1 : end, j] = 1.0


def detect_contacts_velocity_jerk(
    skeleton,
    animation,
    root_tx: wp.transform,
    contact_joint_names: list[str],
    thresholds: BaselineAThresholds | None = None,
    all_global_tx: np.ndarray | None = None,
    ramp_enabled: bool = True,
    propagate_edges: bool = True,
    compute_segment_transforms: bool = True,
) -> BaselineAResult:
    """Baseline A: velocity/jerk state-machine contact detector.

    Parameters
    ----------
    skeleton : Skeleton
    animation : AnimationBuffer  (must have .local_transforms, .sample_rate)
    root_tx : wp.transform – root transform (e.g. Maya-to-Newton rotation)
    contact_joint_names : list[str] – joint names to track
    thresholds : BaselineAThresholds, optional
    all_global_tx : np.ndarray or None – pre-computed (T, J, 7) global transforms
    ramp_enabled : bool – when True (default), apply ``_ramp_contacts_outward``
        to produce smooth ``[0, 1]`` contact weights.  The ramped signal is
        consumed by the plant subsegment detector as the per-frame blend
        strength.  Set to False to keep binary ``{0, 1}`` contacts.
    propagate_edges : bool – when True (default), extend binary contacts toward
        clip start/end where the state machine has a blind spot (jerk is zero in
        the first few frames).  Controlled by
        ``thresholds.edge_propagation_frames``.
    compute_segment_transforms : bool – when True (default), populate each
        ``ContactSegment.transform`` with the world-space joint pose at the
        stable frame.  This is diagnostic metadata used only for debug viewer
        gizmos and NPZ round-trip; set to False to skip.

    Returns
    -------
    BaselineAResult
    """
    if thresholds is None:
        thresholds = BaselineAThresholds()

    joint_indices = [skeleton.joint_names.index(n) for n in contact_joint_names]
    num_channels = len(joint_indices)
    num_frames = animation.num_frames
    sample_rate = animation.sample_rate
    dt = 1.0 / sample_rate

    # Pre-compute global transforms for ALL joints / ALL frames (single kernel launch)
    if all_global_tx is None:
        all_global_tx = pose_utils.compute_global_poses_batch(
            skeleton, animation.local_transforms, root_tx
        )
    else:
        all_global_tx = np.asarray(all_global_tx, dtype=np.float32)

    # Extract world positions for the contact joints only
    idx = np.array(joint_indices, dtype=np.intp)
    positions = all_global_tx[:, idx, :3].astype(np.float32)

    # Velocities – forward difference divided by 2*dt.  The 2x denominator
    # originates from an earlier central-difference formulation and is
    # intentionally kept so that all threshold values (velocity_contact,
    # velocity_uncontact, etc.) remain valid as tuned.  In effect the
    # reported speed is half the true forward-difference velocity.
    velocities = np.zeros_like(positions, dtype=np.float32)
    if num_frames > 1:
        velocities[1:] = np.diff(positions, axis=0) / (2.0 * dt)

    # Accelerations – magnitude of velocity change / dt
    accelerations = np.zeros((num_frames, num_channels), dtype=np.float32)
    if num_frames > 2:
        dv = np.diff(velocities, axis=0)                     # (T-1, C, 3)
        accelerations[2:] = np.linalg.norm(dv[1:], axis=-1).astype(np.float32) / dt

    # Jerk – derivative of acceleration magnitude
    jerk = np.zeros((num_frames, num_channels), dtype=np.float32)
    if num_frames > 3:
        jerk[3:] = np.diff(accelerations[2:], axis=0) / dt

    # Speed per channel (magnitude of velocity vector)
    speed = np.linalg.norm(velocities, axis=-1).astype(np.float32)  # (T, C)

    # State machine -------------------------------------------------------
    contacts = np.zeros((num_frames, num_channels), dtype=np.float32)
    in_contact = [False] * num_channels
    contact_frame = [-1] * num_channels
    lift_off_frame = [-1] * num_channels
    post_jerk_left = [0] * num_channels

    for f in range(num_frames):
        for j in range(num_channels):
            j_jerk = jerk[f, j]
            j_speed = speed[f, j]
            next_speed = speed[min(f + 1, num_frames - 1), j]

            if not in_contact[j]:
                if j_jerk < 0.0 or post_jerk_left[j] > 0:
                    if j_jerk < thresholds.jerk_contact:
                        if post_jerk_left[j] == 0:
                            post_jerk_left[j] = thresholds.post_contact_jerk_window
                        else:
                            post_jerk_left[j] -= 1

                    if j_speed < thresholds.velocity_contact:
                        in_contact[j] = True
                        contact_frame[j] = f

            if in_contact[j]:
                if j_speed >= thresholds.velocity_probably_lift_off:
                    if lift_off_frame[j] == -1:
                        lift_off_frame[j] = f
                else:
                    lift_off_frame[j] = -1

                if j_speed > thresholds.velocity_uncontact and next_speed > thresholds.velocity_uncontact:
                    if lift_off_frame[j] - contact_frame[j] > 0:
                        # Retroactively clear contact from lift-off to current
                        for k in range(lift_off_frame[j], f):
                            contacts[k, j] = 0.0
                    else:
                        contact_frame[j] = -1

                    in_contact[j] = False
                    lift_off_frame[j] = -1
                    post_jerk_left[j] = 0

            contacts[f, j] = 1.0 if in_contact[j] else 0.0

    # Extend contacts at clip boundaries where derivatives are unavailable
    if propagate_edges and thresholds.edge_propagation_frames > 0:
        _propagate_edge_contacts(
            contacts, speed, thresholds.velocity_contact, thresholds.edge_propagation_frames
        )

    # Find contact segments from binary contacts (before ramp smoothing)
    contact_segments = _find_contact_segments(
        contacts, speed, all_global_tx, joint_indices,
        compute_transforms=compute_segment_transforms,
    )

    # Smooth transitions with outward ramp.
    if ramp_enabled and thresholds.transition_frames > 0:
        contacts = _ramp_contacts_outward(contacts, thresholds.transition_frames)

    return BaselineAResult(
        contacts=contacts,
        speed=speed,
        jerk=jerk,
        acceleration=accelerations,
        contact_segments=contact_segments,
    )


@dataclass
class ContactDataBundle:
    """All data loaded from a serialised contact .npz file."""
    result: BaselineAResult
    joint_names: list[str]
    sample_rate: float
    input_targets: np.ndarray | None = None
    feet_effector_indices: list[int] | None = None
    initialization_frames: int | None = None
    skeleton_type: str | None = None
    source_motion_path: str | None = None
    replay_metadata: dict | None = None


def save_baseline_a_result(
    path: str | pathlib.Path,
    result: BaselineAResult,
    joint_names: list[str],
    sample_rate: float,
    input_targets: np.ndarray | None = None,
    feet_effector_indices: list | None = None,
    initialization_frames: int | None = None,
    stabilization_frames: int | None = None,
    skeleton_type: str | None = None,
    source_motion_path: str | None = None,
    foot_effector_scales: np.ndarray | None = None,
    foot_effector_ik_offsets: np.ndarray | None = None,
    foot_channel_map: dict | None = None,
    min_segment_frames: int | None = None,
    transition_frames: int | None = None,
) -> None:
    """Save BaselineAResult and optional pipeline data to a single .npz file.

    Parameters
    ----------
    path : str or pathlib.Path
    result : BaselineAResult from detect_contacts_velocity_jerk
    joint_names : list of contact channel names (e.g. LeftFoot, LeftToeBase, ...)
    sample_rate : float
    input_targets : (T_buf, num_effectors, 7) float32, optional – for replay/gizmos
    feet_effector_indices : (2,) int, optional
    initialization_frames : int, optional
    stabilization_frames : int, optional
    skeleton_type : str, optional – source skeleton type (e.g. "soma")
    source_motion_path : str, optional – absolute path to the source BVH file
    """
    segs = result.contact_segments or []
    seg_joint = np.array([s.joint_channel for s in segs], dtype=np.int32)
    seg_start = np.array([s.start_frame for s in segs], dtype=np.int32)
    seg_end = np.array([s.end_frame for s in segs], dtype=np.int32)
    seg_stable = np.array([s.stable_frame for s in segs], dtype=np.int32)

    has_transforms = segs and all(s.transform is not None for s in segs)
    if has_transforms:
        seg_pos = np.array(
            [[float(s.transform.p[0]), float(s.transform.p[1]), float(s.transform.p[2])] for s in segs],
            dtype=np.float64,
        )
        seg_rot = np.array(
            [
                [float(s.transform.q[0]), float(s.transform.q[1]), float(s.transform.q[2]), float(s.transform.q[3])]
                for s in segs
            ],
            dtype=np.float64,
        )

    save_dict = {
        "contacts": result.contacts,
        "contact_joint_names": np.array(joint_names, dtype=object),
        "sample_rate": np.float64(sample_rate),
        "segments_joint_channel": seg_joint,
        "segments_start_frame": seg_start,
        "segments_end_frame": seg_end,
        "segments_stable_frame": seg_stable,
    }
    if has_transforms:
        save_dict["segments_position"] = seg_pos
        save_dict["segments_rotation"] = seg_rot
    if result.speed is not None:
        save_dict["speed"] = result.speed
    if result.jerk is not None:
        save_dict["jerk"] = result.jerk
    if result.acceleration is not None:
        save_dict["acceleration"] = result.acceleration
    if input_targets is not None:
        save_dict["input_targets"] = np.asarray(input_targets, dtype=np.float32)
    if feet_effector_indices is not None:
        save_dict["feet_effector_indices"] = np.array(feet_effector_indices, dtype=np.int32)
    if initialization_frames is not None:
        save_dict["initialization_frames"] = np.int32(initialization_frames)
    if stabilization_frames is not None:
        save_dict["stabilization_frames"] = np.int32(stabilization_frames)
    if skeleton_type is not None:
        save_dict["skeleton_type"] = np.array(skeleton_type, dtype=object)
    if source_motion_path is not None:
        save_dict["source_motion_path"] = np.array(source_motion_path, dtype=object)
    if foot_effector_scales is not None:
        save_dict["foot_effector_scales"] = np.asarray(foot_effector_scales, dtype=np.float32)
    if foot_effector_ik_offsets is not None:
        save_dict["foot_effector_ik_offsets"] = np.asarray(foot_effector_ik_offsets, dtype=np.float32)
    if foot_channel_map is not None:
        keys = np.asarray(list(foot_channel_map.keys()), dtype=np.int32)
        values = np.asarray(list(foot_channel_map.values()), dtype=np.int32)
        save_dict["foot_channel_map_keys"] = keys
        save_dict["foot_channel_map_values"] = values
    if min_segment_frames is not None:
        save_dict["min_segment_frames"] = np.int32(min_segment_frames)
    if transition_frames is not None:
        save_dict["transition_frames"] = np.int32(transition_frames)
    np.savez(path, **save_dict)


def load_baseline_a_result(path: str | pathlib.Path) -> ContactDataBundle:
    """Load BaselineAResult and optional pipeline data from .npz."""
    data = np.load(path, allow_pickle=True)
    contacts = data["contacts"]
    speed = data["speed"] if "speed" in data else None
    jerk = data["jerk"] if "jerk" in data else None
    acceleration = data["acceleration"] if "acceleration" in data else None
    jn = data["contact_joint_names"]
    if np.isscalar(jn) or (hasattr(jn, "ndim") and jn.ndim == 0):
        joint_names = [str(jn)]
    else:
        joint_names = jn.tolist() if hasattr(jn, "tolist") else list(jn)
        joint_names = [str(x) for x in joint_names]
    sample_rate = float(data["sample_rate"])

    seg_joint = data["segments_joint_channel"]
    seg_start = data["segments_start_frame"]
    seg_end = data["segments_end_frame"]
    seg_stable = data["segments_stable_frame"]
    seg_pos = data["segments_position"] if "segments_position" in data else None
    seg_rot = data["segments_rotation"] if "segments_rotation" in data else None
    n_seg = len(seg_joint)
    contact_segments = []
    for i in range(n_seg):
        tx = None
        if seg_pos is not None and seg_rot is not None:
            tx = wp.transform(
                wp.vec3(float(seg_pos[i, 0]), float(seg_pos[i, 1]), float(seg_pos[i, 2])),
                wp.quat(
                    float(seg_rot[i, 0]), float(seg_rot[i, 1]),
                    float(seg_rot[i, 2]), float(seg_rot[i, 3]),
                ),
            )
        contact_segments.append(ContactSegment(
            joint_channel=int(seg_joint[i]),
            start_frame=int(seg_start[i]),
            end_frame=int(seg_end[i]),
            stable_frame=int(seg_stable[i]),
            transform=tx,
        ))
    result = BaselineAResult(
        contacts=contacts,
        speed=speed,
        jerk=jerk,
        acceleration=acceleration,
        contact_segments=contact_segments,
    )
    input_targets = data["input_targets"] if "input_targets" in data else None
    feet_effector_indices = data["feet_effector_indices"].tolist() if "feet_effector_indices" in data else None
    init_frames = int(data["initialization_frames"]) if "initialization_frames" in data else None
    skel_type = None
    if "skeleton_type" in data:
        st = data["skeleton_type"]
        skel_type = str(st.item()) if hasattr(st, "item") and st.size > 0 else str(st)
    source_path = None
    if "source_motion_path" in data:
        sp = data["source_motion_path"]
        source_path = str(sp.item()) if hasattr(sp, "item") and sp.size > 0 else str(sp)
    foot_channel_map = None
    if "foot_channel_map_keys" in data and "foot_channel_map_values" in data:
        foot_channel_map = {
            int(key): int(value)
            for key, value in zip(data["foot_channel_map_keys"], data["foot_channel_map_values"])
        }
    replay_metadata = {
        "foot_effector_scales": data["foot_effector_scales"] if "foot_effector_scales" in data else None,
        "foot_effector_ik_offsets": data["foot_effector_ik_offsets"] if "foot_effector_ik_offsets" in data else None,
        "foot_channel_map": foot_channel_map,
        "stabilization_frames": int(data["stabilization_frames"]) if "stabilization_frames" in data else 0,
        "min_segment_frames": int(data["min_segment_frames"]) if "min_segment_frames" in data else None,
        "transition_frames": int(data["transition_frames"]) if "transition_frames" in data else None,
    }
    return ContactDataBundle(
        result=result,
        joint_names=joint_names,
        sample_rate=sample_rate,
        input_targets=input_targets,
        feet_effector_indices=feet_effector_indices,
        initialization_frames=init_frames,
        skeleton_type=skel_type,
        source_motion_path=source_path,
        replay_metadata=replay_metadata,
    )
