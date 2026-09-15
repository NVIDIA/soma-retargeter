# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np


def contiguous_true_runs(
    mask: np.ndarray,
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` pairs for contiguous True runs in a 1-D boolean array.

    Returned indices are shifted by *offset* so they can refer to source-animation
    frames rather than mask-local indices.
    """
    if len(mask) == 0:
        return []
    diff = np.diff(mask.astype(np.int32))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return [(int(s) + offset, int(e) + offset) for s, e in zip(starts, ends)]


_STABLE_FRAME_TOLERANCE = 0.05
_STABLE_FRAME_EPSILON = 1e-12


def select_stable_frame_in_segment(
    seg_speed: np.ndarray,
    segment_start: int,
    segment_end: int,
    *,
    tolerance: float = _STABLE_FRAME_TOLERANCE,
    epsilon: float = _STABLE_FRAME_EPSILON,
    context: str | None = None,
) -> int:
    """Pick the lowest-speed frame within [segment_start, segment_end), midpoint-biased.

    Defensive against NaN: if the entire slice is non-finite, returns the
    segment midpoint and emits a warning.  If some entries are finite,
    those are used and NaN entries are excluded from candidates.
    """
    finite_mask = np.isfinite(seg_speed)
    midpoint = (segment_start + segment_end) / 2.0

    if not finite_mask.any():
        if context:
            print(
                f"[WARNING] all-NaN speed segment [{segment_start}:{segment_end}] "
                f"in {context}; using midpoint",
                flush=True,
            )
        return min(round(midpoint), segment_end - 1)

    finite_speed = np.where(finite_mask, seg_speed, np.inf)
    min_speed = float(np.min(finite_speed))
    threshold = min_speed * (1.0 + tolerance) + epsilon
    candidates = np.where((finite_speed <= threshold) & finite_mask)[0] + segment_start

    if candidates.size == 0:
        return int(np.argmin(finite_speed)) + segment_start

    return int(candidates[np.argmin(np.abs(candidates - midpoint))])


def transform_from_array(array: np.ndarray):
    """Construct a wp.transform from a flat array"""
    return wp.transform(wp.vec3(array[0:3]), wp.quat(array[3:7]))


@wp.func
def are_rotations_equal(q1: wp.quat, q2: wp.quat, tolerance: float):
    """Check if two quaternions represent the same rotation within a tolerance."""
    # Check if dot product is close to 1.0 or -1.0
    return wp.abs(wp.abs(wp.dot(q1, q2)) - 1.0) < tolerance


@wp.func
def are_transforms_equal(t1: wp.transform, t2: wp.transform, tolerance: float):
    """Check if two transforms are approximately equal."""
    diff = t1.p - t2.p
    return wp.abs(wp.dot(diff, diff)) < tolerance and are_rotations_equal(wp.quat(t1.q), wp.quat(t2.q), tolerance)


@wp.func
def quat_twist(twist_axis: wp.vec3, q: wp.quat):
    """Extract the twist component of a quaternion around a given axis."""
    v = wp.vec3(q[0], q[1], q[2])
    dotP = wp.dot(twist_axis, v)
    p = twist_axis * dotP
    return wp.normalize(wp.quat(p[0], p[1], p[2], q[3]))


@wp.func
def project_point_to_plane(point: wp.vec3, normal: wp.vec3):
    """Orthogonally project a point onto a plane through the origin."""
    return point - wp.dot(point, normal) * normal
