# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np


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
