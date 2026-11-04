# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

# Basic two-joint IK in the style of:
# https://theorangeduck.com/page/simple-two-joint

_EPSILON = wp.constant(1e-3)


@wp.struct
class TwoBoneIKResult:
    root: wp.transform
    mid : wp.transform
    tip : wp.transform


@wp.func
def wp_solve_two_bone_ik(    in_weight            : wp.float32,
    in_a_parent_world_tx : wp.transform,
    in_a_world_tx        : wp.transform,
    in_b_world_tx        : wp.transform,
    in_c_world_tx        : wp.transform,
    in_t_world_tx        : wp.transform,
    in_use_hint          : wp.bool,
    in_hint_world        : wp.vec3
):
    """
    Solve a simple analytic two-joint IK chain (A-B-C) toward target T in world space.

    Args:
        in_weight (wp.float32): Blend weight between current position and target (0.0 to 1.0).
            0.0 keeps the chain at its current pose, 1.0 fully reaches towards the target.
        in_a_parent_world_tx (wp.transform): World transform of the parent of joint A.
        in_a_world_tx (wp.transform): Current world transform of joint A (root).
        in_b_world_tx (wp.transform): Current world transform of joint B (mid).
        in_c_world_tx (wp.transform): Current world transform of joint C (tip/end effector).
        in_t_world_tx (wp.transform): Target world transform for the end effector.
        in_use_hint (wp.bool): Whether to use the hint vector to choose the bending plane.
        in_hint_world (wp.vec3): World-space position used as a hint to disambiguate the bend direction when in_use_hint is True.
    Returns:
        TwoBoneIKResult:
            Struct containing the solved world-space transforms for root (A), mid (B), and tip (C).
    """
    in_weight = wp.clamp(in_weight, 0.0, 1.0)

    a = in_a_world_tx.p
    b = in_b_world_tx.p
    c = in_c_world_tx.p
    t = wp.lerp(c, in_t_world_tx.p, in_weight)

    # Segment lengths and clamped target distance |AT|
    l_ab = wp.length(b - a)
    l_cb = wp.length(b - c)
    l_at = wp.clamp(wp.length(t - a), _EPSILON, l_ab + l_cb - _EPSILON)

    # Current interior angles at A and B.
    ac_ab_0 = wp.acos(wp.clamp(wp.dot(wp.normalize(c - a), wp.normalize(b - a)), -1.0, 1.0))
    ba_bc_0 = wp.acos(wp.clamp(wp.dot(wp.normalize(a - b), wp.normalize(c - b)), -1.0, 1.0))

    # Desired interior angles from cosine rule so |AC| matches |AT|.
    ac_ab_1 = wp.acos(wp.clamp(((l_ab * l_ab) + (l_at * l_at) - (l_cb * l_cb)) / (2.0 * l_ab * l_at), -1.0, 1.0))
    ba_bc_1 = wp.acos(wp.clamp(((l_ab * l_ab) + (l_cb * l_cb) - (l_at * l_at)) / (2.0 * l_ab * l_cb), -1.0, 1.0))

     # Bend axis: optionally stabilized by a user hint.
    rot_axis = wp.normalize(wp.where(in_use_hint, wp.cross(c - a, in_hint_world - a), wp.cross(c - a, b - a)))
    rot_a_local = wp.quat_from_axis_angle(wp.quat_rotate(wp.quat_inverse(in_a_world_tx.q), rot_axis), ac_ab_1 - ac_ab_0)
    rot_b_local = wp.quat_from_axis_angle(wp.quat_rotate(wp.quat_inverse(in_b_world_tx.q), rot_axis), ba_bc_1 - ba_bc_0)

    a_local_tx = wp.mul(wp.mul(wp.transform_inverse(in_a_parent_world_tx), in_a_world_tx), wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_a_local))
    a_world_tx = wp.mul(in_a_parent_world_tx, a_local_tx)

    ac_local = wp.quat_rotate(wp.quat_inverse(a_world_tx.q), c - a)
    at_local = wp.quat_rotate(wp.quat_inverse(a_world_tx.q), t - a)
    a_local_tx = wp.mul(a_local_tx, wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_between_vectors(ac_local, at_local)))
    a_world_tx = wp.mul(in_a_parent_world_tx, a_local_tx)

    b_local_tx = wp.mul(wp.mul(wp.transform_inverse(in_a_world_tx), in_b_world_tx), wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_b_local))
    b_world_tx = wp.mul(a_world_tx, b_local_tx)
    c_world_tx = wp.mul(b_world_tx, wp.mul(wp.transform_inverse(in_b_world_tx), in_c_world_tx))

    # Return result, renormalizing quaternions to avoid drift.
    result = TwoBoneIKResult()
    result.root = wp.transform(a_world_tx.p, wp.normalize(a_world_tx.q))
    result.mid = wp.transform(b_world_tx.p, wp.normalize(b_world_tx.q))
    result.tip = wp.transform(c_world_tx.p, wp.normalize(wp.quat_slerp(c_world_tx.q, in_t_world_tx.q, in_weight)))

    return result


@wp.kernel
def two_bone_ik_kernel(
    in_weight            : wp.float32,
    in_a_parent_world_tx : wp.transform,
    in_a_world_tx        : wp.transform,
    in_b_world_tx        : wp.transform,
    in_c_world_tx        : wp.transform,
    in_t_world_tx        : wp.transform,
    in_use_hint          : wp.bool,
    in_hint_world        : wp.vec3,
    out_result           : wp.array(dtype=wp.transform)
):
    result = wp_solve_two_bone_ik(
        in_weight, in_a_parent_world_tx,
        in_a_world_tx, in_b_world_tx, in_c_world_tx, in_t_world_tx,
        in_use_hint, in_hint_world)

    out_result[0] = result.root
    out_result[1] = result.mid
    out_result[2] = result.tip
