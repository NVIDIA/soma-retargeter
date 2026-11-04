# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import warp as wp
import soma_retargeter.utils.math_utils as math_utils


@wp.func
def wp_compute_local_pose(
    in_num_joints     : wp.int32,
    in_root_tx        : wp.transform,
    in_parent_indices : wp.array(dtype=wp.int32),
    in_global_pose    : wp.array(dtype=wp.transform),
    out_result        : wp.array(dtype=wp.transform)
):
    """Compute local joint transforms from global joint transforms"""
    out_result[0] = wp.transform_multiply(wp.transform_inverse(in_root_tx), in_global_pose[0])
    for idx in range(1, in_num_joints):
        parent_tx = in_global_pose[in_parent_indices[idx]]
        out_result[idx] = wp.transform_multiply(wp.transform_inverse(parent_tx), in_global_pose[idx])


@wp.kernel
def compute_local_pose_kernel(
    in_num_joints     : wp.int32,
    in_root_tx        : wp.transform,
    in_parent_indices : wp.array(dtype=wp.int32),
    in_global_pose    : wp.array(dtype=wp.transform),
    out_result        : wp.array(dtype=wp.transform)
):
    """Kernel wrapper for wp_compute_local_pose"""
    wp_compute_local_pose(in_num_joints, in_root_tx, in_parent_indices, in_global_pose, out_result)


@wp.func
def wp_compute_global_pose(
    in_num_joints     : wp.int32,
    in_root_tx        : wp.transform,
    in_parent_indices : wp.array(dtype=wp.int32),
    in_local_pose     : wp.array(dtype=wp.transform),
    out_result        : wp.array(dtype=wp.transform)
):
    """Compute global joint transforms from local joint transforms."""
    out_result[0] = wp.mul(in_root_tx, in_local_pose[0])
    for idx in range(1, in_num_joints):
        parent_tx = out_result[in_parent_indices[idx]]
        out_result[idx] = wp.transform_multiply(parent_tx, in_local_pose[idx])


@wp.kernel
def compute_global_pose_kernel(
    in_num_joints     : wp.int32,
    in_root_tx        : wp.transform,
    in_parent_indices : wp.array(dtype=wp.int32),
    in_local_pose     : wp.array(dtype=wp.transform),
    out_result        : wp.array(dtype=wp.transform)
):
    """Kernel wrapper for wp_compute_global_pose."""
    wp_compute_global_pose(in_num_joints, in_root_tx, in_parent_indices, in_local_pose, out_result)


def compute_global_pose(skeleton, local_transforms: List[wp.transform], root_tx=wp.transform_identity()):
    """
    Compute world joint transforms from local joint transforms.
    Args:
        skeleton: A skeleton object containing joint hierarchy information.
        local_transforms (List[wp.transform]): Local joint transforms, one per joint.
        root_tx (wp.transform, optional): The root transformation to apply to the
            entire skeleton. Defaults to identity transform.
    Returns:
        np.ndarray: Array of global transforms for all joints in the skeleton,
            one per joint in hierarchical order.
    Raises:
        ValueError: If the length of local_transforms does not match
            skeleton.num_joints.
    """
    if len(local_transforms) != skeleton.num_joints:
        raise ValueError(f"[ERROR]: Local transform array size [{len(local_transforms)}] doesn't match num_joints [{skeleton.num_joints}]")

    out_global_pose = wp.array([wp.transform_identity()] * skeleton.num_joints, dtype=wp.transform)

    wp.launch(
        compute_global_pose_kernel,
        dim=1,
        inputs=[
            skeleton.num_joints,
            root_tx,
            wp.array(skeleton.parent_indices, dtype=wp.int32),
            wp.array(local_transforms, dtype=wp.transform)],
        outputs=[out_global_pose])

    return out_global_pose.numpy()


def compute_local_pose(skeleton, global_transforms: List[wp.transform], root_tx=wp.transform_identity()):
    """
    Compute local joint transforms from world joint transforms.
    Args:
        skeleton: Skeleton object containing joint hierarchy information.
        global_transforms: List of wp.transform objects representing global transforms for each joint.
        root_tx: Root transform (default: identity transform). Used as the parent transform for root joints.
    Returns:
        np.ndarray: Array of local transforms in parent-relative space, one for each joint in the skeleton.
    Raises:
        ValueError: If the length of global_transforms does not match skeleton.num_joints.
    """
    if len(global_transforms) != skeleton.num_joints:
        raise ValueError(f"[ERROR]: Global transform array size [{len(global_transforms)}] doesn't match num_joints [{skeleton.num_joints}]")

    out_local_pose = wp.array([wp.transform_identity()] * skeleton.num_joints, dtype=wp.transform)

    wp.launch(
        compute_local_pose_kernel,
        dim=1,
        inputs=[
            skeleton.num_joints,
            root_tx,
            wp.array(skeleton.parent_indices, dtype=wp.int32),
            wp.array(global_transforms, dtype=wp.transform)],
        outputs=[out_local_pose])

    return out_local_pose.numpy()


@wp.kernel
def blend_pose_kernel(
    in_local_pose0 : wp.array(dtype=wp.transform),
    in_local_pose1 : wp.array(dtype=wp.transform),
    theta          : wp.float32,
    out_result     : wp.array(dtype=wp.transform)
):
    """Blends between two local poses."""
    idx = wp.tid()
    t = wp.lerp(in_local_pose0[idx].p, in_local_pose1[idx].p, theta)
    q = wp.quat_slerp(in_local_pose0[idx].q, in_local_pose1[idx].q, theta)
    out_result[idx] = wp.transform(t, q)


def blend_poses(pose0_local_transforms: List[wp.transform], pose1_local_transforms: List[wp.transform], blend: wp.float32):
    """
    Blends two poses by interpolating between their local transforms.
    Args:
        pose0_local_transforms (List[wp.transform]): List of local transforms for the first pose.
        pose1_local_transforms (List[wp.transform]): List of local transforms for the second pose.
        blend (wp.float32): Blending factor clamped between 0.0 and 1.0, where 0.0 returns pose0 and 1.0 returns pose1.
    Returns:
        np.ndarray: Blended pose as array of local transforms.
    Raises:
        ValueError: If the number of transforms in pose0_local_transforms differs from pose1_local_transforms.
    """
    num_transforms = len(pose0_local_transforms)
    if num_transforms != len(pose1_local_transforms):
        raise ValueError(f"[ERROR]: Pose0 length [{num_transforms}] is different from Pose1 length [{len(pose1_local_transforms)}]")

    out_pose = wp.empty(shape=num_transforms, dtype=wp.transform)
    wp.launch(
        blend_pose_kernel,
        dim=num_transforms,
        inputs=[
            wp.array(pose0_local_transforms, dtype=wp.transform),
            wp.array(pose1_local_transforms, dtype=wp.transform),
            wp.clamp(blend, 0.0, 1.0)],
        outputs=[out_pose])

    return out_pose.numpy()


def project_hips_to_root(hip_transform : wp.transform, floor_normal=wp.vec3(0.0, 1.0, 0.0)):
    """
    Project a hip transform onto a floor plane and extract a root transform.

    The hip translation is projected onto the plane defined by floor_normal,
    and the orientation is split into a twist around that normal and a
    residual local hip transform.

    Args:
        hip_transform (wp.transform): Hip joint transform in world space.
        floor_normal (wp.vec3, optional): Up/floor normal direction.

    Returns:
        tuple (wp.transform, wp.transform):
            - root_tx: The root transform at floor level.
            - hip_tx: The relative hip transform in root space.
    """
    root_t = math_utils.project_point_to_plane(hip_transform.p, floor_normal)
    root_q = math_utils.quat_twist(floor_normal, hip_transform.q)

    root_tx = wp.transform(root_t, root_q)
    hip_tx = wp.mul(wp.transform_inverse(root_tx), hip_transform)

    return root_tx, hip_tx
