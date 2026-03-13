# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np

import soma_retargeter.utils.pose_utils as pose_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.skeleton import SkeletonInstance


def create_child_parent_map(model):
    """
    Build a mapping between child and parent joints from Newton model.

    Args:
        model: A Newton model object containing joint_parent and joint_child attributes.

    Returns:
        dict: A dictionary where keys are child joint indices and values are their
              corresponding parent joint indices.
    """
    child_parent_map = {}
    joint_parents = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    for i in range(len(joint_parents)):
        parent_index = joint_parents[i]
        child_index = joint_child[i]
        child_parent_map[child_index] = parent_index
    return child_parent_map


def create_joint_coord_masks(model, active_body_masks, default_mask_fill_value):
    """
    Create a joint coord mask array for a Newton model based on specified active body masks.

    Args:
        model: A model object containing joint coordinate information, including:
            - joint_coord_count: Total number of joint coordinates
            - joint_q_start: Array of starting indices for each joint's coordinates
            - joint_dof_dim: Array of DOF dimensions for each joint
            - body_label: List of body labels in order of body indices
        active_body_masks (dict): Dictionary mapping body names to their mask values.
            Only bodies present in this dictionary will have their masks updated.
        default_mask_fill_value (float): The default value to fill the entire mask array with
            before applying active body mask values.

    Returns:
        numpy.ndarray: Array of mask values for each joint coordinate.
    """
    mask_np = np.full(model.joint_coord_count, default_mask_fill_value, dtype=np.float32)
    joint_q_start_np = model.joint_q_start.numpy()
    joint_dof_dim_np = model.joint_dof_dim.numpy()
    body_name_to_idx = {get_name_from_label(k): i for i, k in enumerate(model.body_label)}
    for (key, value) in active_body_masks.items():
        idx = body_name_to_idx[key]
        start_idx = joint_q_start_np[idx]
        dim = joint_dof_dim_np[idx][1]
        mask_np[start_idx:start_idx+dim] = value

    return mask_np


def create_buffer_with_initialization_frames(
        init_pose: SkeletonInstance,
        animation_buffer: AnimationBuffer,
        num_frames_to_insert: int,
        num_stabilization_frames: int):
    """
    Construct a new AnimationBuffer that prepends a sequence of initialization frames
    transitioning smoothly from a given initial pose into an existing animation.
    The generated sequence includes:
      1. Root blending frames (transitioning the global position & orientation)
      2. Joint blending frames (interpolating joint rotations)
      3. Stabilization frames (steady pose holding before animation playback)

    Args:
        init_pose (SkeletonInstance): Starting skeleton pose to initialize from.
        animation_buffer (AnimationBuffer): Existing animation to blend into.
        num_frames_to_insert (int): Total number of transition frames to generate.
        num_stabilization_frames (int): Additional frames to hold the first blended pose.
    Returns:
        AnimationBuffer: A new buffer containing the prepended initialization frames followed by the original animation data.
    """
    num_root_blend_frames = max(0, num_frames_to_insert // 2)
    num_joint_blend_frames = max(0, num_frames_to_insert - num_root_blend_frames)
    num_stabilization_frames = max(0, num_stabilization_frames)

    index_map = np.fromiter(
        (init_pose.skeleton.joint_index(name) for name in animation_buffer.skeleton.joint_names),
        dtype=np.int32,
        count=animation_buffer.skeleton.num_joints)

    mask = index_map != -1
    start_pose = animation_buffer.skeleton.reference_local_transforms
    start_pose[mask] = init_pose.get_local_transforms()[index_map[mask]]
    end_pose = animation_buffer.get_local_transforms(0)

    # Step 1: Root blending from init_pose to first frame of animation_buffer
    start_pose_wp = wp.transform(start_pose[0][:3], start_pose[0][3:])
    root_str_t = wp.transform_get_translation(start_pose_wp)
    root_str_q = wp.transform_get_rotation(start_pose_wp)
    end_pose_wp = wp.transform(end_pose[0][:3], end_pose[0][3:])
    root_end_t = wp.transform_get_translation(end_pose_wp)
    root_end_q = wp.transform_get_rotation(end_pose_wp)
    initialization_poses = []
    for i in range(num_root_blend_frames):
        t = i / (num_root_blend_frames - 1)
        initialization_poses.append(np.copy(start_pose))
        initialization_poses[i][0] = wp.transform(
            wp.lerp(root_str_t, root_end_t, t),
            wp.quat_slerp(root_str_q, root_end_q, t))

    # Step 2: Pose blending from last initialization_poses to first frame of animation_buffer
    start_pose = initialization_poses[-1]
    for i in range(num_joint_blend_frames):
        initialization_poses.append(
            pose_utils.blend_poses(start_pose, end_pose, (i + 1) / num_joint_blend_frames))

    for i in range(num_stabilization_frames):
        initialization_poses.append(end_pose)

    return AnimationBuffer(
        animation_buffer.skeleton,
        animation_buffer.num_frames + num_frames_to_insert + num_stabilization_frames,
        animation_buffer.sample_rate,
        np.concatenate((np.stack(initialization_poses), animation_buffer.local_transforms)))


def get_name_from_label(label: str):
    """Return the leaf component of a hierarchical label.

    Args:
        label: Slash-delimited label string (e.g. ``"robot/link1"``).

    Returns:
        The final path component of the label.
    """
    return label.split("/")[-1]
