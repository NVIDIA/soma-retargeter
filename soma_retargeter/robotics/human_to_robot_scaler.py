# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

import soma_retargeter.io.utils as utils
import soma_retargeter.utils.pose_utils as pose_utils

from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer


class HumanToRobotScaler:
    """
    Scale and map human motion to robot-aligned effectors.
    """
    def __init__(self, skeleton:Skeleton, config_file):
        config = utils.load_json(config_file)
        self.robot_type = config['robot_type']
        self.skeleton = skeleton

        self.human_scale_ratio = config['human_scale_ratio']
        self.reference_joint_q = config.get('reference_joint_q', None)
        if self.reference_joint_q:
            self.reference_joint_q = wp.array(self.reference_joint_q, dtype=wp.float32)

        human_joint_offset_entries = config.get('human_joint_offsets')
        if not isinstance(human_joint_offset_entries, dict) or not human_joint_offset_entries:
            raise ValueError(
                f"Scaler config [{config_file}] is missing or has an empty "
                "'human_joint_offsets' mapping."
            )

        human_joint_offsets = {}
        for human_name, entry in human_joint_offset_entries.items():
            t, q = entry
            human_joint_offsets[human_name] = wp.transform(wp.vec3(*t), wp.normalize(wp.quat(*q)))

        self.mapped_joints = [name for name in self.skeleton.joint_names if name in human_joint_offsets.keys()]
        joint_indices = [self.skeleton.joint_index(name) for name in self.mapped_joints]

        self.mapped_joint_parents = self._build_mapped_joint_parents(joint_indices)
        self.mapped_joint_indices = wp.array(joint_indices, dtype=wp.int32)
        self.mapped_joint_offsets = wp.array([human_joint_offsets[name] for name in self.mapped_joints], dtype=wp.transform)

    def effector_names(self):
        """
        Return the list of mapped joint names used as effectors.

        Returns:
            list[str]: Names of joints for which effectors are computed.
        """
        return self.mapped_joints

    def compute_effectors_from_skeleton(self, skeleton_instance: SkeletonInstance, scale_animation: bool):
        """
        Compute scaled effectors from a single skeleton instance.

        The method computes global joint transforms from the skeleton instance,
        then applies per-joint scaling and offsets to produce effector
        transforms in world space.

        Args:
            skeleton_instance: SkeletonInstance whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.

        Returns:
            np.ndarray: Array of effector transforms (one per mapped joint) in the
            layout ``(num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``skeleton_instance.skeleton`` does not match the scaler's ``skeleton``.
        """
        if skeleton_instance.skeleton != self.skeleton:
            raise ValueError("[ERROR]: SkeletonInstance.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array(dtype=wp.transform),
            out_result        : wp.array(dtype=wp.transform)
        ):
            pose_utils.wp_compute_global_pose(in_num_joints, in_root_tx, in_parent_indices, in_local_pose, out_result)

        @wp.kernel
        def compute_scaled_effectors_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array(dtype=wp.transform),
            in_scale                : wp.float32,
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            out_result              : wp.array(dtype=wp.transform)
        ):
           HumanToRobotScaler._compute_scaled_effectors(
               in_num_mapped_joints, in_global_pose, in_scale,
               in_mapped_joint_indices, in_mapped_joint_offsets, in_scale_animation, out_result)

        wp_global_pose = wp.array([wp.transform_identity()] * skeleton_instance.num_joints, dtype=wp.transform)
        wp.launch(
            compute_global_pose_kernel,
            dim=1,
            inputs=[
                skeleton_instance.num_joints,
                skeleton_instance.xform,
                wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
                wp.array(skeleton_instance.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_pose])

        wp_effectors = wp.array([wp.transform_identity()] * len(self.mapped_joint_indices), dtype=wp.transform)
        wp.launch(
            compute_scaled_effectors_kernel,
            dim=1,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_pose,
                self.human_scale_ratio,
                self.mapped_joint_indices,
                self.mapped_joint_offsets,
                scale_animation
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    def compute_effectors_from_buffer(self, animation_buffer: AnimationBuffer, scale_animation: bool, xform: wp.transform = wp.transform_identity()):
        """
        Compute scaled effectors for all frames in an animation buffer.

        This is a batched variant of ``compute_effectors_from_skeleton`` that
        operates over all frames in an AnimationBuffer.

        Args:
            animation_buffer: AnimationBuffer whose skeleton must match the scaler's ``skeleton``.
            scale_animation: Whether to apply per-joint scaling when computing
                effectors. If False, only height scaling is applied.
            xform: Optional root transform applied to all frames before global
                pose computation.

        Returns:
            np.ndarray: Array of transforms of shape ``(num_frames, num_mapped_joints, wp.transform)``.

        Raises:
            ValueError: If ``animation_buffer.skeleton`` does not match the scaler's ``skeleton``.
        """
        if animation_buffer.skeleton != self.skeleton:
            raise ValueError("[ERROR]: AnimationBuffer.skeleton is not equal to self.skeleton.")

        @wp.kernel
        def batched_compute_global_pose_kernel(
            in_num_joints     : wp.int32,
            in_root_tx        : wp.transform,
            in_parent_indices : wp.array(dtype=wp.int32),
            in_local_pose     : wp.array2d(dtype=wp.transform),
            out_result        : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            pose_utils.wp_compute_global_pose(
                in_num_joints, in_root_tx, in_parent_indices, in_local_pose[frame_idx], out_result[frame_idx])

        @wp.kernel
        def batched_compute_scaled_effector_kernel(
            in_num_mapped_joints    : wp.int32,
            in_global_pose          : wp.array2d(dtype=wp.transform),
            in_scale                : wp.float32,
            in_mapped_joint_indices : wp.array(dtype=wp.int32),
            in_mapped_joint_offsets : wp.array(dtype=wp.transform),
            in_scale_animation      : wp.bool,
            out_result              : wp.array2d(dtype=wp.transform)
        ):
            frame_idx = wp.tid()
            HumanToRobotScaler._compute_scaled_effectors(
               in_num_mapped_joints, in_global_pose[frame_idx], in_scale,
               in_mapped_joint_indices, in_mapped_joint_offsets, in_scale_animation, out_result[frame_idx])

        wp_global_poses = wp.empty(shape=(animation_buffer.num_frames, self.skeleton.num_joints), dtype=wp.transform)
        wp.launch(
            batched_compute_global_pose_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                self.skeleton.num_joints,
                xform,
                wp.array(self.skeleton.parent_indices, dtype=wp.int32),
                wp.array2d(animation_buffer.local_transforms, dtype=wp.transform)],
                outputs=[wp_global_poses])

        wp_effectors = wp.empty(shape=(animation_buffer.num_frames, len(self.mapped_joint_indices)), dtype=wp.transform)
        wp.launch(
            batched_compute_scaled_effector_kernel,
            dim=animation_buffer.num_frames,
            inputs=[
                len(self.mapped_joint_indices),
                wp_global_poses,
                self.human_scale_ratio,
                self.mapped_joint_indices,
                self.mapped_joint_offsets,
                scale_animation
            ],
            outputs=[wp_effectors])

        return wp_effectors.numpy()

    def create_scaled_skeleton(self, skeleton_instance: SkeletonInstance):
        """
        Create a scaled Skeleton from a skeleton instance.

        This method computes scaled global effectors from the input skeleton
        instance, converts them to local transforms based on the mapped joint
        hierarchy, and returns a new Skeleton containing only the mapped joints.

        Args:
            skeleton_instance: SkeletonInstance to be converted into a scaled skeleton.

        Returns:
            Skeleton: A new skeleton with joints, parents, and local transforms
            derived from the mapped joints and their scaled effectors.
        """
        global_tx = self.compute_effectors_from_skeleton(skeleton_instance, True)

        num_joints = len(self.mapped_joints)
        wp_local_tx = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform)

        wp.launch(
            pose_utils.compute_local_pose_kernel,
            dim=1,
            inputs=[
                num_joints,
                skeleton_instance.xform,
                wp.array(self.mapped_joint_parents, dtype=wp.int32),
                wp.array(global_tx, dtype=wp.transform)],
            outputs=[wp_local_tx])

        return Skeleton(
            num_joints,
            self.mapped_joints,
            self.mapped_joint_parents,
            wp_local_tx.numpy())

    def _build_mapped_joint_parents(self, joint_indices):
        # Walk up the full skeleton hierarchy to find each joint's nearest ancestor that is also mapped
        joint_to_mapped_index = {idx: i for i, idx in enumerate(joint_indices)}

        parent_indices = []
        for joint_idx in joint_indices:
            parent_idx = self.skeleton.joint_parent(joint_idx)
            while parent_idx != -1 and parent_idx not in joint_to_mapped_index:
                parent_idx = self.skeleton.joint_parent(parent_idx)

            parent_indices.append(joint_to_mapped_index.get(parent_idx, -1))

        return parent_indices

    @wp.func
    def _compute_scaled_effectors(
        in_num_mapped_joints    : wp.int32,
        in_global_pose          : wp.array(dtype=wp.transform),
        in_scale                : wp.float32,
        in_mapped_joint_indices : wp.array(dtype=wp.int32),
        in_mapped_joint_offsets : wp.array(dtype=wp.transform),
        in_scale_animation      : wp.bool,
        out_result              : wp.array(dtype=wp.transform)
    ):
        root_t = in_global_pose[in_mapped_joint_indices[0]].p
        scale = wp.where(in_scale_animation, wp.vec3(in_scale), wp.vec3(1.0, 1.0, in_scale))
        scaled_root_t = wp.cw_mul(root_t, scale)

        for i in range(in_num_mapped_joints):
            idx = in_mapped_joint_indices[i]
            pose_tx = in_global_pose[idx]
            offset_tx = in_mapped_joint_offsets[i]

            geocentric_scaled_t = wp.cw_mul((pose_tx.p - root_t), scale)

            q = wp.mul(pose_tx.q, offset_tx.q)
            t = scaled_root_t + geocentric_scaled_t + wp.quat_rotate(q, offset_tx.p)
            out_result[i] = wp.transform(t, q)
