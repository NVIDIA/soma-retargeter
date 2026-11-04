# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import soma_retargeter.utils.pose_utils as pose_utils
import soma_retargeter.utils.time_utils as time_utils


class AnimationBuffer:
    """
    Time-sampled local joint transforms for a given skeleton.

    Stores a fixed number of frames of local-space joint transforms and a sample
    rate, and provides utilities for per-frame access, global transform
    computation, and time-based interpolation.
    """
    def __init__(self, skeleton, num_frames: int, sample_rate: float, local_transforms=None):
        """
        Initialize an AnimationBuffer instance.

        Args:
            skeleton: The skeleton structure defining joint hierarchy and reference pose.
            num_frames (int): The number of animation frames in the buffer.
            sample_rate (float): The sampling rate of the animation (frames per second).
            local_transforms (np.ndarray, optional): Array of shape (num_frames, num_joints, wp.transform)
                containing local transformation matrices for each joint at each frame.
                If None, the buffer is initialized with the skeleton's reference local transforms
                repeated for all frames. Defaults to None.
        """
        self.skeleton = skeleton
        self.num_frames = num_frames
        self.sample_rate = sample_rate

        if local_transforms is None:
            # Fill local_transforms from default skeleton pose
            self.local_transforms = np.zeros((num_frames, self.skeleton.num_joints), dtype=wp.transform)
            self.local_transforms[:] = np.tile(skeleton.reference_local_transforms[None, :, :], (num_frames, 1, 1))
        else:
            self.local_transforms = local_transforms

    def get_local_transforms(self, frame):
        """
        Retrieve the local transforms for a specific frame.
        Args:
            frame (int): The frame index to retrieve transforms from.
        Returns:
            Local transforms for the specified frame.
        Raises:
            ValueError: If frame index is negative or exceeds the total number of frames.
        """
        if frame < 0 or frame >= self.num_frames:
            raise ValueError(f"Frame index [{frame}] should be in [0, {self.num_frames}).")

        return self.local_transforms[frame]

    def compute_global_transforms(self, frame, root_tx=wp.transform_identity()):
        """
        Compute global transforms for a given frame.
        Converts local joint transforms to global transforms in world space by applying
        forward kinematics from the skeleton root.
        Args:
            frame (int): The frame index at which to compute global transforms.
                Must be in range [0, num_frames).
            root_tx (wp.transform, optional): The root transform to apply as the base
                transformation. Defaults to identity transform.
        Returns:
            Global transforms for all joints in the skeleton at the specified frame.
        Raises:
            ValueError: If frame index is out of valid range [0, num_frames).
        """
        if frame < 0 or frame >= self.num_frames:
            raise ValueError(f"Frame index [{frame}] should be in [0, {self.num_frames}).")

        return pose_utils.compute_global_pose(self.skeleton, self.local_transforms[frame], root_tx)

    def sample(self, time):
        """
        Sample the animation at an arbitrary time using linear interpolation.
        Args:
            time (float): The time in seconds at which to sample the animation.
        Returns:
            np.ndarray: The interpolated local transforms at the given time.
                       If blend is negligible, returns a copy of the frame at the exact time.
                       Otherwise, returns a blended pose between two consecutive frames.
        """
        frame, blend = time_utils.frame_index_from_time(time, self.sample_rate, self.num_frames)

        if blend <= 1e-5:
            return np.copy(self.local_transforms[frame])

        return pose_utils.blend_poses(self.local_transforms[frame], self.local_transforms[frame + 1], blend)


def create_animation_buffer_for_skeleton(animation_buffer, new_skeleton):
    """
    Creates an ``AnimationBuffer`` retargeted to a new skeleton by joint name.

    The function copies local transforms for all joints whose names exist in both
    skeletons. Frames and sample rate are preserved. If the new skeleton object is
    the same as the original, the input buffer is returned unchanged.
    Args:
        animation_buffer (AnimationBuffer): The source animation buffer with animation data.
        new_skeleton (Skeleton): The target skeleton to create the animation buffer for.
    Returns:
        AnimationBuffer: A new AnimationBuffer for the new_skeleton with remapped animation data.
                        If the skeleton is already the same, returns the original animation_buffer unchanged.
    """
    if animation_buffer.skeleton is not new_skeleton:

        skeleton = animation_buffer.skeleton

        # Create a new buffer initialized to the new skeleton's reference pose.
        new_animation_buffer = AnimationBuffer(new_skeleton, animation_buffer.num_frames, animation_buffer.sample_rate)
        local_transforms = animation_buffer.local_transforms

        # Copy transforms for joints that exist in both skeletons by name.
        for i in range(skeleton.num_joints):
            joint_name = skeleton.joint_name(i)

            new_joint_index = new_skeleton.joint_index(joint_name)
            if new_joint_index != -1:
                new_animation_buffer.local_transforms[:, new_joint_index] = local_transforms[:, i]

        return new_animation_buffer
    else:
        return animation_buffer
