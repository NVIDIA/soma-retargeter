# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np

import soma_retargeter.utils.time_utils as time_utils


class CSVAnimationBuffer:
    """
    A buffer for managing CSV-based animation data with frame interpolation.
    This class stores animation frames from CSV data and provides methods to retrieve
    and sample animation data at specific times with linear interpolation.
    """
    def __init__(self):
        """
        Initialize a ``CSVAnimationBuffer`` object.
        """
        self.num_frames = 0
        self.data = []
        self.sample_rate = 0
        self.xform = wp.transform_identity()

    @staticmethod
    def create_from_raw_data(raw_data, sample_rate, offset_tx: wp.transform = wp.transform_identity()):
        """
        Create a ``CSVAnimationBuffer`` instance from raw CSV data.

        Args:
            raw_data: The raw CSV animation data to be stored in the buffer.
            sample_rate: The sample rate (frames per second) of the animation data.
            offset_tx: An optional transform representing the initial offset transformation.
                       Defaults to identity transform if not provided.

        Returns:
            CSVAnimationBuffer: A new CSVAnimationBuffer instance populated with the provided data,
                               sample rate, and transformation offset.
        """
        buffer = CSVAnimationBuffer()
        buffer.num_frames = len(raw_data)
        buffer.data = raw_data
        buffer.sample_rate = sample_rate
        buffer.xform = offset_tx
        return buffer

    def get_data(self, frame):
        """
        Retrieve CSV animation data for a specific frame.
        Args:
            frame (int): The frame index to retrieve data from.
        Returns:
            The CSV animation data at the specified frame.
        Raises:
            ValueError: If the frame index is out of bounds for the number of frames.
        """
        if frame < 0 or frame >= self.num_frames:
            raise ValueError(f"[ERROR]: frame {frame} is out of bounds for num_frames {self.num_frames}")

        return self.data[frame]

    def sample(self, time):
        """
        Sample CSV animation data at a given time with interpolation.
        This method retrieves CAV animation frame data at the specified time and performs
        interpolation between frames when the time falls between two keyframes.
        Args:
            time (float): The time at which to sample the animation, in seconds.
        Returns:
            np.ndarray: A concatenated array resulting joint coordinates.
        """
        frame, blend = time_utils.frame_index_from_time(time, self.sample_rate, self.num_frames)

        if blend < 1e-5:
            root_tx = wp.mul(self.xform, wp.transform(*(self.data[frame][:7])))
            return np.concatenate((root_tx, self.data[frame][7:]))
        else:
            root_tx0 = wp.mul(self.xform, wp.transform(*(self.data[frame][:7])))
            root_tx1 = wp.mul(self.xform, wp.transform(*(self.data[frame + 1][:7])))
            out_root_tx = wp.transform(
                wp.lerp(wp.transform_get_translation(root_tx0), wp.transform_get_translation(root_tx1), blend),
                wp.quat_slerp(wp.transform_get_rotation(root_tx0), wp.transform_get_rotation(root_tx1), blend))

            # Linear blend joint DOFs
            out_joint_dofs = self.data[frame][7:] * (1.0 - blend) + self.data[frame + 1][7:] * blend

            return np.concatenate((out_root_tx, out_joint_dofs))
