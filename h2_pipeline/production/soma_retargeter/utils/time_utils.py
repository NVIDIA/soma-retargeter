# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp


def frame_index_from_time(time, sample_rate, num_frames):
    """
    Convert a time value to a frame index and blend factor.

    Given a time (in seconds) and a sample rate (frames per second), this computes
    the integer frame index in a sequence and a 0–1 blend factor toward the next
    frame, clamped to the valid frame range [0, num_frames - 1].
    
    Args:
        time (float): The time value in seconds.
        sample_rate (float): The sampling rate (frames per second).
        num_frames (int): The total number of frames in the sequence.
    
    Returns:
        tuple (int, float):
            - frame: The current frame index, clamped between 0 and num_frames-1.
            - next_frame_blend: The blending factor for interpolation with the next frame,
                                in the range [0.0, 1.0]. A value of 0.0 means use only the
                                current frame, and 1.0 means use only the next frame.
    """
    last_frame = wp.max(0, num_frames - 1)
    fractional = time * sample_rate
    frame = wp.clamp(int(wp.floor(fractional)), 0, last_frame)
    next_frame_blend = 0.0
    if frame < last_frame:
        next_frame_blend = wp.clamp(fractional - frame, 0.0, 1.0)

    return frame, next_frame_blend
