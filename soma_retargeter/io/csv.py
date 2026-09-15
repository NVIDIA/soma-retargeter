# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
import csv

from scipy.spatial.transform import Rotation as R
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer

_CSV_EXPORT_ROOT_TITLE = ["Frame", "root_translateX", "root_translateY", "root_translateZ", "root_rotateX", "root_rotateY", "root_rotateZ"]


def load_csv(file_path: str, fps: float = 120.0) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading CSV [{file_path}]")
        csv_data = np.loadtxt(f, delimiter=',', skiprows=1)
        num_frames = csv_data.shape[0]
        # first column is frame index, so skip it
        num_joint_dofs = csv_data.shape[1]# 6 is root translation, root rotation, and 1 is frame index, but we want to include 1 joint for hip
        anim_data = np.zeros((num_frames, num_joint_dofs), dtype=np.float32)
        for i in range(num_frames):
            # first column is frame index, so skip it
            # first 3 are root translation that needs scale
            anim_data[i, 0:3] = csv_data[i, 1:4] * 0.01
            # now next 3 is root rotation that needs to be converted to quaternion
            euler= np.deg2rad(csv_data[i, 4:7])
            quat = wp.quat_rpy(euler[0], euler[1], euler[2])
            anim_data[i, 3:7] = quat
            # note that since we convert 3 eulers to 4 floats quat, now we're in the same index after removing the first column of index
            # next quaternion, we want to copy exactly same, so no touch
            for j in range(7, num_joint_dofs):
                anim_data[i, j] = np.deg2rad(csv_data[i, j])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def save_csv(file_path: str, actuated_joint_names: list[str], buffer: CSVAnimationBuffer) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
		actuated_joint_names (list): List of joints to be written
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_EXPORT_ROOT_TITLE + actuated_joint_names)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)

            t = wp.vec3(*data[0:3]) * 100.0
            q = wp.quat(*data[3:7])
            euler = R.from_quat(q).as_euler('xyz', degrees=True)

            row = [i, t[0], t[1], t[2], euler[0], euler[1], euler[2]]
            row.extend(np.rad2deg(data[7:]))
            writer.writerow(row)
