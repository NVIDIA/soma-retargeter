# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from dataclasses import dataclass
from typing import Protocol, ClassVar, List

import numpy as np
import warp as wp

from scipy.spatial.transform import Rotation as R
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


class RobotCSVConfig(Protocol):
    name: str
    csv_header: List[str]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        ...
    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        ...


@dataclass
class UnitreeG129DOF_CSVConfig:
    name: str = "unitree_g1_29dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        # csv_row 형태: [Frame, tx, ty, tz, rx, ry, rz, joint0 ... joint30]
        tx, ty, tz = csv_row[1:4]
        rx, ry, rz = csv_row[4:7]
        
        # Euler 앵글을 Quaternion(x, y, z, w)으로 변환
        quat = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_quat()
        joints = csv_row[7:]
        
        # [tx, ty, tz, qx, qy, qz, qw, joint0 ... joint30] 배열 반환
        return np.concatenate(([tx, ty, tz], quat, joints))

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> list[float]:
        # anim_row 형태: [tx, ty, tz, qx, qy, qz, qw, joint0 ... joint30]
        tx, ty, tz = anim_row[0:3]
        qx, qy, qz, qw = anim_row[3:7]
        
        # Quaternion을 Euler 앵글로 변환
        euler = R.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=False)
        joints = anim_row[7:]
        
        row = [float(frame_idx), float(tx), float(ty), float(tz), 
               float(euler[0]), float(euler[1]), float(euler[2])]
        row.extend(float(j) for j in joints)
        return row


@dataclass
class H231DOF_CSVConfig:
    name: str = "h2_31dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
        "left_knee_joint_dof", "left_ankle_roll_joint_dof", "left_ankle_pitch_joint_dof",
        "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
        "right_knee_joint_dof", "right_ankle_roll_joint_dof", "right_ankle_pitch_joint_dof",
        "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",
        "head_pitch_joint_dof", "head_yaw_joint_dof",
        "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        return UnitreeG129DOF_CSVConfig.to_anim_frame(self, csv_row)

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        return UnitreeG129DOF_CSVConfig.to_csv_row(self, frame_idx, anim_row)


@dataclass
class T123DOF_CSVConfig:
    name: str = "t1_23dof"
    csv_header: ClassVar[List[str]] = [
        "Frame",
        "root_translateX", "root_translateY", "root_translateZ",
        "root_rotateX", "root_rotateY", "root_rotateZ",
        "AAHead_yaw_dof", "Head_pitch_dof",
        "Left_Shoulder_Pitch_dof", "Left_Shoulder_Roll_dof",
        "Left_Elbow_Pitch_dof", "Left_Elbow_Yaw_dof",
        "Right_Shoulder_Pitch_dof", "Right_Shoulder_Roll_dof",
        "Right_Elbow_Pitch_dof", "Right_Elbow_Yaw_dof",
        "Waist_dof",
        "Left_Hip_Pitch_dof", "Left_Hip_Roll_dof", "Left_Hip_Yaw_dof",
        "Left_Knee_Pitch_dof", "Left_Ankle_Pitch_dof", "Left_Ankle_Roll_dof",
        "Right_Hip_Pitch_dof", "Right_Hip_Roll_dof", "Right_Hip_Yaw_dof",
        "Right_Knee_Pitch_dof", "Right_Ankle_Pitch_dof", "Right_Ankle_Roll_dof"]

    def to_anim_frame(self, csv_row: np.ndarray) -> np.ndarray:
        return UnitreeG129DOF_CSVConfig.to_anim_frame(self, csv_row)

    def to_csv_row(self, frame_idx: int, anim_row: np.ndarray) -> List[float]:
        return UnitreeG129DOF_CSVConfig.to_csv_row(self, frame_idx, anim_row)


def get_csv_config(robot_type: str) -> RobotCSVConfig:
    if robot_type == "unitree_g1":
        return UnitreeG129DOF_CSVConfig()
    if robot_type == "h2":
        return H231DOF_CSVConfig()
    if robot_type == "t1":
        return T123DOF_CSVConfig()

    raise ValueError(f"[ERROR]: Unsupported CSV config for robot type [{robot_type}]")


def load_csv(file_path: str, fps: float = 120.0, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> CSVAnimationBuffer:
    """
    Load a robot motion CSV file into a ``CSVAnimationBuffer``.
    Args:
        file_path (str): Path to the CSV file to load.
        fps (float, optional): Frames per second for the animation. Defaults to 120.0.
        csv_config (RobotCSVConfig, optional): Configuration object that defines how to parse
            CSV rows into animation frames. Defaults to ``UnitreeG129DOF_CSVConfig``.
    Returns:
        CSVAnimationBuffer: An animation buffer containing the loaded and converted animation data.
    Raises:
        FileNotFoundError: If the CSV file at file_path does not exist.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"[INFO]: Loading CSV [{file_path}] for robot [{csv_config.name}]")
        csv_data = np.loadtxt(f, delimiter=",", skiprows=1)
        num_frames = csv_data.shape[0]

        # Each anim row is derived by config, so infer size from first row
        first_row_anim = csv_config.to_anim_frame(csv_data[0])
        anim_data = np.zeros((num_frames, first_row_anim.shape[0]), dtype=np.float32)
        anim_data[0, :] = first_row_anim

        for i in range(1, num_frames):
            anim_data[i, :] = csv_config.to_anim_frame(csv_data[i])

        return CSVAnimationBuffer.create_from_raw_data(anim_data, fps)


def save_csv(file_path: str, buffer: CSVAnimationBuffer, csv_config: RobotCSVConfig = UnitreeG129DOF_CSVConfig()) -> None:
    """
    Save a ``CSVAnimationBuffer`` to a robot motion CSV file.

    Args:
        file_path (str): The path where the CSV file will be saved.
        buffer (CSVAnimationBuffer): The animation buffer containing frame data to be saved.
        csv_config (RobotCSVConfig, optional): Configuration object that defines CSV format and headers.
            Defaults to ``UnitreeG129DOF_CSVConfig``.

    Raises:
        RuntimeError: If the buffer is empty or invalid.
        OSError: If the file cannot be opened or written.
    """
    if buffer is None or buffer.num_frames == 0:
        raise RuntimeError("[ERROR]: Empty or invalid buffer.")

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_config.csv_header)

        for i in range(buffer.num_frames):
            data = buffer.get_data(i)
            row = csv_config.to_csv_row(i, data)
            writer.writerow(row)
