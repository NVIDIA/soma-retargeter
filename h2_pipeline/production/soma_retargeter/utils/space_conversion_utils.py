# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto

import warp as wp


class FacingDirectionType(IntEnum):
    """Enumeration of supported facing directions for source models."""
    MAYA = auto(),
    MUJOCO = auto()

_FACING_DIRECTION_TYPE_TO_STR = {
    FacingDirectionType.MAYA : "Maya",
    FacingDirectionType.MUJOCO : "Mujoco"
}
_STR_TO_FACING_DIRECTION_TYPE = {str : type for type, str in _FACING_DIRECTION_TYPE_TO_STR.items()}


def get_facing_direction_str_from_type(facing_direction: FacingDirectionType) -> str:
    """
    Get the string name associated with a given facing direction type.

    Args:
        facing_direction (FacingDirectionType): The facing direction type enum value.

    Returns:
        str: The string representation of the facing direction type.
    """
    return _FACING_DIRECTION_TYPE_TO_STR[facing_direction]


def get_facing_direction_type_from_str(facing_direction: str) -> FacingDirectionType:
    """
    Get the string name associated with a given facing direction type.

    Args:
        facing_direction (str): The facing direction string value.

    Returns:
        FacingDirectionType: The facing direction type enum value.

    Raises:
        ValueError: If the provided facing direction string does not correspond to any known type.
    """
    try:
        return _STR_TO_FACING_DIRECTION_TYPE[facing_direction]
    except KeyError:
        allowed = ", ".join(_STR_TO_FACING_DIRECTION_TYPE.keys())
        raise ValueError(f"Unknown facing direction type: [{facing_direction}]. Allowed values: {allowed}") from None


class SpaceConverter:
    """
    Utility class for converting between different coordinate spaces, such as from Maya or Mujoco to
    the internal representation used in Newton.
    """
    def __init__(self, facing_direction: FacingDirectionType):
        if facing_direction == FacingDirectionType.MUJOCO:
            self.converter = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.radians(90.0))
        elif facing_direction == FacingDirectionType.MAYA:
            q1 = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), wp.radians(90.0))
            q2 = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(90.0))
            self.converter = q1 * q2
        else:
            self.converter = wp.quat_identity()

        self.converter_inverse = wp.quat_inverse(self.converter)

    def convert_position(self, position, scale=1.0):
        """Convert a position from the source coordinate space to the internal representation."""
        return wp.quat_rotate(self.converter, wp.vec3(position[0]*scale, position[1]*scale, position[2]*scale))

    def convert_rotation(self, quat):
        """Convert a rotation from the source coordinate space to the internal representation."""
        return self.converter * quat * self.converter_inverse

    def inverse_convert_position(self, position, scale=1.0):
        """Convert a position from the internal representation back to the source coordinate space."""
        return wp.quat_rotate(self.converter_inverse, wp.vec3(position[0]*scale, position[1]*scale, position[2]*scale))

    def inverse_convert_rotation(self, quat):
        """Convert a rotation from the internal representation back to the source coordinate space."""
        return self.converter_inverse * quat * self.converter

    def transform(self, transform):
        """Convert a transform from the source coordinate space to the internal representation."""
        return wp.mul(wp.transform(wp.vec3(0, 0, 0), self.converter), transform)
    