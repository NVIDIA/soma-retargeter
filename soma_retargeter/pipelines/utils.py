# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum, auto
import os
from pathlib import Path

import newton
import numpy as np

import soma_retargeter.utils.io_utils as io_utils
import soma_retargeter.assets.usd as usd_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


class TargetType(IntEnum):
    """Enumeration of supported target model types."""
    UNITREE_G1 = auto()
    GR1T2 = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

_TARGET_TYPE_TO_STR = {
    TargetType.UNITREE_G1 : "unitree_g1",
    TargetType.GR1T2 : "gr1t2"
}
_STR_TO_TARGET_TYPE = {s : t for t, s in _TARGET_TYPE_TO_STR.items()}


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def get_target_str_from_type(target: TargetType) -> str:
    """
    Get the string name associated with a given target type.

    Args:
        target (TargetType): The target type enum value.

    Returns:
        str: The string representation of the target type.
    """
    return _TARGET_TYPE_TO_STR[target]


def get_target_type_from_str(target: str) -> TargetType:
    """
    Convert a string to its corresponding TargetType enum value.

    Args:
        target (str): The string representation of a target.

    Returns:
        TargetType: The corresponding target type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid target type.
    """
    try:
        return _STR_TO_TARGET_TYPE[target]
    except KeyError:
        allowed = ", ".join(_STR_TO_TARGET_TYPE.keys())
        raise ValueError(f"Unknown target type: [{target}]. Allowed values: {allowed}") from None


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


def get_retargeter_config(source: SourceType, target: TargetType) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType): The source type.
        target (TargetType): The target type.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    if source != SourceType.SOMA:
        raise ValueError(f"Unknown source type [{source}] for target [{target}].")

    configs = {
        TargetType.UNITREE_G1: ('unitree_g1', 'soma_to_g1_retargeter_config.json'),
        TargetType.GR1T2: ('gr1t2', 'soma_to_gr1t2_retargeter_config.json'),
    }
    try:
        folder, filename = configs[target]
    except KeyError:
        raise ValueError(f"Unknown target type [{target}].") from None
    return io_utils.load_json(io_utils.get_config_file(folder, filename))


def resolve_model_path(target: TargetType, config: dict) -> Path:
    """Resolve a target MJCF while preserving the upstream G1 asset behavior."""

    if target == TargetType.UNITREE_G1:
        return newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml"
    try:
        configured_path = config["model_path"]
    except KeyError:
        raise ValueError(
            f"Retargeter config for {get_target_str_from_type(target)!r} requires model_path"
        ) from None
    path = Path(os.path.expanduser(os.path.expandvars(configured_path)))
    if not path.is_file():
        raise FileNotFoundError(
            f"Target MJCF does not exist: {path}. Set CIBO_ROOT or update model_path."
        )
    return path.resolve()


def build_and_validate_robot(target: TargetType, config: dict) -> newton.ModelBuilder:
    """Load the configured MJCF and reject name/order/limit errors before IK."""

    model_path = resolve_model_path(target, config)
    builder = newton.ModelBuilder()
    builder.add_mjcf(model_path)

    body_names = [label.split("/")[-1] for label in builder.body_label]
    if len(body_names) != len(set(body_names)):
        raise ValueError(f"Target MJCF contains duplicate body names: {model_path}")

    actuated_indices = [
        index for index, joint_type in enumerate(builder.joint_type)
        if joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC)
    ]
    joint_names = [builder.joint_label[index].split("/")[-1] for index in actuated_indices]
    if len(joint_names) != len(set(joint_names)):
        raise ValueError(f"Target MJCF contains duplicate actuated joint names: {model_path}")

    expected_joint_names = config.get("joint_names")
    if expected_joint_names is not None and joint_names != expected_joint_names:
        raise ValueError(
            "Target MJCF joint order mismatch:\n"
            f"expected={expected_joint_names}\nactual={joint_names}"
        )

    required_bodies = set(config.get("required_bodies", []))
    for mapping in config.get("ik_map", {}).values():
        required_bodies.update((mapping["t_body"], mapping["r_body"]))
    missing_bodies = sorted(required_bodies - set(body_names))
    if missing_bodies:
        raise ValueError(f"Target MJCF is missing required bodies: {missing_bodies}")

    lower = np.asarray(builder.joint_limit_lower, dtype=np.float64)
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float64)
    invalid_limits = []
    for index, name in zip(actuated_indices, joint_names):
        dof_start = builder.joint_qd_start[index]
        if not np.isfinite(lower[dof_start]) or not np.isfinite(upper[dof_start]):
            invalid_limits.append(name)
        elif lower[dof_start] > upper[dof_start]:
            invalid_limits.append(name)
    if invalid_limits:
        raise ValueError(f"Target MJCF has invalid joint limits: {invalid_limits}")

    expected_dofs = config.get("num_actuated_dofs")
    if expected_dofs is not None and len(joint_names) != expected_dofs:
        raise ValueError(
            f"Target MJCF must have {expected_dofs} actuated DOFs, got {len(joint_names)}"
        )
    return builder
