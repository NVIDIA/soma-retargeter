# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import newton
import warp as wp

from enum import IntEnum, auto

import soma_retargeter.io.utils as utils
import soma_retargeter.io.usd as usd_utils
import soma_retargeter.utils.newton_utils as newton_utils

from soma_retargeter.robotics.robot_registry import registry as _robot_registry

class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()

_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}

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
            str(utils.get_asset_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")

def get_source_zero_pose_asset_path(source:SourceType):
    """
    Retrieve the asset path for the zero pose model of a given source type.

    Args:
        source (SourceType): The source type for which the zero pose asset path should be retrieved.

    Returns:
        str: The asset path for the zero pose model.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        return str(utils.get_asset_file('soma', 'soma_zero_frame0.bvh'))

    raise ValueError(f"Unknown source type {source}.")

def get_source_contact_detection_config_path(source:SourceType):
    """
    Retrieve the contact detection configuration path for a given source type.

    Args:
        source (SourceType): The source type for which the contact detection config path should be retrieved.
    """

    if source == SourceType.SOMA:
        return str(utils.get_asset_file('soma', 'contact_processing/soma_contact_config.json'))

    raise ValueError(f"Unknown source type {source}.")

def get_target_asset_path(target: str) -> dict:
    """Return asset-path dict (``urdf_path`` or ``xml_path`` key) for *target*.

    The manifest ``desc`` may also contain ``urdf_offset`` (a precomputed
    ground offset) which is forwarded to the caller.
    """
    entry = _robot_registry.get(target)
    desc = entry.desc
    result: dict = {}
    if "newton_asset" in desc:
        base = str(newton.utils.download_asset(desc["newton_asset"]))
        if "xml_path" in desc:
            result["xml_path"] = os.path.join(base, desc["xml_path"])
        elif "urdf_path" in desc:
            result["urdf_path"] = os.path.join(base, desc["urdf_path"])
    else:
        if "xml_path" in desc:
            result["xml_path"] = os.path.join(entry.manifest_dir, desc["xml_path"])
        elif "urdf_path" in desc:
            result["urdf_path"] = os.path.join(entry.manifest_dir, desc["urdf_path"])
    if not result:
        raise ValueError(f"Manifest for '{target}' must specify 'urdf_path' or 'xml_path' in 'desc'")
    if "urdf_offset" in desc:
        result["urdf_offset"] = desc["urdf_offset"]
    return result

def create_robot_builder(target: str) -> newton.ModelBuilder:
    # For URDFs without a baked ground offset, finalize once to measure and re-build with the correct Z lift
    asset_info = get_target_asset_path(target)
    builder = newton.ModelBuilder()
    if 'xml_path' in asset_info:
        builder.add_mjcf(asset_info['xml_path'])
    elif 'urdf_path' in asset_info:
        urdf_path = asset_info['urdf_path']
        if 'urdf_offset' in asset_info:
            offset = asset_info['urdf_offset']
            builder.add_urdf(
                urdf_path,
                floating=True,
                xform=wp.transform(wp.vec3(0.0, 0.0, offset), wp.quat_identity()))
        else:
            builder.add_urdf(urdf_path, floating=True)
            offset = newton_utils.compute_ground_offset(builder)
            final = newton.ModelBuilder()
            final.add_builder(builder, xform=wp.transform(wp.vec3(0.0, 0.0, offset), wp.quat_identity()))
            return final
    else:
        raise ValueError(f"Invalid asset info for target {target}: {asset_info}")
    return builder


def get_retargeter_config(source: str, target: str) -> dict:
    """Load the retargeter config JSON for a *(source, target)* pair."""
    entry = _robot_registry.get(target)
    rel_path = entry.retarget_configs.get(source)
    if rel_path is None:
        available = ", ".join(sorted(entry.retarget_configs.keys()))
        raise ValueError(
            f"No retargeter config for source [{source}] and target [{target}]. "
            f"Available sources: {available}")
    return utils.load_json(os.path.join(entry.manifest_dir, rel_path))

def resolve_post_processing(retargeter_config: dict, source_type: SourceType, asset_root: str) -> dict | None:
    """Load and merge post-processing configs into one dict for NewtonPipeline.

    When ``post_processing`` is present, ``robot_config`` is always required (it
    supplies ``limb_stabilizer`` and ``contact_correction`` settings).
    Contact detection config is loaded from the source type properties.

    Returns None when ``post_processing`` is not set.
    """
    pp = retargeter_config.get("post_processing")
    if pp is None:
        return None

    robot_rel = pp.get("robot_config")
    if not robot_rel:
        raise ValueError("post_processing requires a non-empty 'robot_config' path")
    robot_data = utils.load_json(os.path.join(asset_root, robot_rel))

    merged: dict = {}

    contact_config_path = get_source_contact_detection_config_path(source_type)
    if contact_config_path:
        source_data = utils.load_json(contact_config_path)

        merged = {k: v for k, v in source_data.items() if k not in ("foot_landmarks_path", "foot_landmarks")}

        if "foot_landmarks" in source_data:
            merged["foot_landmarks"] = source_data["foot_landmarks"]
        else:
            lm_rel = source_data.get("foot_landmarks_path")
            if lm_rel:
                src_dir = os.path.dirname(contact_config_path)
                merged["foot_landmarks_path"] = os.path.normpath(os.path.join(src_dir, lm_rel))

    # Fall back to top-level keys for backward compat with the old flat robot config format
    cc = robot_data.get("contact_correction", robot_data)
    bt = cc.get("transition_frames")
    if bt is not None:
        merged["blend_transition_frames"] = bt
    for key in ("propagation_ratio", "rotation_propagation_ratio", "enable_flatten_foot_plant", "sole_normal_local"):
        if key in cc:
            merged[key] = cc[key]

    if "limb_stabilizer" in robot_data:
        merged["limb_stabilizer"] = robot_data["limb_stabilizer"]

    return merged
