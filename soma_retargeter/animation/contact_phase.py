# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import warp as wp

import soma_retargeter.utils.pose_utils as pose_utils


_EPS = 1e-8
_SCHEMA_VERSION = 1
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
_DEFAULT_FOOT_LANDMARK_CONFIG_PATHS = {
    "soma": _ASSET_ROOT / "contact_processing" / "soma_contact_config.json",
}
_SIDE_KEYS = ("left", "right")
POSITION_LANDMARK_FIELDS = (
    "heel_local_in_foot",
    "toe_pivot_local_in_toe",
)
LANDMARK_EDITOR_FIELDS = POSITION_LANDMARK_FIELDS + ("sole_normal_local",)
_LANDMARK_COUNT = 2
_HEEL_IDX = 0
_TOE_PIVOT_IDX = 1


@dataclass(frozen=True)
class _FootLandmarkHeuristicConfig:
    skeleton_type: str
    heel_back_ratio: float
    sole_drop_ratio: float
    sole_normal_space: str
    left_foot_joint: str = "LeftFoot"
    right_foot_joint: str = "RightFoot"
    left_toe_joint: str = "LeftToeBase"
    right_toe_joint: str = "RightToeBase"
    left_toe_end_joint: str | None = None
    right_toe_end_joint: str | None = None


@dataclass
class FootLandmarkRuntime:
    foot_index: int
    foot_joint: str
    toe_joint: str
    toe_end_joint: str | None
    foot_joint_index: int
    toe_joint_index: int
    toe_end_joint_index: int
    heel_local_in_foot: np.ndarray
    toe_pivot_local_in_toe: np.ndarray
    sole_normal_local: np.ndarray
    sole_normal_space: str


@dataclass
class FootLandmarkModel:
    skeleton_type: str
    feet: list[FootLandmarkRuntime]
    config_path: str | None = None


@dataclass(frozen=True)
class FootLandmarkData:
    heel_positions: np.ndarray
    toe_pivot_positions: np.ndarray
    sole_normals: np.ndarray


_HEURISTIC_CONFIGS = {
    "bones": _FootLandmarkHeuristicConfig(
        skeleton_type="bones",
        heel_back_ratio=0.58,
        sole_drop_ratio=0.10,
        sole_normal_space="foot",
    ),
    "soma": _FootLandmarkHeuristicConfig(
        skeleton_type="soma",
        heel_back_ratio=0.55,
        sole_drop_ratio=0.08,
        sole_normal_space="foot",
        left_toe_end_joint="LeftToeEnd",
        right_toe_end_joint="RightToeEnd",
    ),
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = q[:3]
    t = 2.0 * np.cross(u, v)
    return v + q[3] * t + np.cross(u, t)


def _transform_point_np(row: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    return row[:3] + _quat_rotate_np(row[3:7], point_local)


def _inverse_transform_point_np(row: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    return _quat_rotate_np(_quat_conjugate(row[3:7]), point_world - row[:3])


def _inverse_rotate_vector_np(row: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    return _quat_rotate_np(_quat_conjugate(row[3:7]), world_vec)


def _project_to_plane(vec: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    plane_normal = _normalize(plane_normal)
    return np.asarray(vec, dtype=np.float64) - plane_normal * np.dot(vec, plane_normal)


def _as_float32_triplet(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(3)


@wp.func
def _wp_safe_normalize(v: wp.vec3) -> wp.vec3:
    length = wp.length(v)
    return wp.where(length > 1.0e-6, v / length, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def _project_foot_landmarks_kernel(
    foot_joint_indices: wp.array(dtype=wp.int32),
    toe_joint_indices: wp.array(dtype=wp.int32),
    normal_joint_indices: wp.array(dtype=wp.int32),
    global_tx: wp.array2d(dtype=wp.transform),
    heel_local: wp.array(dtype=wp.vec3),
    toe_pivot_local: wp.array(dtype=wp.vec3),
    sole_normal_local: wp.array(dtype=wp.vec3),
    out_landmark_positions: wp.array3d(dtype=wp.vec3),
    out_sole_normals: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()
    frame_idx = tid // 2
    foot_idx = tid - frame_idx * 2

    foot_tx = global_tx[frame_idx, foot_joint_indices[foot_idx]]
    toe_tx = global_tx[frame_idx, toe_joint_indices[foot_idx]]
    normal_tx = global_tx[frame_idx, normal_joint_indices[foot_idx]]

    out_landmark_positions[frame_idx, foot_idx, _HEEL_IDX] = wp.transform_point(foot_tx, heel_local[foot_idx])
    out_landmark_positions[frame_idx, foot_idx, _TOE_PIVOT_IDX] = wp.transform_point(toe_tx, toe_pivot_local[foot_idx])
    out_sole_normals[frame_idx, foot_idx] = _wp_safe_normalize(
        wp.quat_rotate(normal_tx.q, sole_normal_local[foot_idx])
    )


def _stack_landmark_positions(landmarks: FootLandmarkData) -> np.ndarray:
    return np.stack(
        [
            landmarks.heel_positions,
            landmarks.toe_pivot_positions,
        ],
        axis=2,
    ).astype(np.float32)


def _unstack_landmark_positions(
    positions: np.ndarray,
    sole_normals: np.ndarray,
) -> FootLandmarkData:
    return FootLandmarkData(
        heel_positions=positions[:, :, _HEEL_IDX].astype(np.float32),
        toe_pivot_positions=positions[:, :, _TOE_PIVOT_IDX].astype(np.float32),
        sole_normals=sole_normals.astype(np.float32),
    )


def _kernel_landmark_inputs(model: FootLandmarkModel) -> tuple[np.ndarray, ...]:
    foot_joint_indices = np.array([foot.foot_joint_index for foot in model.feet], dtype=np.int32)
    toe_joint_indices = np.array([foot.toe_joint_index for foot in model.feet], dtype=np.int32)
    normal_joint_indices = np.array(
        [
            foot.foot_joint_index if foot.sole_normal_space == "foot" else foot.toe_joint_index
            for foot in model.feet
        ],
        dtype=np.int32,
    )
    return (
        foot_joint_indices,
        toe_joint_indices,
        normal_joint_indices,
        np.stack([foot.heel_local_in_foot for foot in model.feet], axis=0).astype(np.float32),
        np.stack([foot.toe_pivot_local_in_toe for foot in model.feet], axis=0).astype(np.float32),
        np.stack([foot.sole_normal_local for foot in model.feet], axis=0).astype(np.float32),
    )


def _mirror_landmark_vector_lr(values: np.ndarray | list[float]) -> np.ndarray:
    mirrored = np.asarray(values, dtype=np.float32).copy()
    mirrored[1] *= -1.0
    return mirrored.astype(np.float32)


def copy_foot_landmark_model(model: FootLandmarkModel) -> FootLandmarkModel:
    """Return a deep copy of *model* with all per-foot arrays independently owned.

    Args:
        model: Source landmark model to copy.

    Returns:
        A new ``FootLandmarkModel`` whose foot arrays do not share memory with the original.
    """
    copied_feet = []
    for foot in model.feet:
        copied_feet.append(
            FootLandmarkRuntime(
                foot_index=foot.foot_index,
                foot_joint=foot.foot_joint,
                toe_joint=foot.toe_joint,
                toe_end_joint=foot.toe_end_joint,
                foot_joint_index=foot.foot_joint_index,
                toe_joint_index=foot.toe_joint_index,
                toe_end_joint_index=foot.toe_end_joint_index,
                heel_local_in_foot=foot.heel_local_in_foot.copy(),
                toe_pivot_local_in_toe=foot.toe_pivot_local_in_toe.copy(),
                sole_normal_local=foot.sole_normal_local.copy(),
                sole_normal_space=foot.sole_normal_space,
            )
        )
    return FootLandmarkModel(
        skeleton_type=model.skeleton_type,
        feet=copied_feet,
        config_path=model.config_path,
    )


def copy_mirrored_foot_landmarks(model: FootLandmarkModel, src_index: int, dst_index: int) -> None:
    """Copy landmark vectors from *src_index* to *dst_index*, mirroring across the sagittal plane.

    Each vector is copied with its Y component negated, converting a left-foot
    landmark into a right-foot landmark (or vice versa).

    Args:
        model: Landmark model to modify in-place.
        src_index: Foot index to copy from (0 = left, 1 = right).
        dst_index: Foot index to write to.
    """
    source = model.feet[src_index]
    target = model.feet[dst_index]
    for field_name in LANDMARK_EDITOR_FIELDS:
        setattr(target, field_name, _mirror_landmark_vector_lr(getattr(source, field_name)))


def normalize_foot_landmark_normal(foot: FootLandmarkRuntime) -> None:
    foot.sole_normal_local = _normalize(foot.sole_normal_local).astype(np.float32)


def _load_landmark_library(path: str | Path) -> dict | None:
    landmark_path = Path(path)
    if not landmark_path.exists():
        return None
    with landmark_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if "foot_landmarks" in data:
        return data["foot_landmarks"]
    return data


def _extract_rig_data(library: dict, skeleton_type: str) -> dict | None:
    if library is None:
        return None
    rigs = library.get("rigs")
    if isinstance(rigs, dict) and skeleton_type in rigs:
        return rigs[skeleton_type]
    direct = library.get(skeleton_type)
    if isinstance(direct, dict):
        return direct
    return None


def _get_heuristic_landmark_rig_config(skeleton_type: str) -> _FootLandmarkHeuristicConfig:
    try:
        return _HEURISTIC_CONFIGS[skeleton_type]
    except KeyError as exc:
        allowed = ", ".join(sorted(_HEURISTIC_CONFIGS.keys()))
        raise ValueError(f"Unsupported skeleton type '{skeleton_type}'. Expected one of: {allowed}") from exc


def get_default_foot_landmark_config_path(skeleton_type: str) -> Path:
    """Return the default foot-landmark JSON config path for *skeleton_type*.

    Args:
        skeleton_type: Registered skeleton type key (e.g. ``"soma"``).

    Returns:
        Absolute ``Path`` to the config file bundled with the package.

    Raises:
        ValueError: If *skeleton_type* is not registered.
    """
    try:
        return _DEFAULT_FOOT_LANDMARK_CONFIG_PATHS[skeleton_type]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DEFAULT_FOOT_LANDMARK_CONFIG_PATHS.keys()))
        raise ValueError(f"Unsupported skeleton type '{skeleton_type}'. Expected one of: {allowed}") from exc


def _compute_reference_global_tx(skeleton, root_tx: wp.transform) -> np.ndarray:
    ref_pose = np.expand_dims(skeleton.reference_local_transforms, axis=0)
    return pose_utils.compute_global_poses_batch(skeleton, ref_pose, root_tx)[0]


def _build_one_side_heuristic_runtime(
    ref_global_tx: np.ndarray,
    skeleton,
    foot_index: int,
    foot_joint: str,
    toe_joint: str,
    toe_end_joint: str | None,
    rig_config: _FootLandmarkHeuristicConfig,
) -> FootLandmarkRuntime:
    foot_joint_index = skeleton.joint_index(foot_joint)
    toe_joint_index = skeleton.joint_index(toe_joint)
    toe_end_joint_index = skeleton.joint_index(toe_end_joint) if toe_end_joint else -1
    if foot_joint_index < 0 or toe_joint_index < 0:
        raise ValueError(f"Missing foot landmarks for {foot_joint=} {toe_joint=}")

    foot_row = ref_global_tx[foot_joint_index]
    toe_row = ref_global_tx[toe_joint_index]
    foot_pos = foot_row[:3].astype(np.float64)
    toe_pos = toe_row[:3].astype(np.float64)

    toe_end_pos = None
    if toe_end_joint_index >= 0:
        toe_end_pos = ref_global_tx[toe_end_joint_index][:3].astype(np.float64)

    forward_world = toe_end_pos - toe_pos if toe_end_pos is not None else toe_pos - foot_pos
    if np.linalg.norm(forward_world) <= _EPS:
        forward_world = toe_pos - foot_pos
    forward_world = _normalize(forward_world)

    sole_down_world = _project_to_plane(-_WORLD_UP, forward_world)
    if np.linalg.norm(sole_down_world) <= _EPS:
        sole_down_world = _quat_rotate_np(foot_row[3:7], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    sole_down_world = _normalize(sole_down_world)

    foot_to_toe_len = max(float(np.linalg.norm(toe_pos - foot_pos)), 0.05)
    sole_drop = rig_config.sole_drop_ratio * foot_to_toe_len

    heel_world = foot_pos - forward_world * (rig_config.heel_back_ratio * foot_to_toe_len) + sole_down_world * sole_drop
    toe_pivot_world = toe_pos + sole_down_world * sole_drop

    normal_row = foot_row if rig_config.sole_normal_space == "foot" else toe_row

    return FootLandmarkRuntime(
        foot_index=foot_index,
        foot_joint=foot_joint,
        toe_joint=toe_joint,
        toe_end_joint=toe_end_joint,
        foot_joint_index=foot_joint_index,
        toe_joint_index=toe_joint_index,
        toe_end_joint_index=toe_end_joint_index,
        heel_local_in_foot=_inverse_transform_point_np(foot_row, heel_world).astype(np.float32),
        toe_pivot_local_in_toe=_inverse_transform_point_np(toe_row, toe_pivot_world).astype(np.float32),
        sole_normal_local=_normalize(_inverse_rotate_vector_np(normal_row, sole_down_world)).astype(np.float32),
        sole_normal_space=rig_config.sole_normal_space,
    )


def _runtime_to_serializable(foot: FootLandmarkRuntime) -> dict:
    return {
        "foot_joint": foot.foot_joint,
        "toe_joint": foot.toe_joint,
        "toe_end_joint": foot.toe_end_joint,
        "sole_normal_space": foot.sole_normal_space,
        "heel_local_in_foot": foot.heel_local_in_foot.tolist(),
        "toe_pivot_local_in_toe": foot.toe_pivot_local_in_toe.tolist(),
        "sole_normal_local": foot.sole_normal_local.tolist(),
    }


def _runtime_from_serialized(
    skeleton,
    foot_index: int,
    side_data: dict,
) -> FootLandmarkRuntime:
    foot_joint = side_data["foot_joint"]
    toe_joint = side_data["toe_joint"]
    toe_end_joint = side_data.get("toe_end_joint")
    foot_joint_index = skeleton.joint_index(foot_joint)
    toe_joint_index = skeleton.joint_index(toe_joint)
    toe_end_joint_index = skeleton.joint_index(toe_end_joint) if toe_end_joint else -1
    if foot_joint_index < 0 or toe_joint_index < 0:
        raise ValueError(f"Serialized landmark references missing joints: {foot_joint}, {toe_joint}")
    return FootLandmarkRuntime(
        foot_index=foot_index,
        foot_joint=foot_joint,
        toe_joint=toe_joint,
        toe_end_joint=toe_end_joint,
        foot_joint_index=foot_joint_index,
        toe_joint_index=toe_joint_index,
        toe_end_joint_index=toe_end_joint_index,
        heel_local_in_foot=_as_float32_triplet(side_data["heel_local_in_foot"]),
        toe_pivot_local_in_toe=_as_float32_triplet(side_data["toe_pivot_local_in_toe"]),
        sole_normal_local=_normalize(_as_float32_triplet(side_data["sole_normal_local"])).astype(np.float32),
        sole_normal_space=str(side_data.get("sole_normal_space", "foot")),
    )


def build_heuristic_foot_landmark_model(skeleton, skeleton_type: str, root_tx: wp.transform) -> FootLandmarkModel:
    """Build a ``FootLandmarkModel`` using anatomy heuristics on the reference pose.

    Does not require an authored config; derives landmark positions from
    foot-to-toe bone lengths and fixed ratio constants per skeleton type.

    Args:
        skeleton: Source skeleton definition.
        skeleton_type: Registered skeleton key (e.g. ``"soma"``).
        root_tx: Root space-conversion transform (e.g. Maya-to-Newton).

    Returns:
        A ``FootLandmarkModel`` with both feet populated.
    """
    rig_config = _get_heuristic_landmark_rig_config(skeleton_type)
    ref_global_tx = _compute_reference_global_tx(skeleton, root_tx)

    left = _build_one_side_heuristic_runtime(
        ref_global_tx,
        skeleton,
        0,
        rig_config.left_foot_joint,
        rig_config.left_toe_joint,
        rig_config.left_toe_end_joint,
        rig_config,
    )
    right = _build_one_side_heuristic_runtime(
        ref_global_tx,
        skeleton,
        1,
        rig_config.right_foot_joint,
        rig_config.right_toe_joint,
        rig_config.right_toe_end_joint,
        rig_config,
    )
    return FootLandmarkModel(skeleton_type=skeleton_type, feet=[left, right])


def load_authored_foot_landmark_model(
    skeleton,
    skeleton_type: str,
    config_path_or_data: str | Path | dict | None = None,
) -> FootLandmarkModel | None:
    """Load a ``FootLandmarkModel`` from an authored JSON config.

    Args:
        skeleton: Source skeleton used to resolve joint indices.
        skeleton_type: Registered skeleton key (e.g. ``"soma"``).
        config_path_or_data: Path to the JSON file, a pre-loaded dict, or ``None``
            to use the default bundled path for *skeleton_type*.

    Returns:
        A populated ``FootLandmarkModel``, or ``None`` if the config is absent or
        does not contain data for *skeleton_type*.
    """
    if isinstance(config_path_or_data, dict):
        library = config_path_or_data
        resolved_path = None
    else:
        resolved_path = (
            Path(config_path_or_data)
            if config_path_or_data is not None
            else get_default_foot_landmark_config_path(skeleton_type)
        )
        library = _load_landmark_library(resolved_path)

    rig_data = _extract_rig_data(library, skeleton_type)
    if rig_data is None:
        return None

    feet = []
    for foot_index, side_key in enumerate(_SIDE_KEYS):
        side_data = rig_data.get(side_key)
        if side_data is None:
            return None
        feet.append(_runtime_from_serialized(skeleton, foot_index, side_data))
    return FootLandmarkModel(
        skeleton_type=skeleton_type,
        feet=feet,
        config_path=str(resolved_path) if resolved_path else None,
    )


def save_authored_foot_landmark_model(
    model: FootLandmarkModel,
    config_path: str | Path | None = None,
) -> str:
    """Serialize *model* to a JSON config file, merging into any existing data.

    Args:
        model: Landmark model to save.
        config_path: Destination path; defaults to the bundled config for
            ``model.skeleton_type``.

    Returns:
        Absolute path of the written file.
    """
    config_path = (
        Path(config_path)
        if config_path is not None
        else get_default_foot_landmark_config_path(model.skeleton_type)
    )
    library = _load_landmark_library(config_path) or {}
    library["schema_version"] = _SCHEMA_VERSION
    rigs = library.get("rigs")
    if not isinstance(rigs, dict):
        rigs = {}
    rigs[model.skeleton_type] = {
        side_key: _runtime_to_serializable(model.feet[foot_index])
        for foot_index, side_key in enumerate(_SIDE_KEYS)
    }
    library["rigs"] = rigs
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as outfile:
        json.dump(library, outfile, indent=2)
        outfile.write("\n")
    model.config_path = str(config_path)
    return str(config_path)


def _build_default_foot_landmark_model(
    skeleton,
    skeleton_type: str,
    root_tx: wp.transform,
    config_path: str | Path | None = None,
) -> FootLandmarkModel:
    try:
        authored = load_authored_foot_landmark_model(skeleton, skeleton_type, config_path)
        if authored is not None:
            return authored
    except Exception:
        pass
    fallback = build_heuristic_foot_landmark_model(skeleton, skeleton_type, root_tx)
    fallback.config_path = str(config_path or get_default_foot_landmark_config_path(skeleton_type))
    return fallback


def _project_foot_landmarks_numpy(global_tx: np.ndarray, model: FootLandmarkModel) -> FootLandmarkData:
    rows = np.asarray(global_tx)
    squeeze = False
    if rows.ndim == 2:
        rows = rows[None, ...]
        squeeze = True
    if rows.ndim != 3 or rows.shape[-1] != 7:
        raise ValueError(f"Expected global transforms of shape (T, J, 7) or (J, 7), got {rows.shape}")

    num_frames = rows.shape[0]
    heel = np.zeros((num_frames, 2, 3), dtype=np.float32)
    toe_pivot = np.zeros((num_frames, 2, 3), dtype=np.float32)
    sole_normals = np.zeros((num_frames, 2, 3), dtype=np.float32)

    for foot in model.feet:
        foot_rows = rows[:, foot.foot_joint_index, :]
        toe_rows = rows[:, foot.toe_joint_index, :]
        normal_rows = foot_rows if foot.sole_normal_space == "foot" else toe_rows
        foot_idx = foot.foot_index

        for frame in range(num_frames):
            heel[frame, foot_idx] = _transform_point_np(foot_rows[frame], foot.heel_local_in_foot).astype(np.float32)
            toe_pivot[frame, foot_idx] = _transform_point_np(toe_rows[frame], foot.toe_pivot_local_in_toe).astype(np.float32)
            sole_normals[frame, foot_idx] = _normalize(
                _quat_rotate_np(normal_rows[frame, 3:7], foot.sole_normal_local)
            ).astype(np.float32)

    if squeeze:
        return FootLandmarkData(
            heel_positions=heel[0],
            toe_pivot_positions=toe_pivot[0],
            sole_normals=sole_normals[0],
        )

    return FootLandmarkData(
        heel_positions=heel,
        toe_pivot_positions=toe_pivot,
        sole_normals=sole_normals,
    )


def _project_foot_landmarks_warp_arrays(
    wp_global_tx,
    model: FootLandmarkModel,
) -> tuple[object, object]:
    if len(wp_global_tx.shape) != 2:
        raise ValueError(f"Expected Warp global transforms with shape (T, J), got {wp_global_tx.shape}")

    (
        foot_joint_indices,
        toe_joint_indices,
        normal_joint_indices,
        heel_local,
        toe_pivot_local,
        sole_normal_local,
    ) = _kernel_landmark_inputs(model)

    num_frames = wp_global_tx.shape[0]
    wp_landmark_positions = wp.empty(shape=(num_frames, 2, _LANDMARK_COUNT), dtype=wp.vec3)
    wp_sole_normals = wp.empty(shape=(num_frames, 2), dtype=wp.vec3)
    wp.launch(
        _project_foot_landmarks_kernel,
        dim=num_frames * 2,
        inputs=[
            wp.array(foot_joint_indices, dtype=wp.int32),
            wp.array(toe_joint_indices, dtype=wp.int32),
            wp.array(normal_joint_indices, dtype=wp.int32),
            wp_global_tx,
            wp.array(heel_local, dtype=wp.vec3),
            wp.array(toe_pivot_local, dtype=wp.vec3),
            wp.array(sole_normal_local, dtype=wp.vec3),
        ],
        outputs=[wp_landmark_positions, wp_sole_normals],
    )
    return wp_landmark_positions, wp_sole_normals


def _landmark_data_from_wp(
    wp_landmark_positions,
    wp_sole_normals,
) -> FootLandmarkData:
    positions = wp_landmark_positions.numpy()
    sole_normals = wp_sole_normals.numpy()
    return _unstack_landmark_positions(positions, sole_normals)


def _project_foot_landmarks_warp(global_tx: np.ndarray, model: FootLandmarkModel) -> FootLandmarkData:
    rows = np.asarray(global_tx, dtype=np.float32)
    squeeze = False
    if rows.ndim == 2:
        rows = rows[None, ...]
        squeeze = True
    if rows.ndim != 3 or rows.shape[-1] != 7:
        raise ValueError(f"Expected global transforms of shape (T, J, 7) or (J, 7), got {rows.shape}")

    wp_landmark_positions, wp_sole_normals = _project_foot_landmarks_warp_arrays(
        wp.array(rows, dtype=wp.transform, ndim=2),
        model,
    )
    landmarks = _landmark_data_from_wp(wp_landmark_positions, wp_sole_normals)
    if squeeze:
        return FootLandmarkData(
            heel_positions=landmarks.heel_positions[0],
            toe_pivot_positions=landmarks.toe_pivot_positions[0],
            sole_normals=landmarks.sole_normals[0],
        )
    return landmarks


def project_foot_landmarks(
    global_tx: np.ndarray,
    model: FootLandmarkModel,
    use_warp: bool = True,
) -> FootLandmarkData:
    """Project foot landmark points into world space for all frames.

    Args:
        global_tx: ``(T, J, 7)`` or ``(J, 7)`` float32 array of world-space
            transforms (position + quaternion) for every joint.
        model: Foot landmark model defining local-space anchor positions.
        use_warp: When ``True`` (default), runs a GPU kernel via Warp.
            Set to ``False`` for a slower NumPy fallback.

    Returns:
        ``FootLandmarkData`` with per-frame world-space positions for all landmark types.
    """
    if use_warp:
        return _project_foot_landmarks_warp(global_tx, model)
    return _project_foot_landmarks_numpy(global_tx, model)
