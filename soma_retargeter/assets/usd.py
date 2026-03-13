# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from pxr import Usd, UsdGeom, UsdSkel, Gf
from pxr import Vt
from soma_retargeter.animation.skeleton import Skeleton
from soma_retargeter.animation.mesh import SkeletalMesh, SkinnedMesh
from soma_retargeter.animation.animation_buffer import AnimationBuffer

# Constants
_EPSILON = 1e-6

###############################################################################
# USD Stage Metadata Helpers
###############################################################################


def _resolve_time_code(stage, time_code=None):
    """
    Resolve a time code for XformCache / attribute evaluation.

    Why this exists:
    - Many DCC exporters author timeSamples on xformOps but do NOT author a
      meaningful default value. Evaluating at Usd.TimeCode.Default() can then
      produce a different transform than evaluating at the first animation
      frame (stage start time), which is usually what you want for "bind pose"
      loading and for matching animation playback at t=0.
    """
    if time_code is None:
        if stage is None:
            return Usd.TimeCode.Default()

        start = stage.GetStartTimeCode()
        end = stage.GetEndTimeCode()

        # If the stage has a time range, prefer the start time code so we get
        # the authored animated xforms at the first frame (not the 'DEFAULT'
        # fallback, which can be different).
        if start < end:
            return Usd.TimeCode(start)

        return Usd.TimeCode.Default()

    if isinstance(time_code, Usd.TimeCode):
        return time_code

    # Accept ints/floats as time codes.
    return Usd.TimeCode(time_code)


def get_stage_meters_per_unit(stage):
    """
    Get the scale factor to convert USD units to meters.

    Args:
        stage: USD stage

    Returns:
        float: Meters per unit (e.g., 0.01 for centimeters, 1.0 for meters)
    """
    return UsdGeom.GetStageMetersPerUnit(stage)


def get_stage_up_axis(stage):
    """
    Get the up axis of the USD stage.

    Args:
        stage: USD stage

    Returns:
        str: 'Y' or 'Z'
    """
    up_axis = UsdGeom.GetStageUpAxis(stage)
    return 'Y' if up_axis == UsdGeom.Tokens.y else 'Z'


def get_up_axis_transform(stage, target_up='Z'):
    """
    Get transform to convert from USD up-axis to target up-axis.

    For Y-up to Z-up: rotate -90° around X axis.
    This maps: Y→Z (up), Z→-Y (forward), X→X (preserved)

    This is a pure rotation transform (no scale). Unit conversion should be
    handled separately via get_stage_meters_per_unit().

    Args:
        stage: USD stage
        target_up: Target up axis ('Y' or 'Z')

    Returns:
        wp.transform: Transform to convert coordinate systems (rotation only)
    """
    source_up = get_stage_up_axis(stage)

    if source_up == target_up:
        return wp.transform_identity()

    if source_up == 'Y' and target_up == 'Z':
        # Y-up to Z-up: rotate +90° around X axis
        # Maps: (0,1,0) → (0,0,1), (0,0,1) → (0,-1,0), (1,0,0) → (1,0,0)
        return wp.transform(
            wp.vec3(0, 0, 0),
            wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.radians(90.0))
        )
    elif source_up == 'Z' and target_up == 'Y':
        # Z-up to Y-up: rotate -90° around X axis (inverse of above)
        # Maps: (0,0,1) → (0,1,0), (0,1,0) → (0,0,-1), (1,0,0) → (1,0,0)
        return wp.transform(
            wp.vec3(0, 0, 0),
            wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.radians(-90.0))
        )

    return wp.transform_identity()


def get_import_correction(stage, target_up='Z', target_meters_per_unit=1.0):
    """
    Compute the corrective transform and scale for importing a USD stage.

    This provides the single correction to apply at the import root level,
    converting from source stage conventions to target app conventions.

    Args:
        stage: USD stage
        target_up: Target up axis ('Y' or 'Z'), default 'Z'
        target_meters_per_unit: Target linear units (1.0 = meters), default 1.0

    Returns:
        tuple: (rotation_transform: wp.transform, uniform_scale: float)
            - rotation_transform: Pure rotation for up-axis conversion
            - uniform_scale: Uniform scale factor for unit conversion

    Usage:
        rotation_tf, scale = get_import_correction(stage, 'Z', 1.0)
        # For points: final = scale * transform_point(rotation_tf, original)
        # For transforms: final = rotation_tf * original (scale translations separately)
    """
    rotation_tf = get_up_axis_transform(stage, target_up)

    src_meters = get_stage_meters_per_unit(stage)
    uniform_scale = src_meters / target_meters_per_unit

    return rotation_tf, uniform_scale


def get_prim_world_transform(stage, prim_path, scale=None, time_code=None):
    """
    Get a prim's world transform (position and rotation) using XformCache.

    This correctly handles all USD xformOps, resetXformStack, pivots, etc.

    Args:
        stage: USD stage
        prim_path: Path to the prim
        scale: Scale factor for translation. If None, uses stage's metersPerUnit.

    Returns:
        wp.transform: World transform (position scaled, rotation as quaternion)
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return wp.transform_identity()

    if scale is None:
        scale = get_stage_meters_per_unit(stage)
    else:
        scale = scale * get_stage_meters_per_unit(stage)

    tc = _resolve_time_code(stage, time_code)
    xform_cache = UsdGeom.XformCache(tc)
    world_mat = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)

    pos, rot, _ = decompose_matrix(world_mat, translation_scale=scale)

    return wp.transform(pos, rot)


###############################################################################
# Matrix Decomposition Helpers
###############################################################################

def decompose_matrix(mat, translation_scale=1.0):
    """
    Decompose a 4x4 transformation matrix into translation, rotation (quaternion), and scale.

    Args:
        mat: 4x4 numpy array transformation matrix
        translation_scale: Scale factor to apply to translation (e.g., 0.01 for cm to m)

    Returns:
        tuple: (position as wp.vec3, rotation as wp.quat, scale as np.array[3])
    """
    # Extract translation (row 3 for row-major USD matrices)
    pos = mat[3, :3] * translation_scale

    # Extract 3x3 rotation/scale matrix
    rot_matrix = mat[:3, :3].copy()

    # Extract scale (length of each column)
    scale = np.array([
        np.linalg.norm(rot_matrix[:, 0]),
        np.linalg.norm(rot_matrix[:, 1]),
        np.linalg.norm(rot_matrix[:, 2])
    ])

    # Normalize rotation matrix columns to remove scale
    for i in range(3):
        if scale[i] > _EPSILON:
            rot_matrix[:, i] /= scale[i]

    # Convert to quaternion (transpose for warp's expected format)
    rot = wp.quat_from_matrix(wp.mat33(rot_matrix.T.flatten()))

    return wp.vec3(pos[0], pos[1], pos[2]), rot, scale


def get_prim_world_scale(prim, time_code=None):
    """
    Get the world-space scale of a USD prim by decomposing its world transform matrix.

    Args:
        prim: USD prim
        time_code: Optional time code (defaults to Usd.TimeCode.Default())

    Returns:
        np.array: Scale factors [sx, sy, sz]
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    xform_cache = UsdGeom.XformCache(time_code)
    mat = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)

    # Extract scale from the 3x3 rotation/scale matrix (length of each column)
    scale = np.array([
        np.linalg.norm(mat[:3, 0]),
        np.linalg.norm(mat[:3, 1]),
        np.linalg.norm(mat[:3, 2])
    ])

    return scale


def get_prim_meters_scale(stage, prim, time_code=None):
    """
    Get the combined scale factor to convert prim data to meters.

    Combines:
    1. Stage metersPerUnit (converts USD units to meters)
    2. Prim's world transform scale (artistic/geometric scale)

    Args:
        stage: USD stage
        prim: USD prim
        time_code: Optional time code

    Returns:
        float: Uniform scale factor (assumes uniform scale, uses X component)
    """
    tc = _resolve_time_code(stage, time_code)
    meters_per_unit = get_stage_meters_per_unit(stage)
    prim_scale = get_prim_world_scale(prim, tc)

    # Combined scale: USD units to meters * prim artistic scale
    # Assume uniform scale, use X component
    combined_scale = meters_per_unit * prim_scale[0]

    return combined_scale


###############################################################################
# Animation Detection Helpers
###############################################################################

def has_animated_xform(prim):
    """
    Check if a prim or any of its ancestors has animated xform (time samples).

    Args:
        prim: USD prim to check

    Returns:
        bool: True if prim or ancestors have xform animation
    """
    check_prim = prim
    while check_prim and str(check_prim.GetPath()) != "/":
        xformable = UsdGeom.Xformable(check_prim)
        if xformable:
            xform_time_samples = xformable.GetTimeSamples()
            if len(xform_time_samples) > 1:
                return True
        check_prim = check_prim.GetParent()
    return False


###############################################################################
# Mesh Processing Helpers
###############################################################################

def triangulate_mesh(indices, counts):
    """
    Triangulate a mesh from face vertex indices and face vertex counts.

    Args:
        indices: numpy array of face vertex indices
        counts: iterable of vertex counts per face

    Returns:
        numpy array of triangulated indices, or None if no valid faces
    """
    faces = []
    face_id = 0

    for count in counts:
        if count == 3:
            # Triangle - use as-is
            faces.append(indices[face_id : face_id + 3])
        elif count == 4:
            # Quad - split into two triangles
            faces.append(indices[face_id : face_id + 3])
            faces.append(indices[[face_id, face_id + 2, face_id + 3]])
        else:
            # Skip n-gons (polygons with > 4 vertices)
            pass
        face_id += count

    if len(faces) == 0:
        return None

    return np.array(faces, dtype=np.int32).flatten()


###############################################################################
# USD Export Helpers
###############################################################################

def _to_float3(vec):
    """Return a tuple of (x, y, z) from a wp.vec3 or compatible array."""
    if hasattr(vec, "x"):
        return float(vec.x), float(vec.y), float(vec.z)
    arr = np.array(vec, dtype=np.float64).flatten()
    return float(arr[0]), float(arr[1]), float(arr[2])


def _to_quat_xyzw(quat):
    """Return (x, y, z, w) from a wp.quat or compatible array."""
    if hasattr(quat, "x"):
        return float(quat.x), float(quat.y), float(quat.z), float(quat.w)
    arr = np.array(quat, dtype=np.float64).flatten()
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


def _build_joint_paths(joint_names, parent_indices):
    """
    Build hierarchical UsdSkel joint tokens (slash-separated) from flat names and parent indices.
    """
    joint_paths = []
    for i, name in enumerate(joint_names):
        path_elems = [name]
        p = parent_indices[i]
        while p != -1:
            path_elems.append(joint_names[p])
            p = parent_indices[p]
        joint_paths.append("/".join(reversed(path_elems)))
    return joint_paths


def _wp_transform_to_gf_matrix(tf):
    tx, ty, tz = _to_float3(tf.p)
    qx, qy, qz, qw = _to_quat_xyzw(tf.q)
    m = Gf.Matrix4d(1.0)
    m.SetRotate(Gf.Quatf(qw, qx, qy, qz))
    m.SetTranslateOnly(Gf.Vec3d(tx, ty, tz))
    return m


def save_skeleton_and_animation_to_usd(
    path,
    skeleton: Skeleton,
    anim: AnimationBuffer,
    meters_per_unit: float = 1.0,
    up_axis: str = "Z",
    skel_root_path: str = "/SkelRoot",
    skel_path: str = "/SkelRoot/Skeleton",
    anim_path: str = "/SkelRoot/Anim",
):
    """
    Convenience helper: create a new USD at `path` with a SkelRoot, Skeleton,
    and UsdSkel.Animation sourced from the provided skeleton and AnimationBuffer.
    """
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis.upper() == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    stage.SetFramesPerSecond(anim.sample_rate)
    stage.SetTimeCodesPerSecond(anim.sample_rate)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, anim.num_frames - 1))

    skel_root = UsdSkel.Root.Define(stage, skel_root_path)
    skel = UsdSkel.Skeleton.Define(stage, skel_path)
    skel.CreateJointsAttr(_build_joint_paths(skeleton.joint_names, skeleton.parent_indices))
    bind_mats = [_wp_transform_to_gf_matrix(tf) for tf in skeleton.reference_local_transforms]
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray(bind_mats))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray(bind_mats))
    skel_root.GetPrim().CreateRelationship("skeleton", False).SetTargets([skel.GetPath()])

    anim_prim = UsdSkel.Animation.Define(stage, anim_path)
    translations_attr = anim_prim.CreateTranslationsAttr()
    rotations_attr = anim_prim.CreateRotationsAttr()
    scales_attr = anim_prim.CreateScalesAttr()
    unit_scales = Vt.Vec3fArray([Gf.Vec3f(1.0, 1.0, 1.0)] * skeleton.num_joints)
    scales_attr.Set(unit_scales, time=0)

    for frame in range(anim.num_frames):
        locals_frame = anim.local_transforms[frame]
        translations = []
        rotations = []
        for j in range(skeleton.num_joints):
            tf = locals_frame[j]
            tx, ty, tz = _to_float3(tf[0:3])
            qx, qy, qz, qw = _to_quat_xyzw(tf[3:7])
            translations.append(Gf.Vec3f(tx, ty, tz))
            rotations.append(Gf.Quatf(qw, qx, qy, qz))
        translations_attr.Set(Vt.Vec3fArray(translations), time=frame)
        rotations_attr.Set(Vt.QuatfArray(rotations), time=frame)

    skel.GetPrim().CreateRelationship("animationSource", False).SetTargets([anim_prim.GetPath()])
    stage.Save()
    return stage


###############################################################################
# USD Discovery and Loading Functions
###############################################################################

def discover_usd_skel(stage):
    """
    Traverse a USD stage to discover skeleton, animation, skinned mesh prims, and stage metadata.

    Args:
        stage: USD stage to traverse

    Returns:
        dict: Dictionary containing:
            - 'skeletons': list of skeleton prim paths
            - 'animations': list of animation prim paths
            - 'skinned_meshes': list of skinned mesh prim paths
            - 'skeleton': best guess skeleton prim path (first found)
            - 'animation': best guess animation prim path (first found)
            - 'skinned_mesh': best guess skinned mesh prim path (first found)
            - 'meters_per_unit': scale factor to convert USD units to meters
            - 'up_axis': 'Y' or 'Z'
            - 'fps': frames per second
            - 'start_time': start time code
            - 'end_time': end time code
    """
    result = {
        'skeletons': [],
        'animations': [],
        'skinned_meshes': [],
        'skeleton': None,
        'animation': None,
        'skinned_mesh': None,
        # Stage metadata
        'meters_per_unit': get_stage_meters_per_unit(stage),
        'up_axis': get_stage_up_axis(stage),
        'fps': stage.GetFramesPerSecond(),
        'start_time': stage.GetStartTimeCode(),
        'end_time': stage.GetEndTimeCode(),
    }

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        # Check for UsdSkel.Skeleton
        if prim.IsA(UsdSkel.Skeleton):
            result['skeletons'].append(prim_path)
            if result['skeleton'] is None:
                result['skeleton'] = prim_path

        # Check for UsdSkel.Animation
        if prim.IsA(UsdSkel.Animation):
            result['animations'].append(prim_path)
            if result['animation'] is None:
                result['animation'] = prim_path

        # Check for skinned mesh (UsdGeom.Mesh with skinning attributes)
        if prim.IsA(UsdGeom.Mesh):
            binding_api = UsdSkel.BindingAPI(prim)
            if binding_api:
                joint_indices_attr = binding_api.GetJointIndicesAttr()
                joint_weights_attr = binding_api.GetJointWeightsAttr()
                if joint_indices_attr and joint_weights_attr:
                    joint_indices = joint_indices_attr.Get()
                    joint_weights = joint_weights_attr.Get()
                    if joint_indices is not None and joint_weights is not None:
                        result['skinned_meshes'].append(prim_path)
                        if result['skinned_mesh'] is None:
                            result['skinned_mesh'] = prim_path

    print(f"[INFO]: USD discovery found {len(result['skeletons'])} skeleton(s), "
          f"{len(result['animations'])} animation(s), {len(result['skinned_meshes'])} skinned mesh(es)")
    print(f"[INFO]: Stage metadata - units: {result['meters_per_unit']} m/unit, "
          f"up axis: {result['up_axis']}, fps: {result['fps']}")

    return result


def parse_xform(prim):
    xform = UsdGeom.Xform(prim)
    mat = np.array(xform.GetLocalTransformation(), dtype=np.float32)
    rot = wp.quat_from_matrix(wp.mat33(mat[:3, :3].T.flatten()))
    pos = mat[3, :3]
    return wp.transform(pos, rot)


def extract_bind_transforms_from_usd(stage, skeleton_path="/root/Hips"):
    """
    Extract bind transforms from USD skeletal data.

    Args:
        stage: USD stage containing skeletal data
        skeleton_path: Path to the skeleton prim in the USD stage

    Returns:
        tuple: (joint_names, bind_transforms) where:
            - joint_names: List of joint names
            - bind_transforms: List of 4x4 transformation matrices as numpy arrays
    """
    try:
        # Get the skeleton prim
        skeleton_prim = stage.GetPrimAtPath(skeleton_path)
        if not skeleton_prim.IsValid():
            print(f"Warning: Skeleton prim not found at path {skeleton_path}")
            return [], []

        # Create UsdSkel.Skeleton object
        skeleton = UsdSkel.Skeleton(skeleton_prim)
        if not skeleton:
            print(f"Warning: Could not create UsdSkel.Skeleton from prim at {skeleton_path}")
            return [], []

        # Get joint names
        joints_attr = skeleton.GetJointsAttr()
        if not joints_attr:
            print("Warning: No joints attribute found in skeleton")
            return [], []
        joint_names = joints_attr.Get()

        # Get bind transforms
        bind_transforms_attr = skeleton.GetBindTransformsAttr()
        if not bind_transforms_attr:
            print("Warning: No bindTransforms attribute found in skeleton")
            return list(joint_names), []

        bind_transforms_raw = bind_transforms_attr.Get()

        # Convert to numpy arrays
        bind_transforms = []
        for transform_matrix in bind_transforms_raw:
            # Convert USD matrix to numpy array
            mat = np.array(transform_matrix, dtype=np.float32).reshape(4, 4)
            bind_transforms.append(mat)

        print(f"Successfully extracted {len(bind_transforms)} bind transforms for {len(joint_names)} joints")
        return list(joint_names), bind_transforms

    except Exception as e:
        print(f"Error extracting bind transforms: {e}")
        return [], []


def get_bind_transform_as_warp_transform(bind_matrix, scale=1.0):
    """
    Convert a 4x4 bind transform matrix to a warp transform.

    Args:
        bind_matrix: 4x4 numpy array representing the bind transform
        scale: Scale factor to apply to translation

    Returns:
        wp.transform: Warp transform object
    """
    # Extract rotation (3x3 upper-left) and translation (last column, first 3 elements)
    rot_matrix = bind_matrix[:3, :3]
    translation = bind_matrix[3, :3] * scale

    # Convert rotation matrix to quaternion
    rot_quat = wp.quat_from_matrix(wp.mat33(rot_matrix.T.flatten()))

    return wp.transform(translation, rot_quat)

#return points, indices, joint_indices, joint_weights


def load_skinning_data_from_usd_prim(prim, incoming_xform=None, scale=1.0, skeleton_joints=None, stage=None, skeleton_world_transform=None, time_code=None):
    """
    Load skinning data from a USD mesh prim.

    Uses XformCache.GetLocalToWorldTransform() for accurate world transforms,
    consistent with how rigid bodies are loaded.

    Args:
        prim: USD mesh prim
        incoming_xform: Accumulated parent transform (fallback if stage not provided)
        scale: Scale factor for points
        skeleton_joints: Fallback joint list if mesh doesn't have joints attribute
        stage: USD stage (for XformCache access)
        skeleton_world_transform: Skeleton's world transform (pos, rot) to compute geomBindTransform
    """
    # Get mesh's world transform using XformCache (same as rigid bodies)
    # This correctly handles all xformOps, resets, pivots, etc.
    mesh_world_pos = wp.vec3(0, 0, 0)
    mesh_world_rot = wp.quat_identity()
    # Accumulated/world transform for this prim, used to traverse children correctly
    prim_world_xform = None
    if stage is not None:
        tc = _resolve_time_code(stage, time_code)
        xform_cache = UsdGeom.XformCache(tc)
        world_mat = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)
        # Decompose to get position and rotation (scale applied separately to points)
        mesh_world_pos, mesh_world_rot, _ = decompose_matrix(world_mat, translation_scale=scale)
        prim_world_xform = wp.transform(mesh_world_pos, mesh_world_rot)
    else:
        # Fallback to manual accumulation if stage not provided
        xform = parse_xform(prim)
        xform.p *= scale
        if incoming_xform is not None:
            xform = incoming_xform * xform
        mesh_world_pos = xform.p
        mesh_world_rot = xform.q
        prim_world_xform = xform

    # Compute geomBindTransform: transforms mesh points from mesh-local space to skeleton space
    # geom_bind = inv(skeleton_world) * mesh_world
    # This ensures mesh points and bind transforms are in the same coordinate system
    if skeleton_world_transform is not None:
        skel_world_pos = skeleton_world_transform.p
        skel_world_rot = skeleton_world_transform.q

        # Compute relative transform: inv(skel_world) * mesh_world
        inv_skel_rot = wp.quat_inverse(skel_world_rot)
        geom_bind_rot = wp.mul(inv_skel_rot, mesh_world_rot)
        # Position: rotate mesh position into skeleton space, then subtract skeleton position
        geom_bind_pos = wp.quat_rotate(inv_skel_rot, mesh_world_pos - skel_world_pos)
    else:
        geom_bind_rot = mesh_world_rot
        geom_bind_pos = mesh_world_pos

    geom_bind_xform = wp.transform(geom_bind_pos, geom_bind_rot)

    skinning_data = []

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        # Create binding API
        binding_api = UsdSkel.BindingAPI(prim)
        if not binding_api:
            print(f"No skinning binding found on mesh: {prim.GetPath()}")
            # Continue traversing children even if this mesh is not skinned
        else:
            # Load and scale points
            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32) * scale

            # Check if USD has an authored geomBindTransform (preferred over computed)
            # This is the official USD way to specify mesh-to-skeleton space transform
            geom_bind_attr = binding_api.GetGeomBindTransformAttr()
            tc = _resolve_time_code(stage, time_code) if stage is not None else Usd.TimeCode.Default()
            geom_bind_val = geom_bind_attr.Get(tc) if geom_bind_attr else None
            if geom_bind_attr and geom_bind_val is not None:
                geom_bind_mat = np.array(geom_bind_val, dtype=np.float32).reshape(4, 4)
                # Decompose the matrix - USD stores row-major, translation in row 3
                gb_pos, gb_rot, _ = decompose_matrix(geom_bind_mat, translation_scale=scale)
                geom_bind_xform = wp.transform(gb_pos, gb_rot)
                print(f"[INFO]: Using authored geomBindTransform from USD for {prim.GetPath()}")
            # else: use computed geom_bind_xform from above

            # Apply geomBindTransform to align mesh with skeleton coordinate system
            for i in range(len(points)):
                points[i] = wp.transform_point(geom_bind_xform, wp.vec3(points[i]))
            indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            counts = mesh.GetFaceVertexCountsAttr().Get()
            faces = []
            face_id = 0
            for count in counts:
                if count == 3:
                    faces.append(indices[face_id : face_id + 3])
                elif count == 4:
                    faces.append(indices[face_id : face_id + 3])
                    faces.append(indices[[face_id, face_id + 2, face_id + 3]])
                else:
                    pass # Skip
                face_id += count

            face_indices = np.array(faces, dtype=np.int32).flatten()

            # Get joint indices (which joints affect each vertex)
            joint_attr = binding_api.GetJointsAttr()
            joint_indices_attr = binding_api.GetJointIndicesAttr()
            joint_weights_attr = binding_api.GetJointWeightsAttr()

            valid_skinning = True
            if joint_indices_attr is None or joint_weights_attr is None:
                valid_skinning = False

            joints = None
            if valid_skinning:
                joints = joint_attr.Get() if joint_attr else None

                # Check if joints is valid, if not try to use skeleton joints as fallback
                if joints is None:
                    if skeleton_joints is not None:
                        print(f"[INFO]: Mesh {prim.GetPath()} missing 'joints' relationship, using skeleton joints as fallback.")
                        joints = skeleton_joints
                    else:
                        print(f"[WARNING]: Mesh {prim.GetPath()} has skinning attributes but 'joints' relationship is missing or empty.")
                        valid_skinning = False

            if valid_skinning:
                joint_indices = np.array(joint_indices_attr.Get(), dtype=np.int32)
                joint_weights = np.array(joint_weights_attr.Get(), dtype=np.float32)

                if joint_indices is None or joint_weights is None:
                    valid_skinning = False

            if valid_skinning:
                #remap joint indices to the joint indices in the skeleton
                skinning_data.append(SkinningData(points, face_indices, joints, joint_indices, joint_weights, geom_bind_xform))

    for child in prim.GetChildren():
        success, sd = load_skinning_data_from_usd_prim(
            child,
            # IMPORTANT: pass the parent's accumulated/world transform, NOT the mesh->skeleton geomBindTransform
            incoming_xform=prim_world_xform,
            scale=scale,
            skeleton_joints=skeleton_joints,
            stage=stage,
            skeleton_world_transform=skeleton_world_transform,
            time_code=time_code,
        )
        if success:
            for i in range(len(sd)):
                skinning_data.append(sd[i])

    return True, skinning_data

# this is used tof skinning data from usd
# we use this class to hold USD information until we remap the joint indices to the joint indices in the skeleton.


class SkinningData:
    def __init__(self, points, indices, joints, joint_indices, joint_weights, xform):
        self.points = points
        self.indices = indices
        self.usd_joints = []
        if joints is not None:
            for i in range(len(joints)):
                self.usd_joints.append(str(joints[i]).split("/")[-1])
        self.usd_joint_indices = joint_indices
        self.joint_weights = joint_weights
        self.xform = xform
        self.skinned_points = points

    def remap_joint_indices(self, skeleton):
        self.joint_indices = np.zeros(len(self.usd_joint_indices), dtype=int)
        for i in range(len(self.usd_joint_indices)):
            joint_name = self.usd_joints[self.usd_joint_indices[i]]
            ji = skeleton.joint_index(joint_name)
            self.joint_indices[i] = ji


# input : give usd stage, and skeleton, and skeleton path in usd
# @TODO: this USD functionality doesn't create skeleton from USD. Maybe we should add that functionality.
# output : skeletal mesh
# We need skeleton to remap the joint indices in the skinning data to the joint indices in the skeleton.
def _load_skeletal_mesh(stage : Usd.Stage, skeleton : Skeleton, mesh_prim_path : str, skeleton_prim_path : str, name : str = "", scale : float = None, time_code=None):
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        print(f"Error: Mesh prim not found at path {mesh_prim_path}")
        return None

    # NOTE: `scale` is an optional user override. If provided, it is used for both
    # mesh and skeleton data. If not provided, compute each prim's scale independently.
    #
    # This matters because mesh prim scale and skeleton prim scale can differ; reusing
    # the mesh scale for the skeleton prim incorrectly scales skeleton world translations.
    mesh_scale = get_prim_meters_scale(stage, mesh_prim, time_code=time_code)
    if scale is not None:
        mesh_scale = scale * mesh_scale
    print(f"[INFO]: Auto-detected mesh scale: {mesh_scale} (metersPerUnit * primScale)")

    # Get skeleton joints to use as fallback if mesh doesn't have joints relationship
    skeleton_joints = None
    skel_prim = None
    skeleton_scale = mesh_scale
    if skeleton_prim_path:
        skel_prim = stage.GetPrimAtPath(skeleton_prim_path)
        if skel_prim.IsValid():
            # Compute skeleton scale independently (only when user didn't override `scale`)
            if scale is None:
                skeleton_scale = get_prim_meters_scale(stage, skel_prim, time_code=time_code)
            usd_skeleton = UsdSkel.Skeleton(skel_prim)
            if usd_skeleton:
                joints_attr = usd_skeleton.GetJointsAttr()
                if joints_attr:
                    skeleton_joints = joints_attr.Get()

    # Get skeleton's world transform to compute geomBindTransform
    skeleton_world_transform = wp.transform_identity()
    if skel_prim is not None and skel_prim.IsValid():
        tc = _resolve_time_code(stage, time_code)
        xform_cache = UsdGeom.XformCache(tc)
        skel_world_mat = np.array(xform_cache.GetLocalToWorldTransform(skel_prim), dtype=np.float32)
        # Use the skeleton prim's own scale when scaling its world-space translation.
        skel_pos, skel_rot, _ = decompose_matrix(skel_world_mat, translation_scale=skeleton_scale)
        skeleton_world_transform = wp.transform(skel_pos, skel_rot)

    success, skinning_data = load_skinning_data_from_usd_prim(
        mesh_prim,
        scale=mesh_scale,
        skeleton_joints=skeleton_joints,
        stage=stage,
        skeleton_world_transform=skeleton_world_transform,
        time_code=time_code,
    )
    if not success:
        print(f"Error loading skeletal mesh from USD: {mesh_prim_path}")
        return None
    if len(skinning_data) == 0:
        print(f"Warning: No skinning data found in mesh prim: {mesh_prim_path}")
        return None

    skinned_meshes = []
    for i in range(len(skinning_data)):
        skinning_data[i].remap_joint_indices(skeleton)

        skinned_mesh = SkinnedMesh(skinning_data[i].points, skinning_data[i].indices, skinning_data[i].joint_indices, skinning_data[i].joint_weights)
        skinned_meshes.append(skinned_mesh)

    if skeleton_prim_path is not None:
        #I need skeleton before going for this, so do this after _initialize_animation
        # Extract bind transforms from USD skeletal data
        # Bind transforms stay in skeleton space (mesh points are transformed to skeleton space via geomBindTransform)
        mesh_bind_transforms = [wp.transform_identity() for _ in range(len(skeleton.joint_names))]

        usd_joint_names = []
        usd_to_skel_joint_mapping = {}
        if skeleton_prim_path is not None:
            joint_names, bind_transforms = extract_bind_transforms_from_usd(stage, skeleton_prim_path)
            if joint_names and bind_transforms:
                for i in range(len(joint_names)):
                    joint_name = joint_names[i].split("/")[-1]
                    usd_joint_names.append(joint_name)
                    joint_index = skeleton.joint_index(joint_name)
                    usd_to_skel_joint_mapping[i] = joint_index

                for i in range(skeleton.num_joints):
                    # get only last part of the joint name
                    # if we have it
                    if skeleton.joint_names[i] in usd_joint_names:
                        usd_joint_idx = usd_joint_names.index(skeleton.joint_names[i])
                        # Bind transforms are authored in skeleton space; scale them using the
                        # skeleton prim's scale (not the mesh prim's scale).
                        mesh_bind_transforms[i] = get_bind_transform_as_warp_transform(bind_transforms[usd_joint_idx], scale=skeleton_scale)
                    else:
                        parent_index = skeleton.parent_indices[i]
                        if parent_index != -1:
                            # when you don't find bind transform for your joint, use the bind transform of your parent
                            mesh_bind_transforms[i] = mesh_bind_transforms[parent_index]
                        else:
                            mesh_bind_transforms[i] = wp.transform_identity()

                #print(f"mesh_bind_transforms: {mesh_bind_transforms}")
                print(f"Loaded bind transforms from skeleton at: {skeleton_prim_path}")

        return SkeletalMesh(skinned_meshes, skeleton, mesh_bind_transforms, name)

    return None


def load_skeletal_mesh_from_usd(usd_file_path, skeleton: Skeleton, mesh_prim_path : str, skeleton_prim_path: str, name : str = "", scale : float = 1.0):
    stage = Usd.Stage.Open(usd_file_path)
    skeletalmesh = _load_skeletal_mesh(stage, skeleton, mesh_prim_path, skeleton_prim_path, name, scale)

    if skeletalmesh is None:
        print(f"Error: Could not create SkeletalMesh from USD: {usd_file_path}")

    return skeletalmesh
