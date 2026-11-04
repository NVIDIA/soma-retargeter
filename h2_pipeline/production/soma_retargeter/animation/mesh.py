# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
import warp as wp

from soma_retargeter.animation.skeleton import Skeleton


class Mesh:
    """
    Base mesh consisting of vertex positions and triangle indices.

    Args:
        points: Vertex positions with shape ``(N, 3)``.
        indices: Triangle indices as a flat int array with shape ``(M,)`` where
            ``M`` is a multiple of 3.
    """

    def __init__(self, points : np.ndarray, indices : np.ndarray):
        self._points = wp.array(points, dtype=wp.vec3)
        self._indices = wp.array(indices, dtype=wp.int32)


class SkinnedMesh(Mesh):
    """
    A mesh with per-vertex skinning data for linear blend skinning.

    Args:
        points: Vertex positions with shape ``(N, 3)``.
        indices: Triangle indices as a flat int array.
        joint_indices: Flat array of joint indices, length ``N * K``.
        joint_weights: Flat array of skinning weights, length ``N * K``.
    """

    def __init__(self, points : np.ndarray, indices : np.ndarray, joint_indices : np.ndarray, joint_weights : np.ndarray):
        super().__init__(points, indices)
        self._joint_indices = wp.array(joint_indices, dtype=wp.int32)
        self._joint_weights = wp.array(joint_weights, dtype=wp.float32)
        if self._points.size > 0:
            self._num_influences = self.joint_indices.size/self.points.size
        else:
            self._num_influences = 0

    @property
    def num_influences(self):
        """Number of joint influences per vertex."""
        return self._num_influences

    @property
    def joint_indices(self):
        """Flat :class:`warp.array` of per-vertex joint indices (int32)."""
        return self._joint_indices

    @property
    def joint_weights(self):
        """Flat :class:`warp.array` of per-vertex skinning weights (float32)."""
        return self._joint_weights

    @property
    def points(self):
        """Vertex positions as a :class:`warp.array` of ``vec3``."""
        return self._points

    @property
    def indices(self):
        """Triangle indices as a :class:`warp.array` of ``int32``."""
        return self._indices

    @property
    def num_points(self):
        """Total number of vertices in the mesh."""
        return self._points.size

    @property
    def num_indices(self):
        """Total number of triangle indices."""
        return self._indices.size


class SkeletalMesh:
    """
    A complete skeletal mesh: one or more skinned meshes bound to a skeleton.

    The ``bind_transforms`` list provides the inverse bind-pose transform for
    every joint in the skeleton, used to bring vertices from bind-pose space
    into joint-local space during skinning.

    Args:
        skinned_meshes: List of :class:`SkinnedMesh` instances that make up the
            character (e.g. body, clothing, hair).
        skeleton: The :class:`~soma_retargeter.animation.skeleton.Skeleton`
            driving the deformation.
        bind_transforms: Per-joint bind-pose transforms. Must have the same
            length as ``skeleton.num_joints``.
        name: Optional human-readable name for the skeletal mesh.

    Raises:
        ValueError: If the length of *bind_transforms* does not equal
            ``skeleton.num_joints``.
    """

    def __init__(self, skinned_meshes : List[SkinnedMesh], skeleton: Skeleton, bind_transforms : List[wp.transform], name : str = "no_name"):
        self._skinned_meshes = skinned_meshes
        self._skeleton = skeleton
        self._name = name
        self._bind_transforms = wp.array(bind_transforms, dtype=wp.transform)
        if (len(bind_transforms) != skeleton.num_joints):
            raise ValueError("bind_transforms length must be equal to skeleton.num_joints")

    @property
    def num_skinned_meshes(self):
        """Number of skinned mesh parts in this skeletal mesh."""
        return len(self._skinned_meshes)

    @property
    def skinned_meshes(self):
        """List of :class:`SkinnedMesh` instances."""
        return self._skinned_meshes

    @property
    def skeleton(self):
        """The :class:`~soma_retargeter.animation.skeleton.Skeleton` bound to this mesh."""
        return self._skeleton

    @property
    def bind_transforms(self):
        """Per-joint bind-pose transforms as a :class:`warp.array` of ``transform``."""
        return self._bind_transforms

    @property
    def name(self):
        """Human-readable name of the skeletal mesh."""
        return self._name
