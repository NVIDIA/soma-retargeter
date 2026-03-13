# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import newton

from soma_retargeter.renderers.base_renderer import BaseRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.animation.mesh import SkeletalMesh


@wp.kernel
def skinning_kernel(
    points          : wp.array(dtype=wp.vec3),
    joint_indices   : wp.array(dtype=wp.int32),
    joint_weights   : wp.array(dtype=wp.float32),
    num_influences  : wp.int32,
    xform           : wp.array(dtype=wp.transform),
    output_points   : wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    output_points[i] = wp.vec3(0.0, 0.0, 0.0)
    for j in range(num_influences):
        output_points[i] += wp.transform_point(xform[joint_indices[i*num_influences + j]], points[i]) * joint_weights[i*num_influences + j]


@wp.kernel
def update_skinned_transform_kernel(
    num_joints          : wp.int32,
    local_transforms    : wp.array(dtype=wp.transform),
    parent_indices      : wp.array(dtype=wp.int32),
    bind_transforms     : wp.array(dtype=wp.transform),
    character_transform : wp.transform,
    skinned_transforms  : wp.array2d(dtype=wp.transform)):

    frame = wp.tid()
    skinned_transforms[frame, 0] = local_transforms[0]
    for joint_index in range(1, num_joints):
        skinned_transforms[frame, joint_index] = skinned_transforms[frame, parent_indices[joint_index]] * local_transforms[joint_index]

    for joint_index in range(num_joints):
        skinned_transforms[frame, joint_index] = character_transform*skinned_transforms[frame, joint_index]*wp.transform_inverse(bind_transforms[joint_index])


class SkeletalMeshRenderer(BaseRenderer):
    """Renders a SkeletalMesh with GPU-based linear blend skinning."""

    def __init__(self, skeletal_mesh: SkeletalMesh):
        super().__init__()
        self.skeletal_mesh = skeletal_mesh

        self.skinned_points = []
        for i in range(self.skeletal_mesh.num_skinned_meshes):
            self.skinned_points.append(wp.zeros(skeletal_mesh.skinned_meshes[i].num_points, dtype=wp.vec3))

        # 1 frame for now, but the kernel supports multiple frames
        self.skinned_transforms = wp.zeros((1, skeletal_mesh.skeleton.num_joints), dtype=wp.transform)

    @staticmethod
    def _set_color(viewer, object_name, color: wp.vec3):
        if isinstance(viewer, newton.viewer.ViewerGL):
            if object_name in viewer.objects:
                from pyglet import gl
                gl.glBindVertexArray(viewer.objects[object_name].vao)
                gl.glVertexAttrib3f(7, color[0], color[1], color[2])
                gl.glBindVertexArray(0)

    def draw(self, viewer, skeleton_instance: SkeletonInstance, color: wp.vec3, id: wp.int32):
        """Skin and display the mesh for the given skeleton pose."""
        if self.skeletal_mesh.skeleton != skeleton_instance.skeleton:
            raise ValueError("[ERROR]: SkeletalMeshRenderer.skeletal_mesh.skeleton is not equal to SkeletonInstance.skeleton")

        # Calculate delta transform between animation skin transforms and skeleton instance global transform
        animation_transforms = skeleton_instance.get_local_transforms()
        wp.launch(
            update_skinned_transform_kernel,
            dim=(1),
            inputs=[
                skeleton_instance.num_joints,
                wp.array(animation_transforms, dtype=wp.transform),
                wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
                self.skeletal_mesh.bind_transforms,
                skeleton_instance.xform],
            outputs=[self.skinned_transforms])

        skinned_meshes = self.skeletal_mesh.skinned_meshes
        for i in range(self.skeletal_mesh.num_skinned_meshes):
            dimension = skinned_meshes[i].num_points
            if dimension == 0:
                continue
            num_influences = skinned_meshes[i].num_influences
            wp.launch(
                skinning_kernel,
                dim=dimension,
                inputs=[
                    skinned_meshes[i].points,
                    skinned_meshes[i].joint_indices,
                    skinned_meshes[i].joint_weights,
                    int(num_influences),
                    wp.array(self.skinned_transforms[0], dtype=wp.transform)],
                outputs=[self.skinned_points[i]])

        for i in range(len(skinned_meshes)):
            name = f"/skeletal_mesh_{id}_{i}"
            self._register_unique_id(name)
            SkeletalMeshRenderer._set_color(viewer, name, color)
            viewer.log_mesh(name, self.skinned_points[i], wp.array(skinned_meshes[i].indices, dtype=wp.int32))

    def clear(self, viewer):
        """Remove all mesh objects from the viewer."""
        self._clear(viewer.objects)
