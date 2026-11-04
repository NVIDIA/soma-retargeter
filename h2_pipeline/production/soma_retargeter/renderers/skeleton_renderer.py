# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import soma_retargeter.utils.pose_utils as pose_utils

from soma_retargeter.renderers.base_renderer import BaseRenderer
from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance

_const_pyramid_vertices = wp.array([
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(1.0, 1.0, 1.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(1.0, -1.0, 1.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(-1.0, -1.0, 1.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(-1.0, 1.0, 1.0),
    wp.vec3(1.0, 1.0, 1.0)],
    dtype=wp.vec3)

_const_pyramid_vertex_count = wp.constant(_const_pyramid_vertices.size)


@wp.func
def compute_pyramid_vertices(
    in_pyramid_vertices : wp.array(dtype=wp.vec3),
    in_tx               : wp.transform,
    in_scale            : wp.vec3,
    in_offset           : wp.int32,
    out_line_starts     : wp.array(dtype=wp.vec3),
    out_line_ends       : wp.array(dtype=wp.vec3)
):
    for i in range(8):
        idx = in_offset + i
        out_line_starts[idx] = wp.transform_point(in_tx, wp.cw_mul(in_pyramid_vertices[i * 2], in_scale))
        out_line_ends[idx] = wp.transform_point(in_tx, wp.cw_mul(in_pyramid_vertices[1 + i * 2], in_scale))


@wp.func
def compute_bone_lines(
    in_pyramid_vertices : wp.array(dtype=wp.vec3),
    in_offset           : wp.int32,
    in_joint_t          : wp.vec3,
    in_parent_t         : wp.vec3,
    out_line_starts     : wp.array(dtype=wp.vec3),
    out_line_ends       : wp.array(dtype=wp.vec3)
):
    diff = wp.sub(in_parent_t, in_joint_t)
    q = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.normalize(diff))

    length = wp.length(diff)
    width = length / 20.0

    compute_pyramid_vertices(
        in_pyramid_vertices, wp.transform(in_joint_t, q), wp.vec3(width, width, length * 0.8),
        in_offset, out_line_starts, out_line_ends)

    compute_pyramid_vertices(
        in_pyramid_vertices, wp.transform(in_parent_t, q), wp.vec3(width, width, -length * 0.2),
        in_offset + 8, out_line_starts, out_line_ends)


@wp.kernel
def _update_skeleton_lines_kernel(
    in_pyramid_vertices  : wp.array(dtype=wp.vec3),
    in_global_transforms : wp.array(dtype=wp.transform),
    in_bone_indices      : wp.array(dtype=wp.vec2i),
    out_line_starts      : wp.array(dtype=wp.vec3),
    out_line_ends        : wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    bone = in_bone_indices[tid]

    compute_bone_lines(
         in_pyramid_vertices,
         tid * _const_pyramid_vertex_count,
         in_global_transforms[bone.y].p,
         in_global_transforms[bone.x].p,
         out_line_starts,
         out_line_ends)


class SkeletonRenderer(BaseRenderer):
    """Renders a skeleton as pyramid-shaped bone lines."""

    def __init__(self, skeleton: Skeleton, masked_indices=None):
        super().__init__()
        self.skeleton = skeleton
        self.bones = self._build_bones(masked_indices)
        count = len(self.bones) * _const_pyramid_vertex_count
        self.line_starts = wp.zeros(count, dtype=wp.vec3)
        self.line_ends = wp.zeros(count, dtype=wp.vec3)
        self.parent_indices = wp.array(self.skeleton.parent_indices, dtype=wp.int32)

    def draw(self, viewer, skeleton_instance: SkeletonInstance, id: wp.int32):
        """Compute and display bone lines for the given skeleton pose."""
        if skeleton_instance.skeleton != self.skeleton:
            raise ValueError(f"[ERROR]: SkeletonInstance.skeleton [{skeleton_instance.skeleton}] is not equal to SkeletonRenderer.skeleton [{self.skeleton}]")

        global_transforms = pose_utils.compute_global_pose(self.skeleton, skeleton_instance.local_transforms, skeleton_instance.xform)
        wp.launch(
            _update_skeleton_lines_kernel,
            dim=len(self.bones),
            inputs=[
                _const_pyramid_vertices,
                wp.array(global_transforms, dtype=wp.transform),
                self.bones],
            outputs=[self.line_starts, self.line_ends])

        name = f"/skeleton_{id}"
        self._register_unique_id(name)
        viewer.log_lines(name, self.line_starts, self.line_ends, skeleton_instance.color)

    def clear(self, viewer):
        """Remove all skeleton lines from the viewer."""
        self._clear(viewer.lines)

    def _build_bones(self, mask_indices):
        mask = set(mask_indices) if mask_indices is not None else set()
        bones=[]
        for idx in range(1, self.skeleton.num_joints):
            parent_idx = self.skeleton.joint_parent(idx)
            if (idx in mask or parent_idx in mask):
                continue
            bones.append(wp.vec2i(parent_idx, idx))

        return wp.array(bones, dtype=wp.vec2i)
