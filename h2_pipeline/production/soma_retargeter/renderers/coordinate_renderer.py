# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List

import warp as wp
import numpy as np

from soma_retargeter.renderers.base_renderer import BaseRenderer


@wp.kernel
def _compute_coordinate_lines_kernel(
    in_transforms   : wp.array(dtype=wp.transform),
    in_scale        : wp.float32,
    out_axes_starts : wp.array(dtype=wp.vec3),
    out_axes_ends   : wp.array(dtype=wp.vec3),
    out_axes_colors : wp.array(dtype=wp.vec3)
):
    in_idx = wp.tid()
    in_tx = in_transforms[in_idx]

    out_idx = in_idx * 3
    out_axes_starts[out_idx + 0] = in_tx.p
    out_axes_starts[out_idx + 1] = in_tx.p
    out_axes_starts[out_idx + 2] = in_tx.p

    out_axes_ends[out_idx + 0] = in_tx.p + wp.mul(wp.quat_rotate(in_tx.q, wp.vec3(1.0, 0.0, 0.0)), in_scale)
    out_axes_ends[out_idx + 1] = in_tx.p + wp.mul(wp.quat_rotate(in_tx.q, wp.vec3(0.0, 1.0, 0.0)), in_scale)
    out_axes_ends[out_idx + 2] = in_tx.p + wp.mul(wp.quat_rotate(in_tx.q, wp.vec3(0.0, 0.0, 1.0)), in_scale)

    out_axes_colors[out_idx + 0] = wp.vec3(1.0, 0.0, 0.0)
    out_axes_colors[out_idx + 1] = wp.vec3(0.0, 1.0, 0.0)
    out_axes_colors[out_idx + 2] = wp.vec3(0.0, 0.0, 1.0)


class CoordinateRenderer(BaseRenderer):
    """Draws RGB coordinate axes for a set of transforms."""

    def __init__(self):
        super().__init__()
        self.axes_starts = wp.zeros(32 * 3, dtype=wp.vec3)
        self.axes_ends = wp.zeros(32 * 3, dtype=wp.vec3)
        self.axes_colors = wp.zeros(32 * 3, dtype=wp.vec3)

    def draw(self, viewer, transforms: List[wp.transform], scale: wp.float32, id: wp.int32):
        """Compute and display axis lines for the given transforms."""
        dim = 1
        if isinstance(transforms, list) or isinstance(transforms, np.ndarray):
            dim = len(transforms)

        if dim > (self.axes_starts.size / 3):
            self.axes_starts = wp.zeros(dim * 3, dtype=wp.vec3)
            self.axes_ends = wp.zeros(dim * 3, dtype=wp.vec3)
            self.axes_colors = wp.zeros(dim * 3, dtype=wp.vec3)
        else:
            self.axes_starts.zero_()
            self.axes_ends.zero_()
            self.axes_colors.zero_()

        wp.launch(
            _compute_coordinate_lines_kernel,
            dim=dim,
            inputs=[
                wp.array(transforms, dtype=wp.transform),
                scale],
            outputs=[
                self.axes_starts,
                self.axes_ends,
                self.axes_colors])

        name = f"/coordinate_axes{id}"
        self._register_unique_id(name)
        viewer.log_lines(name, self.axes_starts, self.axes_ends, self.axes_colors)

    def clear(self, viewer):
        """Remove all coordinate axes from the viewer."""
        self._clear(viewer.lines)
