# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType


@wp.func
def _wp_smooth_joint_filter_func(
    x            : wp.float32,
    lower_limit  : wp.float32,
    upper_limit  : wp.float32,
    offset_lower_limit : wp.float32,
    offset_upper_limit : wp.float32,
    m            : wp.float32,
    p            : wp.float32
):
    c = (lower_limit + upper_limit) * 0.5
    lower_limit += (offset_lower_limit - c)
    upper_limit -= (offset_upper_limit + c)
    if lower_limit < x and x <= upper_limit:
        return 0.0

    diff = wp.where(x <= lower_limit, lower_limit-x, x-upper_limit) * m
    return 1.0 - wp.exp(-wp.pow(diff, p))


@wp.kernel
def _smooth_joint_filter_residuals(
    joint_q: wp.array2d(dtype=wp.float32),            # (n_batch, n_coords)
    active_dof_indices: wp.array1d(dtype=wp.int32),   # (n_active_dofs)
    dof_to_coord: wp.array1d(dtype=wp.int32),         # (n_dofs)
    joint_limit_lower: wp.array1d(dtype=wp.float32),  # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32),  # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),        # (n_coords)
    offset_limit_lower: wp.array1d(dtype=wp.float32), # (n_coords)
    offset_limit_upper: wp.array1d(dtype=wp.float32), # (n_coords)
    weight: wp.array1d(dtype=wp.float32),             # (1)
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),         # (n_batch, n_residuals)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    c = (lower + upper) * 0.5

    lower_offset = offset_limit_lower[coord_idx]
    upper_offset = offset_limit_upper[coord_idx]

    q = joint_q[problem, coord_idx]
    error = (q - c)

    smoother = _wp_smooth_joint_filter_func(error, lower, upper, lower_offset, upper_offset, 1.0, 6.5)
    residuals[problem, start_idx + active_idx] = error * smoother * weight[0] * mask


@wp.kernel
def _update_weight(
    in_value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),  # (1)
):
    out_weight[0] = in_value


@wp.kernel
def _smooth_joint_filter_jac_analytic(
    active_dof_indices: wp.array1d(dtype=wp.int32), # (n_active_dofs)
    dof_to_coord: wp.array1d(dtype=wp.int32),       # (n_dofs)
    coord_masks: wp.array1d(dtype=wp.float32),      # (n_coords)
    start_idx: int,
    weight: wp.array1d(dtype=wp.float32),           # (1)
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),         # (n_batch, n_residuals, n_dofs)
):
    problem, active_idx = wp.tid()
    dof_idx = active_dof_indices[active_idx]
    coord_idx = dof_to_coord[dof_idx]
    mask = coord_masks[coord_idx]

    # Jacobian is diagonal: dr[dof]/dq[dof] = weight
    jacobian[problem, start_idx + active_idx, dof_idx] = weight[0] * mask


class IKSmoothJointFilter(ik.IKObjective):
    """
    Smooth joint filter objective eases in penalization when joint gets close to
    limits using a inverse gaussian filter. Joints can also be masked to an isolated set of DOFs.
    Otherwise, when DOF value is in limit range there is no penalization.

    Only allocates residual slots for active (non-zero masked) DOFs, reducing GPU shared
    memory usage in Newton's tiled solver.

    Args:
        joint_limit_lower: Lower bounds for each joint DoF.
        joint_limit_upper: Upper bounds for each joint DoF.
        weight: Scalar weight for the objective.
        coord_masks: Joint coord masks (n_coords,) to control which should be considered during the penalty computation.
                     As numpy array or wp.array. If all coords_masks are set to 0.0, the objective will be a NO-OP.
        offset_limit_lower: Optional lower offsets for each joint coord (n_coords,) to control the penalization range. Set to 1.0 by default.
        offset_limit_upper: Optional upper offsets for each joint coord (n_coords,) to control the penalization range. Set to 1.0 by default.

    Raises:
        ValueError: If coord_masks is not provided as a numpy array or wp.array.
    """

    def __init__(self, joint_limit_lower, joint_limit_upper, weight, coord_masks, offset_limit_lower=None, offset_limit_upper=None):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.n_dofs = len(joint_limit_lower)
        self.dof_to_coord = None
        self.e_array = None
        self._weight = wp.array([max(0.0, weight)], dtype=wp.float32)

        self.coord_masks = None
        self.coord_masks_np = None
        self._compact = False
        self._n_active_dofs = self.n_dofs

        if coord_masks is not None:
            if isinstance(coord_masks, np.ndarray):
                self.coord_masks_np = coord_masks.astype(np.float32)
                self._n_active_dofs = int(np.count_nonzero(coord_masks))
            else:
                self.coord_masks = coord_masks
                self._n_active_dofs = int(np.count_nonzero(coord_masks.numpy()))
            n_coords = len(coord_masks)
        else:
            raise ValueError("[ERROR]: coord_masks must be provided as a numpy array or wp.array.")

        if offset_limit_lower is None or n_coords != len(offset_limit_lower):
            self.offset_limit_lower = wp.ones(shape=n_coords, dtype=wp.float32)
        elif isinstance(offset_limit_lower, np.ndarray):
            self.offset_limit_lower = wp.array(offset_limit_lower, dtype=wp.float32)
        else:
            self.offset_limit_lower = offset_limit_lower

        if offset_limit_upper is None or n_coords != len(offset_limit_upper):
            self.offset_limit_upper = wp.ones(shape=n_coords, dtype=wp.float32)
        elif isinstance(offset_limit_upper, np.ndarray):
            self.offset_limit_upper = wp.array(offset_limit_upper, dtype=wp.float32)
        else:
            self.offset_limit_upper = offset_limit_upper

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()

        if self.coord_masks_np is not None:
            self.coord_masks = wp.array(self.coord_masks_np, dtype=wp.float32, device=self.device)

        if len(self.coord_masks) != model.joint_coord_count:
            raise ValueError("[ERROR]: coord_masks must have the same length as the number of joint coordinates.")

        # Build DOF to coordinate mapping
        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()

        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                if dof0 + k < self.n_dofs:
                    dof_to_coord_np[dof0 + k] = coord0 + k

        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

        # Build active DOF indices from coord masks
        masks_np = self.coord_masks.numpy()
        active = []
        for dof_idx in range(self.n_dofs):
            coord_idx = dof_to_coord_np[dof_idx]
            if coord_idx >= 0 and coord_idx < len(masks_np) and masks_np[coord_idx] > 0.0:
                active.append(dof_idx)
        self.active_dof_indices = wp.array(np.array(active, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._n_active_dofs = len(active)

        # For autodiff mode
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for i in range(self._n_active_dofs):
                    e[prob_idx, self.residual_offset + i] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self._n_active_dofs

    def set_weight(self, value):
        if self.coord_masks is None:
            return

        wp.launch(
            _update_weight,
            dim=1,
            inputs=[max(0.0, value)],
            outputs=[self._weight],
            device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_residuals,
            dim=[count, self._n_active_dofs],
            inputs=[
                joint_q,
                self.active_dof_indices,
                self.dof_to_coord,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coord_masks,
                self.offset_limit_lower,
                self.offset_limit_upper,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[self.n_batch, self._n_active_dofs],
            inputs=[
                self.active_dof_indices,
                self.dof_to_coord,
                self.coord_masks,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        count = joint_q.shape[0]
        wp.launch(
            _smooth_joint_filter_jac_analytic,
            dim=[count, self._n_active_dofs],
            inputs=[
                self.active_dof_indices,
                self.dof_to_coord,
                self.coord_masks,
                start_idx,
                self._weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
