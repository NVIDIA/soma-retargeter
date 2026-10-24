# SPDX-FileCopyrightText: Copyright (c) 2026 gohiroaki
# SPDX-License-Identifier: Apache-2.0
"""Optional per-frame continuity prior for explicitly selected scalar joints."""

import math

import newton
import newton.ik as ik
import numpy as np
import warp as wp


@wp.kernel
def _continuity_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    previous_q: wp.array2d(dtype=wp.float32),
    problem_idx: wp.array(dtype=wp.int32),
    coords: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    row, i = wp.tid()
    coord = coords[i]
    residuals[row, start_idx + i] = (
        joint_q[row, coord] - previous_q[problem_idx[row], coord]
    ) * weights[i] * scale[0]


@wp.kernel
def _continuity_jacobian(
    dofs: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    scale: wp.array(dtype=wp.float32),
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    row, i, dof = wp.tid()
    value = float(0.0)
    if dof == dofs[i]:
        value = weights[i] * scale[0]
    jacobian[row, start_idx + i, dof] = value


class IKJointContinuity(ik.IKObjective):
    """Penalize ``weight * (q_t - q_previous)`` during the base IK solve.

    The reference is frozen once per input frame, never once per LM iteration.
    Only named revolute/prismatic joints are supported; floating/quaternion
    coordinates must not be compared by scalar subtraction. Values are in
    radians/metres and weights multiply residuals (and are squared in the cost).
    This is a soft per-frame prior, not a velocity limit or an output filter.
    """

    def __init__(self, model, num_envs: int, joint_weights: dict):
        super().__init__()
        if not joint_weights:
            raise ValueError("joint_continuity_weights must select at least one joint")
        labels = [label.rsplit('/', 1)[-1] for label in model.joint_label]
        q_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        types = model.joint_type.numpy()
        coords, dofs, weights = [], [], []
        for name, value in joint_weights.items():
            weight = float(value)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"Continuity weight for {name!r} must be positive and finite")
            if labels.count(name) != 1:
                raise ValueError(f"Continuity joint {name!r} must match exactly one joint")
            joint = labels.index(name)
            if types[joint] not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                raise ValueError(f"Continuity joint {name!r} must be revolute or prismatic")
            coords.append(q_starts[joint])
            dofs.append(dof_starts[joint])
            weights.append(weight)
        self._coords = np.asarray(coords, dtype=np.int32)
        self._dofs = np.asarray(dofs, dtype=np.int32)
        self._weights = np.asarray(weights, dtype=np.float32)
        self._num_envs = num_envs
        self._num_coords = model.joint_coord_count
        self.previous_q = None

    def residual_dim(self):
        return len(self._coords)

    def supports_analytic(self):
        return True

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.coords = wp.array(self._coords, dtype=wp.int32, device=self.device)
        self.dofs = wp.array(self._dofs, dtype=wp.int32, device=self.device)
        self.weights = wp.array(self._weights, dtype=wp.float32, device=self.device)
        self.scale = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.previous_q = wp.zeros(
            (self._num_envs, self._num_coords), dtype=wp.float32, device=self.device)

    def update_reference(self, joint_q, enabled=True):
        """Copy the last base-pass solution without aliasing its mutable buffer."""
        if joint_q.shape != self.previous_q.shape:
            raise ValueError("Continuity reference must match the base-problem batch")
        wp.copy(self.previous_q, joint_q)
        self.scale.fill_(1.0 if enabled else 0.0)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        wp.launch(
            _continuity_residuals, dim=(joint_q.shape[0], self.residual_dim()),
            inputs=[joint_q, self.previous_q, problem_idx, self.coords, self.weights,
                    self.scale, start_idx], outputs=[residuals], device=self.device)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        # Clear the entire block as well as writing its nonzero entries.
        wp.launch(
            _continuity_jacobian,
            dim=(joint_q.shape[0], self.residual_dim(), model.joint_dof_count),
            inputs=[self.dofs, self.weights, self.scale, start_idx],
            outputs=[jacobian], device=self.device)

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        # This scalar-coordinate prior is linear in the selected DoF updates;
        # its exact derivative is the same for both solver modes.
        self.compute_jacobian_analytic(None, dq_dof, model, jacobian, None, start_idx)
