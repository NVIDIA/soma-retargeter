# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp
import numpy as np


class JointLimitClamper:
    """
    A utility class for clamping joint coordinates to their specified limits.
    """
    def __init__(self, model):
        self.n_dofs = model.joint_dof_count
        self.n_coords = model.joint_coord_count
        self.joint_limit_lower = model.joint_limit_lower
        self.joint_limit_upper = model.joint_limit_upper

        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]
            for k in range(lin + ang):
                dof_to_coord_np[dof0 + k] = coord0 + k
        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32)

    def apply(self, joint_q):
        """
        Clamps joint coordinates to their specified joint limits.
        Args:
            joint_q (wp.array2d): Joint configuration array of shape (n_batch, n_coords)
                containing the joint positions to be clamped. Each row represents a batch
                of joint configurations.
        Returns:
            wp.array2d: The same joint_q array with values clamped to joint limits.
                Shape is (n_batch, n_coords).
        Raises:
            ValueError: If joint_q.shape does not match the expected number of coordinates and the provided
                joint configuration dimensionality.
        """
        if joint_q.shape[1] != self.n_coords:
            raise ValueError(f"[ERROR]: joint_q size mismatch. Expected joint_q shape of [{joint_q.shape[0]}, {self.n_coords}] but received [{joint_q.shape}]")

        @wp.kernel
        def clamp_to_joint_limits_kernel(
            in_joint_limit_lower : wp.array1d(dtype=wp.float32),  # (n_dofs)
            in_joint_limit_upper : wp.array1d(dtype=wp.float32),  # (n_dofs)
            in_dof_to_coord      : wp.array1d(dtype=wp.int32),    # (n_dofs)
            inout_joint_q        : wp.array2d(dtype=wp.float32)   # (n_batch, n_coords)
        ):
            env, dof_idx = wp.tid()
            coord_idx = in_dof_to_coord[dof_idx]
            if coord_idx < 0:
                return

            inout_joint_q[env, coord_idx] = wp.clamp(
                inout_joint_q[env, coord_idx],
                in_joint_limit_lower[dof_idx],
                in_joint_limit_upper[dof_idx])

        wp.launch(
            clamp_to_joint_limits_kernel,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.dof_to_coord,
                joint_q])

        return joint_q
