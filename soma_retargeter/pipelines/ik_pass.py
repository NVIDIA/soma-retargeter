# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level IK pass chain for soma_retargeting_pipeline.

Each pass owns its CUDA graph and is driven by execute(frame). Build a list
of only the enabled passes and run the frame loop as:

    for pass_ in passes:
        pass_.execute(frame)

Typical pass order:
    BasePass → PostProcessPass
"""

import numpy as np
import warp as wp
import newton

from soma_retargeter.pipelines.ik_solver import BaseIKSolver

class BasePass:
    """Set frame targets, ramp the smooth filter, run the base IK solve.

    The GPU solve is captured into a CUDA graph once. Target setting and
    weight ramping run on CPU before each graph launch.
    """

    def __init__(
        self,
        ik_solver: BaseIKSolver,
        joint_q: wp.array,
        num_envs: int,
        input_targets: list,
        position_objectives: list,
        rotation_objectives: list,
        smooth_filter_objective,
        smooth_filter_weight: float,
        num_frames_to_remove: int,
        ik_iterations: int,
        state
    ):
        self._ik_solver = ik_solver
        self._joint_q = joint_q
        self._num_envs = num_envs

        def transpose_and_pad_transforms(input_targets: list[np.ndarray]):
            """
            input_targets: list of N np arrays, each shape (n_i, 7) float32 (wp.transform layout)
            returns:
                effector_size: int (max n_i across input_targets)
                positions: wp.array dtype=wp.vec3,  shape (max_size, effector_size, num_envs)
                rotations: wp.array dtype=wp.quat,  shape (max_size, effector_size, num_envs)
            """
            num_envs = len(input_targets)
            if num_envs == 0:
                raise ValueError("input_targets must be a non-empty list of np arrays.")

            max_size = max(len(t) for t in input_targets)
            effector_size = input_targets[0].shape[1]  # effector size

            positions = np.empty((max_size, effector_size, num_envs, 3), dtype=np.float32)
            rotations = np.empty((max_size, effector_size, num_envs, 4), dtype=np.float32)

            for env, targets in enumerate(input_targets):
                n = len(targets)
                positions[:n, :, env] = targets[:,:,0:3]
                rotations[:n, :, env] = targets[:,:,3:7]
                if n < max_size:
                    positions[n:, :, env] = targets[-1:,:,0:3]  # (1, 3) broadcasts to (max_size-n, 3)
                    rotations[n:, :, env] = targets[-1:,:,3:7]  # (1, 4) broadcasts to (max_size-n, 4)

            return (
                wp.array(positions, dtype=wp.vec3, ndim=2),
                wp.array(rotations, dtype=wp.vec4, ndim=2),
            )

        # convert the input_targets to separate position and rotation arrays, transposed to (max_size, effector_size, num_envs)
        self._input_positions, self._input_rotations = transpose_and_pad_transforms(input_targets)

        self._pos_objs = position_objectives
        self._rot_objs = rotation_objectives
        self._smooth_filter_obj = smooth_filter_objective
        self._smooth_filter_weight = smooth_filter_weight
        self._num_frames_to_remove = num_frames_to_remove
        self._ik_iterations = ik_iterations
        self._state = state

        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self._gpu_step()
            self._graph = cap.graph
        else:
            self._graph = None

    def _gpu_step(self):
        self._ik_solver.solve(self._state, self._ik_iterations)
        wp.copy(self._joint_q, self._ik_solver.joint_q)

    def execute(self, frame: int) -> None:
        """Set per-frame effector targets, ramp the smooth filter, and launch the IK solve.

        Args:
            frame: Current frame index into the input target buffer.
        """
        if frame <= self._num_frames_to_remove:
            w = self._smooth_filter_weight * (frame / float(self._num_frames_to_remove))
            self._smooth_filter_obj.set_weight(w)

        for i in range(len(self._pos_objs)):
            self._pos_objs[i].set_target_positions(self._input_positions[frame][i])
            self._rot_objs[i].set_target_rotations(self._input_rotations[frame][i])

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()


class PostProcessPass:
    """Limb stabiliser + foot-plant correction."""

    def __init__(
        self,
        joint_q: wp.array,
        num_envs: int,
        model,
        state,
        limb_stabilizer,
        input_targets: list,
        limb_effector_indices: list,
        env_limb_effector_tx: np.ndarray,
        contact_processing_enabled: bool,
        plant_blenders: list,
    ):
        self._joint_q = joint_q
        self._num_envs = num_envs
        self._model = model
        self._state = state
        self._limb_stabilizer = limb_stabilizer
        self._input_targets = input_targets
        self._limb_effector_indices = limb_effector_indices
        self._env_limb_effector_tx = env_limb_effector_tx
        self._contact_processing_enabled = contact_processing_enabled
        self._plant_blenders = plant_blenders

    def execute(self, frame: int) -> None:
        """Apply limb stabilizer and optional plant correction for a single frame.

        Reads effector targets for *frame*, blends in foot-plant corrections if
        contact processing is enabled, runs the two-bone IK solve, and updates
        both ``joint_q`` and ``model.joint_q``.

        Args:
            frame: Current frame index into the input target buffer.
        """
        self._limb_stabilizer.reset_state(self._joint_q)

        for env in range(self._num_envs):
            frame_idx = -1 if frame > len(self._input_targets[env]) - 1 else frame
            self._env_limb_effector_tx[env] = np.asarray(
                self._input_targets[env][frame_idx][self._limb_effector_indices]
            )

        if self._contact_processing_enabled and self._plant_blenders:
            for env in range(self._num_envs):
                if frame <= len(self._input_targets[env]) - 1:
                    self._env_limb_effector_tx[env][0:2] = (
                        self._plant_blenders[env].blend_targets(
                            frame, self._env_limb_effector_tx[env][0:2]
                        )
                    )

        self._limb_stabilizer.solve(self._env_limb_effector_tx)
        # Write into the existing buffer — do NOT reassign joint_q here;
        # Other instance could hold baked warp references to this array.
        wp.copy(self._joint_q, self._limb_stabilizer.current_state())
        wp.copy(self._model.joint_q, self._joint_q.flatten())
        newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state)

