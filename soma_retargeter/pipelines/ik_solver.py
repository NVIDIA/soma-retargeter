# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IK pass abstractions for multi-pass retargeting pipelines.

Classes
-------
IKSolver
    Base: owns objectives, solver, and joint_q buffer. Factory static methods
    for each objective type.
BaseIKSolver(IKSolver)
    Kinematic-only solver (position + rotation + optional constraint objectives).

Typical 3-pass sequence::

    base_pass = BaseIKSolver(ik_model, num_envs, model,
        IKSolver.make_position_objectives(...) +
        IKSolver.make_rotation_objectives(...) +
        [IKSolver.make_smooth_joint_filter(ik_model)])

    # Per-frame:
    base_pass.solve(state, iterations)

Low-level solvers only; for the full pass chain with CUDA graphs and
execute(frame) see ``ik_pass_chain.py``.
"""

from typing import List

import numpy as np
import warp as wp
import newton
import newton.ik as ik

from soma_retargeter.pipelines.ik_objectives import (
    IKSmoothJointFilter
)

class IKSolver:
    """One IK solve pass: objectives, solver, and joint_q buffer.

    Three responsibilities:
    - owns the IK solver and its joint_q buffer
    - ``inherit_from_state`` pulls current FK body transforms into the
      position / rotation objective targets (mirrors
      ``_update_position_rotation_targets_from_states`` in NewtonPipeline)
    - ``solve`` runs the solver step, updates ``model.joint_q``, and evals FK
    """

    def __init__(
        self,
        ik_model,
        num_envs: int,
        model,
        position_objectives: List[ik.IKObjectivePosition],
        rotation_objectives: List[ik.IKObjectiveRotation],
        extra_objectives: list = None,
        reference_joint_q=None,
    ):
        self.ik_model = ik_model
        self.num_envs = num_envs
        self.model = model
        self.position_objectives = position_objectives
        self.rotation_objectives = rotation_objectives

        all_objectives = [*position_objectives, *rotation_objectives, *(extra_objectives or [])]
        self.solver = ik.IKSolver(
            model=ik_model,
            n_problems=num_envs,
            objectives=all_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.joint_q = wp.empty(shape=(num_envs, ik_model.joint_coord_count))
        if reference_joint_q is not None:
            for i in range(num_envs):
                wp.copy(self.joint_q[i], reference_joint_q)
        else:
            wp.copy(self.joint_q, model.joint_q)

        self.solver.reset()

    # ------------------------------------------------------------------
    # Objective factories
    # ------------------------------------------------------------------

    @staticmethod
    def make_position_objectives(
        num_envs: int,
        link_weights: list,
        pos_targets_per_link: list,
    ) -> List[ik.IKObjectivePosition]:
        """Build a list of ik.IKObjectivePosition.

        Args:
            link_weights: list of ``(link_idx, weight)`` tuples.
            pos_targets_per_link: list of ``(num_envs, 3)`` float32 arrays,
                one per entry in *link_weights*.
        """
        objectives = []
        for i, (link_idx, weight) in enumerate(link_weights):
            targets_wp = wp.array(pos_targets_per_link[i], dtype=wp.vec3)
            objectives.append(
                ik.IKObjectivePosition(
                    link_index=link_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=targets_wp,
                    weight=weight
                )
            )
        return objectives

    @staticmethod
    def make_rotation_objectives(
        num_envs: int,
        link_weights: list,
        rot_targets_per_link: list,
    ) -> List[ik.IKObjectiveRotation]:
        """Build a list of IKObjectiveRotation.

        Args:
            link_weights: list of ``(link_idx, weight)`` tuples.
            rot_targets_per_link: list of ``(num_envs, 4)`` float32 arrays
                (xyzw quaternions), one per entry in *link_weights*.
        """
        objectives = []
        for i, (link_idx, weight) in enumerate(link_weights):
            targets_wp = wp.array(rot_targets_per_link[i], dtype=wp.vec4)
            objectives.append(
                ik.IKObjectiveRotation(
                    link_index=link_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=targets_wp,
                    weight=weight
                )
            )
        return objectives

    @staticmethod
    def make_smooth_joint_filter(
        ik_model,
        weight: float = 0.0,
        coord_masks=None,
        offset_limit_lower=None,
        offset_limit_upper=None,
    ) -> IKSmoothJointFilter:
        """Build an IKSmoothJointFilter from the ik_model's joint limits.

        Args:
            ik_model: the IK model (provides ``joint_limit_lower/upper``).
            weight: initial weight; typically 0.0 and raised after warm-up.
            coord_masks: per-coord mask array; None handling delegated to IKSmoothJointFilter.
            offset_limit_lower: lower offset array; None handling delegated to IKSmoothJointFilter.
            offset_limit_upper: upper offset array; None handling delegated to IKSmoothJointFilter.
        """
        return IKSmoothJointFilter(
            joint_limit_lower=ik_model.joint_limit_lower,
            joint_limit_upper=ik_model.joint_limit_upper,
            weight=weight,
            coord_masks=coord_masks,
            offset_limit_lower=offset_limit_lower,
            offset_limit_upper=offset_limit_upper,
        )

    # ------------------------------------------------------------------
    # Per-frame operations
    # ------------------------------------------------------------------

    def inherit_from_state(self, state) -> None:
        """Copy current body positions/rotations from *state* into this pass's
        position and rotation objective targets.

        Call this before ``solve`` when this pass should track the result of a
        previous pass rather than the original retargeting targets.
        """
        body_count = self.ik_model.body_count
        for env_idx in range(self.num_envs):
            base = env_idx * body_count
            for obj in self.position_objectives:
                obj.set_target_position_from_transforms(
                    env_idx, base + obj.link_index, state.body_q
                )
            for obj in self.rotation_objectives:
                obj.set_target_rotation_from_transforms(
                    env_idx, base + obj.link_index, state.body_q
                )

    def solve(
        self,
        state,
        iterations: int
    ) -> None:
        """Run one IK solve step.

        Updates ``model.joint_q``, evaluates FK into *state*
        """
        self.solver.step(self.joint_q, self.joint_q, iterations=iterations)
        wp.copy(self.model.joint_q, self.joint_q.flatten())
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

    def sync_joint_q_from(self, source: wp.array) -> None:
        """Copy *source* into this pass's joint_q buffer.

        Useful when you want a later pass to continue from the result of an
        earlier one without going through ``inherit_from_state``.
        """
        wp.copy(self.joint_q, source)


# ---------------------------------------------------------------------------
# Concrete pass types
# ---------------------------------------------------------------------------

class BaseIKSolver(IKSolver):
    """Kinematic-only IK pass: position + rotation + optional constraint objectives.

    Identical to IKPass in behaviour; additionally stores per-objective scalar
    base weights as numpy arrays so a downstream pass can reference
    the original weights when restoring.
    """

    def __init__(
        self,
        ik_model,
        num_envs: int,
        model,
        position_objectives: List[ik.IKObjectivePosition],
        rotation_objectives: List[ik.IKObjectiveRotation],
        extra_objectives: list = None,
        reference_joint_q=None,
    ):
        super().__init__(ik_model, num_envs, model, position_objectives, rotation_objectives, extra_objectives, reference_joint_q)
        self.position_base_weights = np.array(
            [o.weight for o in self.position_objectives], dtype=np.float32
        )
        self.rotation_base_weights = np.array(
            [o.weight for o in self.rotation_objectives], dtype=np.float32
        )


