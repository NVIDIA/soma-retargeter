# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.animation.ik as ik_utils
import soma_retargeter.io.utils as utils
import soma_retargeter.pipelines.utils as pipeline_utils

_LIMB_DATA_IDX_EFFECTOR_INDICES = 0
_LIMB_DATA_IDX_HINT_REF         = 1
_LIMB_DATA_IDX_HINT_OFFSET      = 2

@wp.kernel
def _root_correction_kernel(
    body_q: wp.array2d(dtype=wp.transform),
    targets: wp.array2d(dtype=wp.transform),
    chain_indices: wp.array2d(dtype=wp.int32),
    pelvis_idx: wp.int32,
    chains_to_process: wp.array(dtype=wp.vec2i),
    num_chains_to_process: wp.int32,
    correction_ratio: wp.float32,
    smooth_alpha: wp.float32,
    prev_root_z: wp.array1d(dtype=wp.float32),
    joint_q: wp.array2d(dtype=wp.float32),
):
    env = wp.tid()

    avg_z_delta = float(0.0)  # noqa: UP018
    for i in range(num_chains_to_process):
        chain_idx = chains_to_process[i].x
        target_idx = chains_to_process[i].y
        tip_body_idx = chain_indices[chain_idx, 2]
        fk_foot_z = wp.transform_get_translation(body_q[env, tip_body_idx])[2]
        target_foot_z = wp.transform_get_translation(targets[env, target_idx])[2]
        avg_z_delta = avg_z_delta + (target_foot_z - fk_foot_z)
    avg_z_delta = avg_z_delta / float(num_chains_to_process)

    avg_verticality = float(0.0)  # noqa: UP018
    for i in range(num_chains_to_process):
        chain_idx = chains_to_process[i].x
        tip_body_idx = chain_indices[chain_idx, 2]
        tip_pos = wp.transform_get_translation(body_q[env, tip_body_idx])
        pelvis_pos = wp.transform_get_translation(body_q[env, pelvis_idx])
        leg_vec = tip_pos - pelvis_pos
        leg_len = wp.length(leg_vec)
        if leg_len > 1.0e-8:
            vert = leg_vec[2] / leg_len
            avg_verticality = avg_verticality + vert * vert
    avg_verticality = avg_verticality / float(num_chains_to_process)

    raw_z = correction_ratio * avg_z_delta * avg_verticality
    smoothed_z = smooth_alpha * raw_z + (1.0 - smooth_alpha) * prev_root_z[env]
    prev_root_z[env] = smoothed_z
    joint_q[env, 2] = joint_q[env, 2] + smoothed_z


class LimbStabilizer:
    def __init__(self, robot_type: str, active_human_ik_target_names: list[str], active_human_ik_mapped_indices: list[int], config: str | dict):
        assert len(active_human_ik_target_names) > 0, "[ERROR]: At least one human IK target must be enabled."
        assert len(active_human_ik_target_names) == len(active_human_ik_mapped_indices), "[ERROR]: active_human_ik_target_names and active_human_ik_mapped_indices must have the same length."

        self._load_config(active_human_ik_target_names, active_human_ik_mapped_indices, config)

        self.robot_builder = pipeline_utils.create_robot_builder(robot_type)
        self.num_body_count = self.robot_builder.body_count
        self.ik_model = self._build_model(1)

        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        self.effector_mapped_indices = [body_names.index(body_name) for (body_name, _) in self.effectors.items()]
        self.effector_body_indices = wp.array(self.effector_mapped_indices, dtype=wp.int32)
        self.effector_weights = [wp.vec2(*tr_weights) for (_, tr_weights) in self.effectors.items()]
        effector_parent_indices = [self.robot_builder.joint_parent[idx] for idx in self.effector_mapped_indices]

        self.pelvis_idx = self.effector_mapped_indices[self.ik_root]
        self.two_bone_ik_effector_indices = wp.array2d([limb[_LIMB_DATA_IDX_EFFECTOR_INDICES] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_chains = wp.array2d([[self.effector_mapped_indices[i] for i in limb[_LIMB_DATA_IDX_EFFECTOR_INDICES]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_chain_parent = wp.array([effector_parent_indices[limb[_LIMB_DATA_IDX_EFFECTOR_INDICES][0]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_hint_references = wp.array([self.effector_mapped_indices[limb[_LIMB_DATA_IDX_HINT_REF]] for limb in self.ik_limb_data], dtype=wp.int32)
        self.two_bone_ik_hint_offsets = wp.array([limb[_LIMB_DATA_IDX_HINT_OFFSET] for limb in self.ik_limb_data], dtype=wp.vec3)

        self.num_envs = -1
        self._prev_root_z_correction = wp.zeros(1, dtype=wp.float32)

    def setup_num_envs(self, num_envs):
        """Allocate per-environment model, state, and IK solver; must be called before ``solve``.

        Args:
            num_envs: Number of parallel environments to simulate.
        """
        self.num_envs = num_envs
        self._prev_root_z_correction = wp.zeros(num_envs, dtype=wp.float32)
        self.model = self._build_model(num_envs)
        self.state = self.model.state()
        self.joint_q = wp.array(self.model.joint_q, shape=(self.num_envs, self.ik_model.joint_coord_count))
        self.out_effectors = wp.empty(shape=[self.num_envs, self.num_effectors], dtype=wp.transform)
        self.reset_state()
        self._create_objectives_and_solver()

    def reset_state(self, joint_q=None):
        """Reset FK state, optionally seeding joint coordinates from *joint_q*.

        Args:
            joint_q: Optional ``(num_envs, joint_coord_count)`` Warp array to copy
                into the internal buffer before evaluating FK. Omit to re-evaluate
                FK from the current buffer.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if joint_q is not None:
            if joint_q.shape != self.joint_q.shape:
                raise ValueError(f"[ERROR]: joint_q size mismatch. Expected joint_q shape of [{self.joint_q.shape}] but received [{joint_q.shape}]")

            wp.copy(self.joint_q, joint_q)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def current_state(self):
        return self.joint_q

    def solve(self, targets_tx):
        """Run root correction, two-bone IK per limb, and the full IK solve.

        Updates ``self.joint_q`` in-place; call ``current_state()`` afterwards
        to retrieve results.

        Args:
            targets_tx: ``(num_envs, num_limbs, 7)`` float32 array of
                world-space effector target transforms (position + quaternion).
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if (targets_tx.shape[0] != self.num_envs
            or targets_tx.shape[1] < self.two_bone_ik_chains.shape[0]
            or targets_tx.shape[2] != 7):
            raise ValueError(f"[ERROR]: targets_tx size mismatch. Expected targets_tx shape is [{(self.num_envs, self.two_bone_ik_chains.shape[0], 7)}] but received [{targets_tx.shape}]")

        if self.ik_limb_target_indices is None:
            # No compatible IK targets, skip IK limb solver altogether.
            # Warning has already been printed during config loading.
            return

        targets_wp = wp.array2d(targets_tx, dtype=wp.transform)

        if self.ik_feet_indices is not None and self.root_correction_ratio > 0.0:
            wp.launch(
                _root_correction_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state.body_q.reshape(shape=[self.num_envs, self.num_body_count]),
                    targets_wp,
                    self.two_bone_ik_chains,
                    self.pelvis_idx,
                    self.ik_feet_indices,
                    self.ik_feet_indices.shape[0],
                    float(self.root_correction_ratio),
                    float(self.root_smooth_alpha),
                    self._prev_root_z_correction,
                    self.joint_q,
                ],
            )
            wp.copy(self.model.joint_q, self.joint_q.flatten())
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        @wp.kernel
        def reset_effectors_kernel(
            in_body_q                : wp.array2d(dtype=wp.transform),
            in_num_effectors         : wp.int32,
            in_effector_body_indices : wp.array1d(dtype=wp.int32),
            out_result               : wp.array2d(dtype=wp.transform)
        ):
            env = wp.tid()
            body_q = in_body_q[env]

            for i in range(in_num_effectors):
                out_result[env, i] = body_q[in_effector_body_indices[i]]

        @wp.kernel
        def solve_two_bone_ik_batched_kernel(
            in_body_q                : wp.array2d(dtype=wp.transform),
            in_num_ik_chains         : wp.int32,
            in_chain_effector_indices: wp.array2d(dtype=wp.int32),
            in_chain_indices         : wp.array2d(dtype=wp.int32),
            in_chain_parent_indices  : wp.array1d(dtype=wp.int32),
            in_chain_hint_indices    : wp.array1d(dtype=wp.int32),
            in_chain_hint_offsets    : wp.array1d(dtype=wp.vec3),
            in_ik_targets_indices    : wp.array(dtype=wp.int32),
            in_ik_targets            : wp.array2d(dtype=wp.transform),
            out_result               : wp.array2d(dtype=wp.transform)
        ):
            env = wp.tid()
            body_q = in_body_q[env]

            for i in range(in_num_ik_chains):
                chain_indices  = in_chain_indices[i]
                chain_effector_indices = in_chain_effector_indices[i]
                chain_hint_idx = in_chain_hint_indices[i]
                chain_target_idx = in_ik_targets_indices[i]


                use_hint = chain_hint_idx != -1
                chain_hint_world = wp.vec3(0.0, 0.0, 0.0)
                if use_hint:
                    chain_hint_world = wp.transform_point(body_q[chain_hint_idx], in_chain_hint_offsets[i])

                result = ik_utils.wp_solve_two_bone_ik(
                    1.0,
                    body_q[in_chain_parent_indices[i]],
                    body_q[chain_indices[0]],
                    body_q[chain_indices[1]],
                    body_q[chain_indices[2]],
                    in_ik_targets[env, chain_target_idx],
                    use_hint,
                    chain_hint_world)

                out_result[env, chain_effector_indices[0]] = result.root
                out_result[env, chain_effector_indices[1]] = result.mid
                out_result[env, chain_effector_indices[2]] = result.tip

        body_q_2d = self.state.body_q.reshape(shape=[self.num_envs, self.num_body_count])
        wp.launch(
            reset_effectors_kernel,
            dim=self.num_envs,
            inputs=[
                body_q_2d,
                self.num_effectors,
                self.effector_body_indices],
                outputs=[self.out_effectors])

        wp.launch(
            solve_two_bone_ik_batched_kernel,
            dim=self.num_envs,
            inputs=[
                body_q_2d,
                self.two_bone_ik_chains.shape[0],
                self.two_bone_ik_effector_indices,
                self.two_bone_ik_chains,
                self.two_bone_ik_chain_parent,
                self.two_bone_ik_hint_references,
                self.two_bone_ik_hint_offsets,
                self.ik_limb_target_indices,
                targets_wp],
                outputs=[self.out_effectors])

        out_results_np = self.out_effectors.numpy()
        for i in range(self.num_effectors):
            self.position_objectives[i].set_target_positions(wp.array(out_results_np[:, i, 0:3], dtype=wp.vec3))
            self.rotation_objectives[i].set_target_rotations(wp.array(out_results_np[:, i, 3:7], dtype=wp.vec4))

        if self.captured_graph is not None:
            wp.capture_launch(self.captured_graph)
        else:
            self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)

    def _load_config(self, active_human_ik_target_names: list[str], active_human_ik_mapped_indices: list[int], config: str | dict):
        data = config if isinstance(config, dict) else utils.load_json(config)
        self.ik_iterations = data['ik_iterations']
        self.joint_limit_weight = data['joint_limit_weight']

        self.effectors = data['effectors']
        self.num_effectors = len(self.effectors)

        self.ik_root = data['ik_root']
        self.root_correction_ratio = data.get('root_correction_ratio', 0.0)
        self.root_smooth_alpha = float(np.clip(data.get('root_smooth_alpha', 0.4), 0.0, 1.0))

        # Sort limb data based on the order of active human IK target provided
        # and filter out any inactive targets
        ik_limbs = data['ik_limbs']
        ik_limb_target_indices = []
        self.ik_limb_data = []
        feet_indices = []
        for i, target_name in enumerate(active_human_ik_target_names):
            if target_name in ik_limbs and active_human_ik_mapped_indices[i] != -1:
                ik_limb_target_indices.append(i)
                if 'Foot' in target_name:
                    feet_indices.append(wp.vec2i(len(self.ik_limb_data), i))
                values = ik_limbs[target_name]
                self.ik_limb_data.append([values['effectors'], values['hint_reference'], wp.vec3(*values['hint_offset'])])

        self.ik_limb_target_indices = wp.array(ik_limb_target_indices, dtype=wp.int32) if len(ik_limb_target_indices) > 0 else None
        self.ik_feet_indices = wp.array(feet_indices, dtype=wp.vec2i) if len(feet_indices) > 0 else None

        if self.ik_limb_target_indices is None:
            print("[WARNING]: No compatible IK targets found in the limb_stabilizer config! Please verify your post processing robot_config file.")

    def _create_objectives_and_solver(self):
        body_q_np = self.state.body_q.numpy().reshape(self.num_envs, self.num_body_count, 7)
        pos_effector_arrays, rot_effector_arrays = [], []
        for i in range(self.num_effectors):
            body_idx = self.effector_mapped_indices[i]
            pos_effector_arrays.append(wp.array(body_q_np[:, body_idx, 0:3], dtype=wp.vec3))
            rot_effector_arrays.append(wp.array(body_q_np[:, body_idx, 3:7], dtype=wp.vec4))

        self.position_objectives = []
        self.rotation_objectives = []
        for i in range(self.num_effectors):
            body_idx = self.effector_mapped_indices[i]
            t_weight = self.effector_weights[i][0]
            r_weight = self.effector_weights[i][1]
            self.position_objectives.append(
                newton.ik.IKObjectivePosition(
                    link_index=body_idx,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=pos_effector_arrays[i],
                    weight=t_weight
                    )
                )
            self.rotation_objectives.append(
                newton.ik.IKObjectiveRotation(
                    link_index=body_idx,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=rot_effector_arrays[i],
                    weight=r_weight
                    )
                )

        # Joint limit objective
        self.joint_limit_objective = newton.ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)

        self.ik_solver = newton.ik.IKSolver(
            model=self.ik_model,
            objectives=[*self.position_objectives, *self.rotation_objectives, self.joint_limit_objective],
            lambda_initial=0.1,
            n_problems=self.num_envs,
            jacobian_mode=newton.ik.IKJacobianType.ANALYTIC)

        self.ik_solver.reset()
        self.captured_graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)
            self.captured_graph = cap.graph

    def _build_model(self, num_envs: int):
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_builder(self.robot_builder, xform=wp.transform_identity())
        builder.add_ground_plane()
        return builder.finalize()
