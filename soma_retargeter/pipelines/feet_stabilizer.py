# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

import newton
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.animation.ik as ik_utils
import soma_retargeter.utils.io_utils as io_utils

_LIMB_DATA_IDX_NAME = 0
_LIMB_DATA_IDX_EFFECTOR_INDICES = 1
_LIMB_DATA_IDX_HINT_REF = 2
_LIMB_DATA_IDX_HINT_OFFSET = 3


class FeetStabilizer:
    """
    FeetStabilizer class for managing inverse kinematics and feet stabilization for robotic motion transfer.
    """
    def __init__(self, config: str):
        """
        Initialize the feet stabilizer with the specified configuration.
        Args:
            config (str): Path to the configuration file.
        Raises:
            ValueError: If the robot type specified in the config is unknown.
        """
        self._load_config(config)

        if self.robot_type == 'unitree_g1':
            self.robot_builder = newton.ModelBuilder()
            self.robot_builder.add_mjcf(
                newton.utils.download_asset("unitree_g1") / "mjcf/g1_29dof_rev_1_0.xml")

            self.num_body_count = self.robot_builder.body_count
            self.ik_model = self._build_model(1)

            body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
            self.effector_mapped_indices = [body_names.index(body_name) for (body_name, _) in self.effectors.items()]
            self.effector_weights = [wp.vec2(*tr_weights) for (_, tr_weights) in self.effectors.items()]
            effector_parent_indices = [self.robot_builder.joint_parent[idx] for idx in self.effector_mapped_indices]

            self.pelvis_idx = self.effector_mapped_indices[self.ik_root]
            self.two_bone_ik_chains = wp.array2d([[self.effector_mapped_indices[i] for i in limb[_LIMB_DATA_IDX_EFFECTOR_INDICES]] for limb in self.ik_limb_data], dtype=wp.int32)
            self.two_bone_ik_chain_parent = wp.array([effector_parent_indices[limb[_LIMB_DATA_IDX_EFFECTOR_INDICES][0]] for limb in self.ik_limb_data], dtype=wp.int32)
            self.two_bone_ik_hint_references = wp.array([self.effector_mapped_indices[limb[_LIMB_DATA_IDX_HINT_REF]] for limb in self.ik_limb_data], dtype=wp.int32)
            self.two_bone_ik_hint_offsets = wp.array([limb[_LIMB_DATA_IDX_HINT_OFFSET] for limb in self.ik_limb_data], dtype=wp.vec3)

            self.num_envs = -1
        else:
            raise ValueError(f"[ERROR]: Unknown robot type {self.robot_type}")

    def setup_num_envs(self, num_envs):
        """
        Initialize the setup for the feet stabilizer with the specified number of environments.

        This method configures the model, state, joint parameters, and effectors for inverse kinematics
        computation across multiple parallel environments. It also initializes the objectives and solver.

        Args:
            num_envs (int): The number of parallel environments to set up.
        """
        self.num_envs = num_envs
        self.model = self._build_model(num_envs)
        self.state = self.model.state()
        self.joint_q = wp.array(self.model.joint_q, shape=(self.num_envs, self.ik_model.joint_coord_count))
        self.out_effectors = wp.empty(shape=[self.num_envs, self.num_effectors], dtype=wp.transform)
        self.reset_state()
        self._create_objectives_and_solver()

    def reset_state(self, joint_q=None):
        """
        Resets the current state of the feet stabilizer, optionally with a provided joint configuration..

        Args:
            joint_q (wp.array, optional): The joint configuration to reset to. If None, the default configuration is used.

        Raises:
            ValueError: If the provided joint_q does not match the expected shape for the model's joint coordinates.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if joint_q is not None:
            if joint_q.shape != self.joint_q.shape:
                raise ValueError(f"[ERROR]: joint_q size mismatch. Expected joint_q shape of [{self.joint_q.shape}] but received [{joint_q.shape}]")

            wp.copy(self.joint_q, joint_q)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def current_state(self):
        """Returns the current joint configuration of the model."""
        return self.joint_q

    def solve(self, targets_tx):
        """
        Solves the inverse kinematics problem for the specified target transforms of the effectors.

        Args:
            targets_tx (np.ndarray): An array of shape (num_envs, num_effectors, wp.transform) containing the target transforms for each effector in each environment.

        Raises:
            ValueError: If the number of environments has not been initialized or if the shape of targets_tx
                    does not match the expected shape based on the number of environments and effectors.
        """
        assert self.num_envs != -1, "[ERROR]: Environments have not been initialized. Call setup_num_envs to create a valid model."
        if targets_tx.shape != (self.num_envs, self.two_bone_ik_chains.shape[0], 7):
            raise ValueError(f"[ERROR]: targets_tx size mismatch. Expected targets_tx shape is [{(self.num_envs, self.two_bone_ik_chains.shape[0], 7)}] but received [{targets_tx.shape}]")

        @wp.kernel
        def solve_two_bone_ik_batched_kernel(
            in_body_q               : wp.array2d(dtype=wp.transform),
            in_pelvis_index         : wp.int32,
            in_num_ik_chains        : wp.int32,
            in_chain_indices        : wp.array2d(dtype=wp.int32),
            in_chain_parent_indices : wp.array1d(dtype=wp.int32),
            in_chain_hint_indices   : wp.array1d(dtype=wp.int32),
            in_chain_hint_offsets   : wp.array1d(dtype=wp.vec3),
            in_ik_targets           : wp.array2d(dtype=wp.transform),
            out_result              : wp.array2d(dtype=wp.transform)
        ):
            env = wp.tid()
            body_q = in_body_q[env]

            out_result[env, 0] = body_q[in_pelvis_index]
            offset = wp.int32(1)
            for i in range(in_num_ik_chains):
                chain_indices = in_chain_indices[i]
                chain_hint_idx = in_chain_hint_indices[i]

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
                    in_ik_targets[env, i],
                    use_hint,
                    chain_hint_world)

                out_result[env, offset + 0] = result.root
                out_result[env, offset + 1] = result.mid
                out_result[env, offset + 2] = result.tip
                offset += wp.int32(3)

        wp.launch(
            solve_two_bone_ik_batched_kernel,
            dim=self.num_envs,
            inputs=[
                self.state.body_q.reshape(shape=[self.num_envs, self.num_body_count]),
                self.pelvis_idx,
                self.two_bone_ik_chains.shape[0],
                self.two_bone_ik_chains,
                self.two_bone_ik_chain_parent,
                self.two_bone_ik_hint_references,
                self.two_bone_ik_hint_offsets,
                wp.array2d(targets_tx, dtype=wp.transform)],
                outputs=[self.out_effectors])

        out_results_np = self.out_effectors.numpy()
        for i in range(self.num_effectors):
            self.position_objectives[i].set_target_positions(wp.array(out_results_np[:, i, 0:3], dtype=wp.vec3))
            self.rotation_objectives[i].set_target_rotations(wp.array(out_results_np[:, i, 3:7], dtype=wp.vec4))

        if self.captured_graph is not None:
            wp.capture_launch(self.captured_graph)
        else:
            self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iterations)

    def _load_config(self, config: str):
        data = io_utils.load_json(config)
        self.robot_type = data['robot_type']
        self.ik_iterations = data['ik_iterations']
        self.joint_limit_weight = data['joint_limit_weight']

        self.effectors = data['effectors']
        self.num_effectors = len(self.effectors)

        self.ik_root = data['ik_root']
        self.ik_limb_data = []
        for label, values in data['ik_limbs'].items():
            self.ik_limb_data.append([label, values['effectors'], values['hint_reference'], wp.vec3(*values['hint_offset'])])

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
