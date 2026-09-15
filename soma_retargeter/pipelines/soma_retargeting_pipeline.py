# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-pass IK retargeting pipeline with contact detection and foot-plant correction for humanoid-to-robot motion transfer."""
import os
from typing import Optional

import warp as wp
import numpy as np

import newton
import newton.ik as ik
import soma_retargeter.io.bvh as bvh_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.pose_utils as pose_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.robotics.robot_registry as robot_registry
from soma_retargeter.pipelines.ik_solver import IKSolver, BaseIKSolver
from soma_retargeter.pipelines.ik_pass import BasePass, PostProcessPass

from soma_retargeter.animation.skeleton import Skeleton, SkeletonInstance
from soma_retargeter.animation.animation_buffer import AnimationBuffer
from soma_retargeter.animation.contact_detection import (
    BaselineAThresholds,
    BaselineAResult,
    detect_contacts_velocity_jerk,
)
from soma_retargeter.robotics.human_to_robot_scaler import HumanToRobotScaler
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer
from soma_retargeter.pipelines.limb_stabilizer import LimbStabilizer
from soma_retargeter.pipelines.joint_limit_clamper import JointLimitClamper
from soma_retargeter.animation.contact_phase import (
    load_authored_foot_landmark_model,
    build_heuristic_foot_landmark_model,
)
from soma_retargeter.pipelines.plant_subsegment_detector import (
    detect_plant_subsegments,
    PlantSubsegmentConfig,
    PlantSubsegmentResult,
)
from soma_retargeter.pipelines.plant_correction_blender import (
    PlantCorrectionBlender,
    DEFAULT_CONTACT_JOINTS,
    DEFAULT_FOOT_CHANNEL_MAP,
)
from tqdm import trange

_DEFAULT_IK_SOLVER_ITERATIONS = 24
_DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT = 10.0
_DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT = 5.5
_DEFAULT_SELF_PENETRATION_OBJECTIVE_WEIGHT = 0.0
_DEFAULT_SELF_PENETRATION_SUB_STEP_COUNT = 3
_DEFAULT_NUM_INITIALIZATION_FRAMES = 10
_DEFAULT_NUM_STABILIZATION_FRAMES = 5

class SomaRetargetingPipeline:
    """
    Newton-based motion retargeting pipeline with contact-aware post-processing.

    This pipeline retargets human motion captured on a common skeleton
    to a target robot using inverse kinematics (IK), custom objectives,
    and optional post-processing filters such as joint limit clamping,
    limb stabilization, foot-plant detection, and plant correction blending.
    """

    def __init__(self, skeleton: Skeleton, source_type, robot_type = 'unitree_g1', retarget_config: dict = None):
        """Initialize the retargeting pipeline.

        Args:
            skeleton: Common skeleton definition shared by all input clips.
            source_type: Source skeleton type. Should be ``'soma'``.
            robot_type: Target robot type name. Use
                ``robot_registry.list_available_targets()`` to see registered names.
            retarget_config: Optional configuration dictionary. If ``None``, loaded
                from disk based on the source and robot types.

        Raises:
            ValueError: If the retargeting config has no IK matches, or if
                post-processing is enabled but its config is missing or does
                not contain a ``'limb_stabilizer'`` section.
        """
        self.source_type = pipeline_utils.get_source_type_from_str(source_type)
        self.target_type = robot_type
        self.robot_asset_root = robot_registry.get_robot_asset_root(robot_type)
        self.input_targets           = []
        self.input_sample_rates      = []
        self.max_frames              = -1

        if retarget_config is None:
            retargeter_config = pipeline_utils.get_retargeter_config(source_type, robot_type)
        else:
            retargeter_config = retarget_config

        ik_match_table = retargeter_config.get("ik_match_table")
        if not isinstance(ik_match_table, dict) or not ik_match_table:
            raise ValueError(
                f"Retargeting aborted for target [{robot_type}] due to missing or empty "
                "'ik_match_table'. Please review it in the Robot Configurator."
            )

        self.ik_iterations = retargeter_config.get('ik_iterations', _DEFAULT_IK_SOLVER_ITERATIONS)
        self.joint_limit_weight = retargeter_config.get('joint_limit_weight', _DEFAULT_JOINT_LIMIT_OBJECTIVE_WEIGHT)
        self.smooth_joint_filter_weight = retargeter_config.get('smooth_joint_filter_weight', _DEFAULT_SMOOTH_JOINT_FILTER_OBJECTIVE_WEIGHT)
        self.post_processing_enabled = retargeter_config.get('enable_post_processing', True)
        self.self_penetration_weight = retargeter_config.get('self_penetration_weight', _DEFAULT_SELF_PENETRATION_OBJECTIVE_WEIGHT)
        self.self_penetration_sub_step_count = retargeter_config.get('self_penetration_sub_step_count', _DEFAULT_SELF_PENETRATION_SUB_STEP_COUNT)
        self.smooth_joint_filter_coord_masks = None
        self.smooth_joint_limit_offsets = [None, None]
        self.joint_limit_clamper = None

        self.robot_builder = pipeline_utils.create_robot_builder(robot_type)
        self.human_robot_scaler = HumanToRobotScaler(
            skeleton, os.path.join(self.robot_asset_root, retargeter_config['human_robot_scaler_config']))

        self.num_body_count = self.robot_builder.body_count
        self.num_dofs = self.robot_builder.joint_dof_count
        # this is with assuming we don't have more than 1000000 environments
        self.ik_model = self._build_model(1)

        (
            self.mapped_joints,
            self.mapped_joint_indices,
            self.mapped_body_link_pos_data,
            self.mapped_body_link_rot_data
        ) = self._build_target_mapping(
            self.ik_model,
            self.human_robot_scaler.skeleton,
            retargeter_config)

        smooth_joint_filter_objective_body_masks = retargeter_config.get('smooth_joint_filter_objective_body_masks', None)
        self.smooth_joint_filter_coord_masks, self.smooth_joint_limit_offsets = newton_utils.create_joint_coord_masks(
            self.ik_model, smooth_joint_filter_objective_body_masks, 0.0)
        if int(np.count_nonzero(self.smooth_joint_filter_coord_masks)) == 0:
            self.smooth_joint_filter_weight = 0.0

        effector_names = self.human_robot_scaler.effector_names()
        self.target_effector_indices = [effector_names.index(name) for name in self.mapped_joints]
        self.limb_target_ik_names = ["LeftFoot", "RightFoot", "LeftHand", "RightHand"]
        self.limb_effector_indices = [
            self.mapped_joints.index(name) if name in self.mapped_joints else -1
            for name in self.limb_target_ik_names]

        target_offsets = self.human_robot_scaler.mapped_joint_offsets.numpy()[self.target_effector_indices]

        self._feet_effector_indices = self.limb_effector_indices[0:2]
        self._feet_effector_scale = self.human_robot_scaler.human_scale_ratio
        self._feet_effector_ik_offsets = np.asarray(target_offsets[self._feet_effector_indices], dtype=np.float32)

        self.joint_limit_clamper = JointLimitClamper(self.ik_model)

        zero_pose_path = pipeline_utils.get_source_zero_pose_asset_path(self.source_type)
        init_skel, init_anim = bvh_utils.load_bvh(zero_pose_path)
        self.initialization_pose = SkeletonInstance(init_skel, [0, 0, 0], wp.transform_identity())
        self.initialization_pose.set_local_transforms(init_anim.get_local_transforms(0))
        self.num_initialization_frames = max(0, retargeter_config.get('num_initialization_frames', _DEFAULT_NUM_INITIALIZATION_FRAMES))
        self.num_stabilization_frames = max(0, retargeter_config.get('num_stabilization_frames', _DEFAULT_NUM_STABILIZATION_FRAMES))

        # Post-processing & contact detection config
        self.contact_processing_enabled: bool = retargeter_config.get('enable_contact_processing', False)
        self.post_processing_config: Optional[dict] = None
        self.limb_stabilizer = None

        if self.post_processing_enabled:
            self.post_processing_config = pipeline_utils.resolve_post_processing(retargeter_config, self.source_type, self.robot_asset_root)
            if self.post_processing_config is None or "limb_stabilizer" not in self.post_processing_config:
                raise ValueError(
                    "enable_post_processing is set but post_processing config is missing "
                    "or does not contain a 'limb_stabilizer' section")
            self.limb_stabilizer = LimbStabilizer(
                self.target_type, self.limb_target_ik_names, self.limb_effector_indices, self.post_processing_config["limb_stabilizer"])

        self.auto_detect_contacts: bool = retargeter_config.get(
            'auto_detect_contacts',
            self.contact_processing_enabled,
        )
        self.contact_results: list[BaselineAResult] = []
        self.plant_subsegment_results: list[PlantSubsegmentResult] = []
        self.plant_blenders: list[PlantCorrectionBlender] = []
        self._landmark_model = None
        self._foot_landmarks_data: Optional[dict] = None
        self._foot_landmarks_config_path: Optional[str] = None

        if self.contact_processing_enabled and self.post_processing_config is not None:
            cd = self.post_processing_config
            self._foot_landmarks_data = cd.get("foot_landmarks")
            self._foot_landmarks_config_path = cd.get("foot_landmarks_path")
            self._contact_thresholds = BaselineAThresholds(
                velocity_contact=cd.get('velocity_contact', BaselineAThresholds.velocity_contact),
                jerk_contact=cd.get('jerk_contact', BaselineAThresholds.jerk_contact),
                velocity_uncontact=cd.get('velocity_uncontact', BaselineAThresholds.velocity_uncontact),
                velocity_probably_lift_off=cd.get('velocity_probably_lift_off', BaselineAThresholds.velocity_probably_lift_off),
                post_contact_jerk_window=cd.get('post_contact_jerk_window', BaselineAThresholds.post_contact_jerk_window),
                transition_frames=cd.get('transition_frames', BaselineAThresholds.transition_frames),
                edge_propagation_frames=cd.get('edge_propagation_frames', BaselineAThresholds.edge_propagation_frames),
            )
            self._contact_joint_names = cd.get('joint_names', DEFAULT_CONTACT_JOINTS)
            raw_map = cd.get('foot_channel_map', None)
            self._foot_channel_map = {int(k): v for k, v in raw_map.items()} if raw_map else dict(DEFAULT_FOOT_CHANNEL_MAP)
            self._blend_transition_frames = int(
                cd.get(
                    "blend_transition_frames",
                    cd.get("transition_frames", BaselineAThresholds.transition_frames),
                )
            )
            self._propagation_ratio = cd.get('propagation_ratio', 0.4)
            self._rotation_propagation_ratio = cd.get('rotation_propagation_ratio', 0.15)
            self._enable_flatten_foot_plant = cd.get('enable_flatten_foot_plant', False)
            self._sole_normal_local = cd.get('sole_normal_local', None)
            self._plant_subsegment_config = PlantSubsegmentConfig.from_dict(cd.get('plant_subsegment', None))
        else:
            self._contact_thresholds = BaselineAThresholds()
            self._contact_joint_names = list(DEFAULT_CONTACT_JOINTS)
            self._foot_channel_map = dict(DEFAULT_FOOT_CHANNEL_MAP)
            self._blend_transition_frames = BaselineAThresholds.transition_frames
            self._propagation_ratio = 0.4
            self._rotation_propagation_ratio = 0.15
            self._enable_flatten_foot_plant = False
            self._sole_normal_local = None
            self._plant_subsegment_config = PlantSubsegmentConfig()

    def clear(self):
        """Clear all accumulated input motions and reset per-run state.

        The robot model, IK settings, and contact configuration are preserved
        so the pipeline can be reused for a new batch without re-initializing.
        """
        self.input_targets      = []
        self.input_sample_rates = []
        self.max_frames         = -1
        self.contact_results    = []
        self.plant_subsegment_results = []
        self.plant_blenders     = []

    def _needs_contact_processing(self) -> bool:
        """Return True if contact and plant-subsegment processing should run this pass."""
        return (
            self.post_processing_enabled
            and self.contact_processing_enabled
            and self.post_processing_config is not None
        )

    def set_contact_data(
        self,
        plant_subsegment_results: list[PlantSubsegmentResult],
    ):
        """Attach externally-computed plant subsegment data for foot stabilization.

        Call after add_input_motions() and before execute().
        """
        self.plant_subsegment_results = list(plant_subsegment_results)
        self._build_plant_blenders()

    def _detect_contacts_and_plants(
        self,
        animation: AnimationBuffer,
        root_tx: wp.transform,
    ) -> tuple[BaselineAResult, PlantSubsegmentResult]:
        """Run contact detection and plant-subsegment detection for a single animation.

        Args:
            animation: Input animation buffer in source-skeleton space.
            root_tx: Root transform applied to the animation for world-space evaluation.

        Returns:
            A tuple of ``(BaselineAResult, PlantSubsegmentResult)`` for the animation.
        """
        all_global_tx = pose_utils.compute_global_poses_batch(
            animation.skeleton,
            animation.local_transforms,
            root_tx,
        )
        result = detect_contacts_velocity_jerk(
            animation.skeleton,
            animation,
            root_tx,
            self._contact_joint_names,
            self._contact_thresholds,
            all_global_tx=all_global_tx,
        )
        skeleton_type = pipeline_utils.get_source_str_from_type(self.source_type)
        if self._landmark_model is None:
            landmark_source = self._foot_landmarks_data or self._foot_landmarks_config_path
            self._landmark_model = load_authored_foot_landmark_model(
                animation.skeleton,
                skeleton_type,
                landmark_source,
            )
            if self._landmark_model is None:
                self._landmark_model = build_heuristic_foot_landmark_model(
                    animation.skeleton, skeleton_type, root_tx,
                )
        plant_result = detect_plant_subsegments(
            result,
            animation.skeleton,
            animation,
            root_tx,
            skeleton_type,
            self._contact_joint_names,
            self._foot_channel_map,
            config=self._plant_subsegment_config,
            all_global_tx=all_global_tx,
            landmark_model=self._landmark_model,
        )
        return result, plant_result

    def detect_contacts(
        self,
        animations: list[AnimationBuffer],
        root_txs: list[wp.transform],
        build_blenders: bool = True,
    ) -> list[BaselineAResult]:
        """Run contact and plant-subsegment detection for each animation.

        Args:
            animations: Original ``AnimationBuffer`` objects without initialization frames.
            root_txs: Per-animation root transforms used for world-space evaluation.
            build_blenders: If ``True`` and input targets are already loaded, also
                rebuild ``PlantCorrectionBlender`` instances.

        Returns:
            List of ``BaselineAResult``, one per animation.
        """
        self.contact_results = []
        self.plant_subsegment_results = []
        for anim, tx in zip(animations, root_txs):
            result, plant_result = self._detect_contacts_and_plants(anim, tx)
            self.contact_results.append(result)
            self.plant_subsegment_results.append(plant_result)
        if build_blenders and self.input_targets and self.contact_processing_enabled:
            self._build_plant_blenders()
        return self.contact_results

    def _build_plant_blenders(self) -> None:
        """Create PlantCorrectionBlender instances from plant subsegment results."""
        if not self.contact_processing_enabled or not self.plant_subsegment_results:
            self.plant_blenders = []
            return
        self.plant_blenders = []
        for env, plant_result in enumerate(self.plant_subsegment_results):
            if env >= len(self.input_targets):
                break
            blender = PlantCorrectionBlender(
                transition_frames=self._blend_transition_frames,
                propagation_ratio=self._propagation_ratio,
                rotation_propagation_ratio=self._rotation_propagation_ratio,
                enable_flatten_foot_plant=self._enable_flatten_foot_plant,
                sole_normal_local=self._sole_normal_local,
            )
            blender.build(
                plant_result,
                self.input_targets[env],
                self._feet_effector_indices,
                self.num_initialization_frames,
                self.num_stabilization_frames,
                foot_effector_scale=self._feet_effector_scale,
                foot_effector_ik_offsets=self._feet_effector_ik_offsets,
            )
            self.plant_blenders.append(blender)

    def add_input_motions(self, buffers: list[AnimationBuffer], offsets: list[wp.transform], scale_animation: bool):
        """Preprocess animation buffers into effector targets and optionally run contact detection.

        Prepends initialization and stabilization frames to each buffer, scales effectors,
        and stores the result for use in ``execute``.

        Args:
            buffers: One ``AnimationBuffer`` per environment.
            offsets: Per-buffer root-space offsets; falls back to identity if length mismatches.
            scale_animation: Whether to apply the human-to-robot scale ratio to effector positions.
        """
        # Prepend init/stabilization frames, compute scaled effectors, and optionally run contact detection
        offsets = offsets if len(offsets) == len(buffers) else [wp.transform_identity()] * len(buffers)
        num_frames_to_insert = self.num_initialization_frames + self.num_stabilization_frames

        for i in trange(len(buffers), desc="[INFO] Converting Motions for Newton"):
            buffer = buffers[i]
            if buffer.num_frames <= 0:
                raise ValueError(f"Input buffer {i} has no frames.")

            if num_frames_to_insert > 0:
                buffer = newton_utils.create_buffer_with_initialization_frames(
                    self.initialization_pose, buffers[i], self.num_initialization_frames, self.num_stabilization_frames)

            self.max_frames = max(self.max_frames, buffer.num_frames)
            buffer_effectors = self.human_robot_scaler.compute_effectors_from_buffer(buffer, scale_animation, offsets[i])

            self.input_targets.append(buffer_effectors[:,self.target_effector_indices,:])
            self.input_sample_rates.append(buffers[i].sample_rate)

        if self._needs_contact_processing() and self.auto_detect_contacts:
            self.detect_contacts(buffers, offsets, build_blenders=False)

    def execute(self):
        """Run the full IK retargeting pipeline and return one output buffer per environment.

        Builds Newton models, constructs IK objectives, captures CUDA graphs, and
        iterates over all frames.  Initialization and stabilization frames are stripped
        from the output.  Joint coordinates are clamped to limits before returning.

        Returns:
            List of ``CSVAnimationBuffer`` objects, one per input environment.
        """
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return

        # Clamp objective weights to valid values
        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)

        print("[INFO] Soma Retargeter Settings: ")
        print(f"[INFO]\t  Source Skeleton Type: {pipeline_utils.get_source_str_from_type(self.source_type)}")
        print(f"[INFO]\t  Target Robot Type: {self.target_type}")
        print(f"[INFO]\t  Post-Processing Enabled: {self.post_processing_enabled}")
        print(f"[INFO]\t  Contact Processing Enabled: {self.contact_processing_enabled}")
        print(f"[INFO]\t  Initialization Pose: {self.initialization_pose is not None}")
        print(f"[INFO]\t  Initialization Frame Count: {self.num_initialization_frames}")
        print(f"[INFO]\t  Constraint Stabilization Frame Count: {self.num_stabilization_frames}")
        print(f"[INFO]\t  IK Solver Iterations: {self.ik_iterations}")
        print(f"[INFO]\t  Joint Limit Objective Weight: {self.joint_limit_weight}")
        print(f"[INFO]\t  Smooth Joint Filter Objective Weight: {self.smooth_joint_filter_weight}")

        if self._needs_contact_processing() and self.plant_subsegment_results and not self.plant_blenders:
            self._build_plant_blenders()

        model = self._build_model(num_envs)
        state = model.state()

        if self.post_processing_enabled:
            self.limb_stabilizer.setup_num_envs(num_envs)
            env_limb_effector_tx = np.empty((num_envs, len(self.limb_effector_indices), 7), dtype=np.float32)

        (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_filter_objective
        ) = self._create_ik_objectives(num_envs, model, state)

        extra_objectives = []
        if self.joint_limit_weight > 0.0:
            extra_objectives.append(joint_limit_objective)
        if self.smooth_joint_filter_weight > 0.0:
            extra_objectives.append(smooth_joint_filter_objective)

        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames

        base_ik_solver = BaseIKSolver(
            self.ik_model, num_envs, model,
            position_objectives=position_objectives,
            rotation_objectives=rotation_objectives,
            extra_objectives=extra_objectives,
            reference_joint_q=self.human_robot_scaler.reference_joint_q)

        # shared joint_q written by every pass; all later passes hold a reference to this buffer
        joint_q = wp.clone(base_ik_solver.joint_q)

        passes = []

        passes.append(BasePass(
            ik_solver=base_ik_solver,
            joint_q=joint_q,
            num_envs=num_envs,
            input_targets=self.input_targets,
            position_objectives=position_objectives,
            rotation_objectives=rotation_objectives,
            smooth_filter_objective=smooth_joint_filter_objective,
            smooth_filter_weight=self.smooth_joint_filter_weight,
            num_frames_to_remove=num_frames_to_remove,
            ik_iterations=self.ik_iterations,
            state=state,
        ))

        if self.post_processing_enabled:
            passes.append(PostProcessPass(
                joint_q=joint_q,
                num_envs=num_envs,
                model=model,
                state=state,
                limb_stabilizer=self.limb_stabilizer,
                input_targets=self.input_targets,
                limb_effector_indices=self.limb_effector_indices,
                env_limb_effector_tx=env_limb_effector_tx,
                contact_processing_enabled=self.contact_processing_enabled,
                plant_blenders=self.plant_blenders,
            ))

        n_coords = joint_q.shape[1]
        joint_q_frames_wp = [
            wp.empty(
                (len(self.input_targets[env]) - num_frames_to_remove, n_coords),
                dtype=wp.float32,
            )
            for env in range(num_envs)
        ]

        # main pass for every frame
        for frame in trange(self.max_frames, desc="[INFO] Retargeting Motions"):
            for pass_ in passes:
                pass_.execute(frame)

            # record to joint_q_frames_wp buffer
            if frame >= num_frames_to_remove:
                for env in range(num_envs):
                    if frame > (len(self.input_targets[env])-1):
                        continue
                    wp.copy(joint_q_frames_wp[env][frame - num_frames_to_remove], joint_q[env])
        #clamp
        for env in range(num_envs):
            self.joint_limit_clamper.apply(joint_q_frames_wp[env])

        return [CSVAnimationBuffer.create_from_raw_data(joint_q_frames_wp[i].numpy(), self.input_sample_rates[i])
            for i in range(num_envs)]

    def _build_model(self, num_envs: int):
        """Build a Newton model containing ``num_envs`` robot copies for batched simulation.

        Args:
            num_envs: Number of parallel environments to instantiate.

        Returns:
            A finalized ``newton.Model`` ready for IK solving.
        """
        builder = newton.ModelBuilder()
        for _ in range(num_envs):
            builder.add_world(self.robot_builder, xform=wp.transform_identity())

        builder.add_ground_plane()
        model = builder.finalize()

        return model

    def _build_target_mapping(self, model, skeleton, retargeter_config):
        """Map human joint names from ``ik_match_table`` to robot body link indices.

        Returns:
            A tuple of ``(mapped_joints, mapped_joint_indices,
            mapped_body_link_pos_data, mapped_body_link_rot_data)``.
        """
        mapped_joints = []
        mapped_joint_indices = []
        mapped_body_link_pos_data = []
        mapped_body_link_rot_data = []
        body_names = [newton_utils.get_name_from_label(label) for label in self.robot_builder.body_label]
        for joint, mapping_data in retargeter_config["ik_match_table"].items():
            mapped_joints.append(joint)
            mapped_joint_indices.append(skeleton.joint_index(joint))
            mapped_body_link_pos_data.append((body_names.index(mapping_data['t_body']), mapping_data['t_weight']))
            mapped_body_link_rot_data.append((body_names.index(mapping_data['r_body']), mapping_data['r_weight']))

        return (
            mapped_joints,
            mapped_joint_indices,
            mapped_body_link_pos_data,
            mapped_body_link_rot_data)

    def _extract_initial_targets(self, num_envs: int, state):
        """Extract per-link initial target positions/rotations from body state."""
        body_q = state.body_q.numpy()
        num_pos = len(self.mapped_body_link_pos_data)
        num_rot = len(self.mapped_body_link_rot_data)
        pos_targets = np.zeros((num_envs, num_pos), dtype=wp.vec3)
        rot_targets = np.zeros((num_envs, num_rot), dtype=wp.quat)

        for env in range(num_envs):
            base = env * self.num_body_count
            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_pos_data):
                pos_targets[env, ee_idx] = body_q[base + link_idx][0:3]
            for ee_idx, (link_idx, _) in enumerate(self.mapped_body_link_rot_data):
                rot_targets[env, ee_idx] = wp.normalize(wp.quat(body_q[base + link_idx][3:7]))

        return (
            [pos_targets[:, i] for i in range(num_pos)],
            [rot_targets[:, i] for i in range(num_rot)],
        )

    def _create_ik_objectives(self, num_envs, model, state):
        """Build IK position, rotation, joint-limit, and smooth-filter objectives.

        Args:
            num_envs: Number of parallel environments.
            model: Finalized Newton model for the current batch.
            state: Initial model state used to seed target positions and rotations.

        Returns:
            A tuple of ``(position_objectives, rotation_objectives,
            joint_limit_objective, smooth_joint_filter_objective)``.
        """
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pos_targets, rot_targets = self._extract_initial_targets(num_envs, state)

        position_objectives = IKSolver.make_position_objectives(
            num_envs, self.mapped_body_link_pos_data, pos_targets)
        rotation_objectives = IKSolver.make_rotation_objectives(
            num_envs, self.mapped_body_link_rot_data, rot_targets)
        joint_limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=self.joint_limit_weight)
        smooth_joint_filter_objective = IKSolver.make_smooth_joint_filter(
            self.ik_model,
            weight=0.0,
            coord_masks=self.smooth_joint_filter_coord_masks,
            offset_limit_lower=self.smooth_joint_limit_offsets[0],
            offset_limit_upper=self.smooth_joint_limit_offsets[1])

        return position_objectives, rotation_objectives, joint_limit_objective, smooth_joint_filter_objective
