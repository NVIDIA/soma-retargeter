# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import newton

import pathlib
import time
import warp as wp

from soma_retargeter.io import utils
from soma_retargeter.robotics import robot_registry
import soma_retargeter.utils.math_utils as math_utils
import soma_retargeter.io.bvh as bvh_utils
import soma_retargeter.io.csv as csv_utils
import soma_retargeter.pipelines.utils as pipeline_utils

from soma_retargeter.utils import newton_utils
from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str

from tqdm import trange

_UI_NEWTON_PANEL_WIDTH  = 320
_UI_NEWTON_PANEL_MARGIN = 10
_UI_NEWTON_PANEL_ALPHA  = 0.9

def _ui_scale(ui) -> float:
    """Return the DPI scale factor for the current display."""
    return ui.get_font_size() / 13.0  # 13px is imgui's default base size
_DEFAULT_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)

class Viewer:
    def __init__(self, viewer, config):
        self.viewer = viewer
        self.viewer.vsync = True
        self.config = config
        self.converter = SpaceConverter(get_facing_direction_type_from_str(self.config['retarget_source_facing_direction']))
        self._error_message = None

        if isinstance(self.viewer, newton.viewer.ViewerNull):
            # Headless mode for batch processing
            return

        self.fps      = 60
        self.frame_dt = 1.0 / self.fps
        self.time     = 0.0

        self.is_playing          = True
        self.playback_time       = 0.0
        self.playback_speed      = 1.0
        self.playback_loop       = True
        self.playback_total_time = 0.0

        retarget_target = config['retarget_target']
        self.retarget_target_options = robot_registry.list_available_soma_targets()
        if not self.retarget_target_options:
            raise ValueError("[ERROR]: No robots with a soma retargeter config found. "
                             "Use the Robot Configurator to set one up.")
        if retarget_target not in self.retarget_target_options:
            if retarget_target in robot_registry.list_available_targets():
                raise ValueError(f"[ERROR]: Robot '{retarget_target}' has no soma retargeter config. "
                                 f"Open the Robot Configurator and save configs for it first.")
            raise ValueError(f"[ERROR]: Unknown robot target: '{retarget_target}'. "
                             f"Configured targets: {', '.join(self.retarget_target_options)}")

        self.retarget_target = None # Will be set later by _reset_viewer_model()
        self.retarget_target_idx = self.retarget_target_options.index(retarget_target)

        self.retarget_source_options = ['soma']
        self.retarget_solver_options = ['Newton']
        self.retarget_solver_idx     = 0
        self.retarget_source_idx     = 0

        self.show_skeleton_mesh = True
        self.show_skeleton = False
        self.show_skeleton_joint_axes = False
        self.show_gizmos = True

        self.viewer.renderer.set_title("BVH to CSV Converter")

        self.model = None
        self.state = None
        self.num_robots = 1
        self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
        self.coordinate_renderer = CoordinateRenderer()
        self.skeleton = None
        self.skeleton_renderer = None
        self.skeletal_mesh_renderer = None

        self.animation_offsets = []
        self.animation_buffers = []
        self.skeleton_instances = []
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]

    def _capture_camera_state(self):
        # Snapshot the camera pose to a plain dict so it can survive a model rebuild
        camera = self.viewer.camera
        return {
            "pos": [float(camera.pos[i]) for i in range(3)],
            "pivot": [float(camera.pivot[i]) for i in range(3)],
            "pitch": float(camera.pitch),
            "yaw": float(camera.yaw),
            "fov": float(camera.fov),
            "near": float(camera.near),
            "far": float(camera.far),
        }

    def _apply_camera_state(self, camera_state):
        # Restore the camera pose from a previously captured state; no-op if None
        if camera_state is None:
            return

        camera = self.viewer.camera
        vec3_type = type(camera.pos)
        camera.pos = vec3_type(*camera_state["pos"])
        camera.pivot = vec3_type(*camera_state["pivot"])
        camera.pitch = camera_state["pitch"]
        camera.yaw = camera_state["yaw"]
        camera.fov = camera_state["fov"]
        camera.near = camera_state["near"]
        camera.far = camera_state["far"]

    def _refresh_robot_list(self):
        robot_registry.registry.reload()
        current = self.retarget_target_options[self.retarget_target_idx] if self.retarget_target_options else None
        names = robot_registry.list_available_soma_targets()
        self.retarget_target_options = names
        if current in names:
            self.retarget_target_idx = names.index(current)
        else:
            self.retarget_target_idx = 0

    def _reset_viewer_model(self):
        # Reset model based on the currently selected robot target
        selected_target = self.retarget_target_options[self.retarget_target_idx]
        if self.retarget_target == selected_target:
            return

        camera_state = self._capture_camera_state()
        self.retarget_target = selected_target
        self.robot_builder = pipeline_utils.create_robot_builder(self.retarget_target)
        self.actuated_joint_names = newton_utils.get_filtered_joint_names(
            self.robot_builder, [newton.JointType.REVOLUTE])

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.robot_shape_start = builder.shape_count
        self.robot_shape_count = self.robot_builder.shape_count
        for _ in range(self.num_robots):
            builder.add_world(self.robot_builder, wp.transform_identity())
        self.model = builder.finalize()
        end = self.robot_shape_start + self.robot_shape_count
        self.robot_base_shape_colors = self.model.shape_color.numpy()[self.robot_shape_start:end]
        self.robot_shape_min_channel = self.robot_base_shape_colors.min(axis=1)
        self.robot_shape_channel_spread = self.robot_base_shape_colors.max(axis=1) - self.robot_shape_min_channel

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self._apply_camera_state(camera_state)
        self.state = self.model.state()

        self.robot_num_joint_q = self.model.joint_coord_count // self.model.articulation_count
        self.robot_joint_q_offsets = [int(i * self.robot_num_joint_q) for i in range(self.model.articulation_count)]
        self.robot_default_joint_q_values = self.model.joint_q.numpy()
        self.robot_csv_animation_buffers = [None for _ in range(self.num_robots)]
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

    def gui(self, ui):
        self.ui_playback_controls(ui)
        self.ui_scene_options(ui)
        self._ui_error_popup(ui)

    def _ui_error_popup(self, ui):
        if self._error_message:
            ui.open_popup("##error_msg")
        result = ui.begin_popup_modal("##error_msg", None,
                                      ui.WindowFlags_.always_auto_resize |
                                      ui.WindowFlags_.no_title_bar)
        if not result[0]:
            return
        ui.text(self._error_message or "")
        ui.spacing()
        btn_w = 120
        avail = ui.get_content_region_avail().x
        ui.set_cursor_pos_x(ui.get_cursor_pos_x() + (avail - btn_w) / 2)
        if ui.button("OK", ui.ImVec2(btn_w, 0)):
            self._error_message = None
            ui.close_current_popup()
        ui.end_popup()

    def load_csv_file(self, path):
        self.robot_csv_animation_buffers[0] = csv_utils.load_csv(path)
        self.compute_playback_total_time()

    def load_bvh_file(self, path):
        self.animation_buffers = []
        self.skeleton_instances = []
        if self.skeleton_renderer is not None:
            self.skeleton_renderer.clear(self.viewer)
        if self.skeletal_mesh_renderer is not None:
            self.skeletal_mesh_renderer.clear(self.viewer)
        if self.coordinate_renderer is not None:
            self.coordinate_renderer.clear(self.viewer)

        self.skeleton, animation = bvh_utils.load_bvh(path)
        self.skeleton_renderer = SkeletonRenderer(self.skeleton, [0])
        self.skeleton_instances = [SkeletonInstance(self.skeleton, _DEFAULT_COLOR, self.converter.transform(wp.transform_identity()))]
        self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        self.animation_buffers = [animation]

        self.skeletal_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.skeleton)
        self.skeletal_mesh_renderer = SkeletalMeshRenderer(self.skeletal_mesh)
        self.compute_playback_total_time()

    def compute_playback_total_time(self):
        bvh_max_time = 0.0
        for buffer in self.animation_buffers:
            if buffer is not None:
                bvh_max_time = max(bvh_max_time, buffer.num_frames * (1 / buffer.sample_rate))

        csv_max_time = 0.0
        for buffer in self.robot_csv_animation_buffers:
            if buffer is not None:
                csv_max_time = max(csv_max_time, buffer.num_frames * (1 / buffer.sample_rate))

        self.playback_total_time = max(bvh_max_time, csv_max_time)
        self.playback_time = wp.clamp(self.playback_time, 0.0, self.playback_total_time)

    def update_robot_states(self):
        for i in range(self.num_robots):
            robot_offset = self.robot_offsets[i]

            joint_q_offset = self.robot_joint_q_offsets[i]
            if self.robot_csv_animation_buffers[i] is not None:
                buffer = self.robot_csv_animation_buffers[i]
                # Apply visual offset
                prev_xform = wp.transform(buffer.xform)
                buffer.xform = robot_offset

                data = buffer.sample(self.playback_time)
                wp.copy(self.model.joint_q, wp.array(data, dtype=wp.float32), joint_q_offset, 0, self.robot_num_joint_q)
                buffer.xform = prev_xform
            else:
                root_tx = wp.mul(
                    robot_offset,
                    wp.transform(*self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + 7)]))

                wp.copy(
                    self.model.joint_q,
                    wp.array(self.robot_default_joint_q_values[joint_q_offset:(joint_q_offset + self.robot_num_joint_q)], dtype=wp.float32),
                    joint_q_offset,
                    0, self.robot_num_joint_q)
                wp.copy(self.model.joint_q, wp.array(root_tx[0:7], dtype=wp.float32), joint_q_offset, 0, 7)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)

    def step(self):
        self._reset_viewer_model()
        self.time += self.frame_dt
        if self.is_playing:
            self.playback_time += self.frame_dt * self.playback_speed
            if self.playback_loop and self.playback_total_time > 0.0:
                self.playback_time %= self.playback_total_time
            else:
                self.playback_time = max(0.0, min(self.playback_time, self.playback_total_time))

        for i in range(len(self.animation_buffers)):
            self.skeleton_instances[i].set_local_transforms(self.animation_buffers[i].sample(self.playback_time))

        def clamp_gizmo_transform(tx: wp.transform):
            return wp.transform(
                wp.vec3(tx.p[0], tx.p[1], 0.0),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        for i in range(len(self.robot_offsets)):
            self.robot_offsets[i] = clamp_gizmo_transform(self.robot_offsets[i])
        for i in range(len(self.animation_offsets)):
            self.animation_offsets[i] = clamp_gizmo_transform(self.animation_offsets[i])

        self.update_robot_states()

    def render(self):
        self.viewer.begin_frame(self.time)
        if len(self.animation_buffers) > 0:
            for i in range(len(self.skeleton_instances)):
                prev_xform = wp.transform(self.skeleton_instances[i].xform)
                self.skeleton_instances[i].xform = wp.mul(self.animation_offsets[i], self.skeleton_instances[i].xform)
                if self.show_skeleton:
                    self.skeleton_renderer.draw(self.viewer, self.skeleton_instances[i], i)
                if self.show_skeleton_joint_axes:
                    tx = self.skeleton_instances[i].compute_global_transforms()
                    self.coordinate_renderer.draw(self.viewer, tx, 0.1, i)
                if self.show_skeleton_mesh:
                    self.skeletal_mesh_renderer.draw(self.viewer, self.skeleton_instances[i], self.skeleton_instances[i].color, i)
                self.skeleton_instances[i].xform = prev_xform

        if self.show_gizmos:
            for i, offset in enumerate(self.robot_offsets):
                self.viewer.log_gizmo(f"robot_offset{i}", offset)
            for i, offset in enumerate(self.animation_offsets):
                self.viewer.log_gizmo(f"animation_offset{i}", offset)

        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def run(self):
        while self.viewer.is_running():
            with wp.ScopedTimer("step", active=False):
                self.step()
            with wp.ScopedTimer("render", active=False):
                self.render()

        self.viewer.close()

    def retarget_motion(self):
        retarget_source = self.retarget_source_options[self.retarget_source_idx]
        retarget_target = self.retarget_target_options[self.retarget_target_idx]
        retarget_solver = self.retarget_solver_options[self.retarget_solver_idx]

        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.soma_retargeting_pipeline as soma_retargeting_pipeline
            try:
                pipeline = soma_retargeting_pipeline.SomaRetargetingPipeline(self.skeleton, retarget_source, retarget_target)
            except Exception as error:
                message = f"[ERROR]: {error}"
                print(message)
                self._error_message = message
                self.robot_csv_animation_buffers[0] = None
                self.compute_playback_total_time()
                return
        else:
            raise(ValueError(f"[ERROR]: Unknown retargeter solver [{retarget_solver}"))

        r_offsets = [wp.transform(wp.vec3(0,0,0), wp.quat(*s.xform[3:7])) for s in self.skeleton_instances]
        pipeline.add_input_motions(self.animation_buffers, r_offsets, True)
        buffers = pipeline.execute()

        if buffers is not None:
            t_offsets = [wp.transform(wp.vec3(*s.xform[:3]), wp.quat_identity()) for s in self.skeleton_instances]
            for i, buffer in enumerate(buffers):
                buffer.xform = t_offsets[i]

        self.robot_csv_animation_buffers[0] = buffers[0]

    def ui_scene_options(self, ui):
        import tkinter as tk
        from tkinter import filedialog as tk_filedialog

        viewport = ui.get_main_viewport()

        _s = _ui_scale(ui)
        _panel_w = int(_UI_NEWTON_PANEL_WIDTH * _s)
        _margin  = int(_UI_NEWTON_PANEL_MARGIN * _s)
        panel_size = ui.ImVec2(_panel_w, int(450 * _s))
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.size.x - _margin - panel_size.x,
                viewport.size.y - _margin - panel_size.y))

        ui.set_next_window_size(panel_size)
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Scene Options", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        ui.separator()

        ui.text("Robot Options")
        ui.set_next_item_width(-60)
        _, self.retarget_target_idx = ui.combo("##target", self.retarget_target_idx, self.retarget_target_options)
        ui.same_line()
        if ui.button("Reload"):
            self._refresh_robot_list()

        # Motion options
        if ui.collapsing_header("Motion", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()
            ui.align_text_to_frame_padding()
            ui.text("BVH Motion:")
            ui.same_line()

            ui.push_id(100)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                bvh_path = tk_filedialog.askopenfilename(
                    title='Load BVH File',
                    defaultextension=".bvh",
                    filetypes=[('BVH files', '*.bvh')])

                if bvh_path:
                    self.load_bvh_file(bvh_path)
            ui.pop_id()

            if (len(self.animation_buffers) == 0):
                ui.begin_disabled()

            ui.same_line()
            if ui.button("Retarget"):
                self.retarget_motion()

            if (len(self.animation_buffers) == 0):
                ui.end_disabled()

            ui.align_text_to_frame_padding()
            ui.text("CSV Motion:")
            ui.same_line()

            ui.push_id(200)
            if ui.button("Load"):
                root = tk.Tk()
                root.withdraw()
                csv_path = tk_filedialog.askopenfilename(
                    title='Load CSV File',
                    defaultextension=".csv",
                    filetypes=[('CSV files', '*.csv')])

                if csv_path:
                    self.load_csv_file(csv_path)

            if self.robot_csv_animation_buffers[0] is None:
                ui.begin_disabled()
            ui.pop_id()

            ui.same_line()
            if ui.button("Save"):
                root = tk.Tk()
                root.withdraw()

                save_path = tk_filedialog.asksaveasfilename(
                    title="Save CSV File",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_path:
                    csv_utils.save_csv(save_path, self.actuated_joint_names, self.robot_csv_animation_buffers[0])

            if self.robot_csv_animation_buffers[0] is None:
                ui.end_disabled()

        # Visibility options
        ui.spacing()
        if ui.collapsing_header("Visibility", flags=ui.TreeNodeFlags_.default_open):
            ui.separator()

            changed, self.show_skeleton_mesh = ui.checkbox("Show Mesh", self.show_skeleton_mesh)
            if changed and self.skeletal_mesh_renderer is not None:
                self.skeletal_mesh_renderer.clear(self.viewer)
            changed, self.show_skeleton = ui.checkbox("Show Skeleton", self.show_skeleton)
            if changed and self.skeleton_renderer is not None:
                self.skeleton_renderer.clear(self.viewer)
            changed, self.show_skeleton_joint_axes = ui.checkbox("Show Joint Axes", self.show_skeleton_joint_axes)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            _, self.show_gizmos = ui.checkbox("Show Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset"):
                self.robot_offsets = [wp.transform(wp.vec3(0.0, i - (self.num_robots - 1) / 2.0, 0.0), wp.quat_identity()) for i in range(self.num_robots)]
                self.animation_offsets = [wp.transform_identity()] * len(self.skeleton_instances)
        ui.end()

    def ui_playback_controls(self, ui):
        viewport = ui.get_main_viewport()

        _s = _ui_scale(ui)
        _panel_w = int(_UI_NEWTON_PANEL_WIDTH * _s)
        _margin  = int(_UI_NEWTON_PANEL_MARGIN * _s)
        panel_height = int(105 * _s)
        panel_width = viewport.size.x - 2 * (2 * _margin + _panel_w)

        ui.set_next_window_pos(ui.ImVec2(_panel_w + _margin, viewport.size.y - _margin - panel_height))
        ui.set_next_window_size(ui.ImVec2(panel_width, panel_height))
        ui.set_next_window_bg_alpha(_UI_NEWTON_PANEL_ALPHA)

        ui.begin("Playback Controls", flags=(ui.WindowFlags_.no_collapse | ui.WindowFlags_.no_resize))
        # Time slider
        ui.align_text_to_frame_padding()
        ui.text("Time (s):")
        ui.same_line()
        ui.set_next_item_width(panel_width - int(150 * _s))
        changed, new_time = ui.slider_float(
            "##TimeSlider",
            self.playback_time,
            0.0,
            self.playback_total_time,
            "%.2f")
        if changed:
            self.playback_time = wp.clamp(new_time, 0.0, self.playback_total_time)
        ui.same_line()
        ui.text_colored(ui.ImVec4(0.6, 0.8, 1.0, 1.0), f"{self.playback_total_time:.2f}s")

        self.is_playing = not ui.button("Pause") if self.is_playing else ui.button("Play ")
        ui.same_line()

        # Speed slider
        ui.align_text_to_frame_padding()
        ui.text("Speed")
        ui.same_line()
        ui.set_next_item_width(int(100 * _s))
        changed, new_speed = ui.slider_float(
            "##SpeedSlider",
            self.playback_speed,
            -2.0, 2.0,
            "%.2f"
        )
        if changed:
            self.playback_speed = new_speed
        ui.same_line()
        _, self.playback_loop = ui.checkbox("Loop", self.playback_loop)
        ui.end()

    def batched_retargeting(self):
        if not os.path.isdir(self.config['import_folder']):
            print(f"[ERROR]: Import folder does not exist {self.config['import_folder']}.")
            exit(-1)

        import_path = pathlib.Path(self.config['import_folder'])
        if len(self.config['export_folder']) == 0:
            print("[ERROR]: No export folder specified.")
            exit(-1)

        export_path = pathlib.Path(self.config['export_folder'])
        if not export_path.is_dir():
            print(f"[WARNING]: Export folder does not exist! Creating new folder at {str(export_path)}!")
            export_path.mkdir(parents=True, exist_ok=True)

        batch_size = self.config['batch_size']
        bvh_files = list(import_path.rglob("*.bvh"))
        if (len(bvh_files) == 0):
            print(f"[ERROR]: Import folder {str(import_path)}, does not contain any BVH files.")
            exit(-1)

        # Sort files based on size (largest first)
        bvh_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        batches = [bvh_files[i:i + batch_size] for i in range(0, len(bvh_files), batch_size)]

        # All skeletons should be the same, load one as our reference
        bvh_importer = bvh_utils.BVHImporter()
        bvh_skeleton, _ = bvh_importer.create_skeleton(batches[0][0])

        bvh_tx_converter = self.converter.transform(wp.transform_identity())
        expected_num_joints = bvh_skeleton.num_joints

        retarget_source = self.config['retarget_source']
        retarget_solver = self.config['retargeter']
        retarget_target = self.config["retarget_target"]
        retarget_pipeline = None
        if (retarget_solver == 'Newton'):
            import soma_retargeter.pipelines.soma_retargeting_pipeline as soma_retargeting_pipeline
            try:
                retarget_pipeline = soma_retargeting_pipeline.SomaRetargetingPipeline(
                    bvh_skeleton, retarget_source, retarget_target)
            except Exception as error:
                print(f"[ERROR]: {error}")
                exit(-1)
            actuated_joint_names = newton_utils.get_filtered_joint_names(
                retarget_pipeline.robot_builder, [newton.JointType.REVOLUTE])
        if retarget_pipeline is None:
            print(f"[ERROR]: Invalid retarget solver selected [{retarget_solver}]. Use 'Newton'.")
            exit(-1)

        nb_retargeted_motions = 0
        start_time = time.time()

        for i, batch in enumerate(batches):
            print(f"[INFO]: Processing batch {i+1} of {len(batches)}")

            print(f"[INFO]: Loading {len(batch)} animations...")
            animations = []
            for file_path in batch:
                _, animation = bvh_utils.load_bvh(file_path, bvh_skeleton)
                # All animations should be on the same skeleton
                assert expected_num_joints == animation.skeleton.num_joints, (
                    f"[ERROR]: Unexpected number of joints in input motion. Expected {expected_num_joints}, "
                    f"got {animation.skeleton.num_joints}")

                animations.append(animation)
            assert(len(animations) == len(batch))

            if (len(animations) > 0):
                print("[INFO]: Retargeting...")
                retarget_pipeline.clear()
                retarget_pipeline.add_input_motions(animations, [bvh_tx_converter] * len(animations), True)
                csv_buffers = retarget_pipeline.execute()

                assert(len(csv_buffers) == len(animations))
                for i in trange(len(csv_buffers), desc="[INFO]: Exporting CSV Files"):
                    csv_buffer = csv_buffers[i]
                    dst_path = export_path / pathlib.Path(batch[i]).relative_to(import_path).with_suffix(".csv")
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    csv_utils.save_csv(dst_path, actuated_joint_names, csv_buffer)

            nb_retargeted_motions += len(batch)

        elapsed_time = time.time() - start_time
        elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
        print(
            f"[INFO]: Retargeted {nb_retargeted_motions} animations successfully "
            f"in {elapsed_str} "
            f"[{(elapsed_time/nb_retargeted_motions):.2f}s per motion]!")

def main():
    import newton.examples

    parser = newton.examples.create_parser()
    parser.set_defaults(viewer=("null"))
    parser.add_argument(
        "--config",
        type=lambda x: None if x == "None" else str(x),
        default="./assets/default_bvh_to_csv_converter_config.json",
        help="Input json config file.")

    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Main config json file not found: {args.config}")
        exit(1)

    if args.viewer != "null" and hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()

    config = utils.load_json(args.config)
    custom_paths = config.get('extra_robot_paths', [])
    if isinstance(custom_paths, str):
        custom_paths = [custom_paths]
    for path in custom_paths:
        robot_registry.register_robots_path(path)

    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        if not isinstance(viewer, newton.viewer.ViewerNull):
            app.run()
        else:
            app.batched_retargeting()

if __name__ == "__main__":
    main()
