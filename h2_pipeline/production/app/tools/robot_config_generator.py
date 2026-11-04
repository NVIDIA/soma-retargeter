# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import re
import shutil
import defusedxml.ElementTree as ET
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import newton
import warp as wp

import soma_retargeter.io.bvh as bvh_utils
import soma_retargeter.pipelines.utils as pipeline_utils
import soma_retargeter.utils.newton_utils as newton_utils
import soma_retargeter.utils.math_utils as math_utils

from soma_retargeter.io.utils import load_json, save_json
from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

from soma_retargeter.robotics.robot_registry import list_available_targets
from soma_retargeter.robotics.robot_registry import registry as robot_registry

from soma_retargeter.renderers.skeleton_renderer import SkeletonRenderer
from soma_retargeter.renderers.mesh_renderer import SkeletalMeshRenderer
from soma_retargeter.renderers.coordinate_renderer import CoordinateRenderer
from soma_retargeter.animation.skeleton import SkeletonInstance


_UI_PANEL_WIDTH  = 500
_UI_PANEL_MARGIN = 10
_UI_PANEL_ALPHA  = 0.9

def _ui_scale(ui) -> float:
    """Return the DPI scale factor for the current display."""
    return ui.get_font_size() / 13.0  # 13px is imgui's default base size

_SOMA_COLOR = (235.0 / 255.0, 245.0 / 255.0, 112.0 / 255.0)

_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def _mirror_link_name(link: str) -> str:
    """Swap left↔right in *link* preserving token case. Returns '' if no side token is found."""
    def _replace(m):
        w = m.group(0)
        r = "right" if w.lower() == "left" else "left"
        if w.isupper():   return r.upper()
        if w[0].isupper(): return r.capitalize()
        return r
    result = re.sub(r'(?<![a-zA-Z])(?:left|right)(?![a-zA-Z])', _replace, link, flags=re.IGNORECASE)
    return result if result != link else ""

def _is_safe_path_component(value: str) -> bool:
    """Return True when *value* can safely be used as one path component."""
    return bool(value) and value not in (".", "..") and bool(_SAFE_PATH_COMPONENT_RE.fullmatch(value))

def _sanitize_path_component(value: str, fallback: str) -> str:
    """Convert arbitrary text into a conservative single path component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return cleaned if _is_safe_path_component(cleaned) else fallback

def _is_relative_to(path: pathlib.Path, parent: pathlib.Path) -> bool:
    """Return True when resolved *path* is contained by resolved *parent*."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def _compute_local_offset(source_tx: wp.transform, target_tx: wp.transform) -> wp.transform:
    """Compute the local offset transform that maps *source_tx* to *target_tx*.

    The returned transform encodes the rotational difference between source and target, and the
    translation of target expressed in target's local frame.
    """
    offset_q = wp.normalize(wp.mul(wp.quat_inverse(source_tx.q), target_tx.q))
    offset_p = wp.quat_rotate(wp.quat_inverse(target_tx.q), target_tx.p - source_tx.p)
    return wp.transform(offset_p, offset_q)

def _apply_local_offset(source_tx: wp.transform, offset_tx: wp.transform) -> wp.transform:
    """Apply a precomputed local offset to *source_tx* and return the resulting world transform.

    Inverse of :func:`_compute_local_offset`: given *source_tx* and the stored *offset_tx*,
    reconstructs the original target world transform (up to floating-point error).
    """
    q = wp.mul(source_tx.q, offset_tx.q)
    return wp.transform(source_tx.p + wp.quat_rotate(q, offset_tx.p), q)

def _parse_robot_name(path: str) -> str:
    """Parse the robot/model name from a URDF or MJCF file in one XML parse pass.

    Checks the ``name`` attribute (URDF) then ``model`` attribute (MJCF) on the
    root element. Returns ``"no_name"`` if neither is present or parsing fails.
    """
    try:
        root = ET.parse(path).getroot()
        return root.get("name", "") or root.get("model", "") or ""
    except Exception:
        return "no_name"

# ------------------------------------------------------------------
# Message Dialogs
# ------------------------------------------------------------------

class MsgButtons(Enum):
    OK     = auto()
    YES_NO = auto()

@dataclass
class _PendingMessage:
    text:     str
    buttons:  MsgButtons = MsgButtons.OK
    callback: Optional[Callable[[bool], None]] = field(default=None, repr=False)

# ------------------------------------------------------------------
# Main Viewer
# ------------------------------------------------------------------
class Viewer:
    """Interactive Newton-based viewer for registering and configuring robots for SOMA retargeting.

    Provides a GUI to:
    - Load any robot registered in the robot registry (URDF or MJCF).
    - Edit DOF zero-pose via sliders.
    - Map SOMA skeleton joints to robot body links and compute scale / offset transforms.
    - Configure per-link smooth filter weights.
    - Save the resulting scaler, retargeter, and post-processing config files.
    """

    def __init__(self, viewer, config):
        """Initialize the viewer and load the first registered robot, if any.

        Args:
            viewer: Newton viewer instance (e.g. from ``newton.examples.init``).
            config: Dict loaded from the robot_config_generator config JSON.
        """
        self.viewer = viewer
        self.config = config
        self.assets_root = pathlib.Path(__file__).resolve().parent.parent.parent / config["paths"]["assets_root"]
        self.viewer.vsync = False
        self.viewer.renderer.set_title("Robot Configurator")

        self.fps      = 120
        self.frame_dt = 1.0 / self.fps
        self.time     = 0.0

        # Newton model/state for the currently loaded robot
        self.robot_builder   = None
        self.model           = None
        self.state           = None
        self.current_robot   = ""
        self.default_joint_q = None

        # Mapping state (populated when a robot loads)
        self.body_link_names        = []
        self.mapping_body_idx       = {}
        self.mapping_weights        = {}
        self.retargeter_config      = {}
        self.retargeter_config_path = None

        # Highlight state — hovered link in Mapping tab
        self.hovered_link       = ""
        self._prev_hovered_link = ""
        self.base_shape_colors  = None
        self.shape_body_np      = None
        self.body_name_to_model_idx = {}

        # Flash highlight: link name + expiry time (seconds)
        self._flash_link  = ""
        self._flash_until = 0.0

        # DOF editor state
        self.dof_joints               = []   # list of (display_name, q_idx, lo, hi)
        self.current_joint_q          = None
        self.loaded_reference_joint_q = None  # reference pose from scaler config (or robot default)

        # Remapped coordinate axes
        self.coordinate_renderer    = CoordinateRenderer()
        self.show_remapped_coords   = True
        self.remapped_coords_blend  = 1.0

        # SOMA reference skeleton + mesh
        self.soma_skeleton          = None
        self.soma_instance          = None
        self.soma_skeleton_renderer = None
        self.soma_mesh_renderer     = None
        self.show_soma_mesh         = False
        self.show_soma_skeleton     = True

        # Scale & offset state
        self.human_scale_ratio        = 1.0
        self._saved_scale             = 1.0   # default/loaded scale shown as yellow marker
        self._scale_set_to_idx        = 0     # 0=Bounding Box, 1=Pelvis
        self.human_offset_transforms  = {}  # soma_joint → wp.transform
        self.human_offset_eulers_deg  = {}  # soma_joint → wp.vec3 (degrees)

        self._load_soma()

        # Per-joint offset editing
        self.selected_mapping_joint = ""
        self.edit_offset_tx         = wp.transform_identity()

        # Smooth filter masks
        self.smooth_filter_masks   = {}  # link_name → [v0, v1, v2]
        self.smooth_filter_add_idx = 0

        # Robot visibility
        self.show_robot = True

        # Gizmo offsets (XY translation + Z rotation only)
        self.robot_offset = wp.transform_identity()
        self.soma_offset  = wp.transform_identity()
        self.show_gizmos  = False

        # Robot dropdown
        self.robot_list = []
        self.robot_idx  = 0
        self._refresh_robot_list()

        # Add-Robot popup state
        self._pending_popup     = False
        self._popup_vendor      = ""
        self._popup_name        = ""

        # Message dialog queue
        self._message_queue: list[_PendingMessage] = []
        self._popup_src_path    = ""
        self._popup_overwrite   = False

        # Load the first registered robot on startup; otherwise just register the UI
        if self.robot_list and not self.robot_list[0].startswith("("):
            self._load_robot(self.robot_list[0])
        else:
            self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main loop: step simulation and render until the viewer window is closed."""
        while self.viewer.is_running():
            self.step()
            self.render()
        self.viewer.close()

    def _sync_robot_fk(self) -> bool:
        """Copy the current editable joint pose into the model and refresh FK state."""
        if (self.model is None or self.state is None or
                self.current_joint_q is None or self.default_joint_q is None):
            return False

        joint_q = self.current_joint_q.copy()
        root_tx = wp.mul(self.robot_offset, wp.transform(*self.default_joint_q[:7]))
        joint_q[:7] = list(root_tx)
        wp.copy(self.model.joint_q, wp.array(joint_q, dtype=wp.float32))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state, None)
        return True

    def step(self):
        """Advance simulation by one frame: clamp gizmos, sync edit gizmo offset, run FK."""
        self.time += self.frame_dt

        def _clamp_gizmo(tx: wp.transform) -> wp.transform:
            return wp.transform(
                wp.vec3(tx.p[0], tx.p[1], 0.0),
                math_utils.quat_twist(wp.vec3(0.0, 0.0, 1.0), tx.q))

        self.robot_offset = _clamp_gizmo(self.robot_offset)
        self.soma_offset  = _clamp_gizmo(self.soma_offset)

        # Recompute local offset from the edit gizmo position if joint is selected to edit
        if self.selected_mapping_joint and self.soma_instance is not None:
            joint = self.selected_mapping_joint
            soma_joint_idx = self.soma_skeleton.joint_index(joint)
            if soma_joint_idx >= 0:
                prev_xform = wp.transform(self.soma_instance.xform)
                self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
                np_soma_tx = self.soma_instance.compute_global_transforms()
                self.soma_instance.xform = prev_xform
                soma_tx = wp.transform(*np_soma_tx[soma_joint_idx])
                self.human_offset_transforms[joint] = _compute_local_offset(
                    soma_tx, self.edit_offset_tx)
                self._sync_euler_cache(joint)

        if self.model is not None and self.state is not None and self.current_joint_q is not None:
            if self._flash_link and self.time >= self._flash_until:
                self._flash_link = ""
                self._update_highlight_colors()
            elif self.hovered_link != self._prev_hovered_link:
                self._update_highlight_colors()
                self._prev_hovered_link = self.hovered_link
            self._sync_robot_fk()

    def render(self):
        """Draw one frame: SOMA skeleton/mesh, remapped coordinate axes, gizmos, and robot state."""
        self.viewer.begin_frame(self.time)
        if self.soma_instance is not None:
            self.soma_instance.scale = self.human_scale_ratio
            prev_xform = wp.transform(self.soma_instance.xform)
            self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
            if self.show_soma_skeleton and self.soma_skeleton_renderer is not None:
                self.soma_skeleton_renderer.draw(self.viewer, self.soma_instance, 0)
            if self.show_soma_mesh and self.soma_mesh_renderer is not None:
                self.soma_mesh_renderer.draw(self.viewer, self.soma_instance, _SOMA_COLOR, 0)
            self.soma_instance.xform = prev_xform
        if self.show_remapped_coords and self.human_offset_transforms:
            self._render_remapped_coordinates()
        if self.show_gizmos:
            self.viewer.log_gizmo("robot_offset", self.robot_offset)
            if self.soma_instance is not None:
                self.viewer.log_gizmo("soma_offset", self.soma_offset)
        if self.selected_mapping_joint:
            self.viewer.log_gizmo("edit_offset", self.edit_offset_tx)
        if self.state is not None:
            self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def _apply_robot_visibility(self):
        """Show or hide the robot world in the viewport."""
        # Temporary GL workaround: clear stale instancers so their capacity resizes properly.
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            for name in list(self.viewer.objects):
                if not name.startswith("/model/shapes/"):
                    continue
                obj = self.viewer.objects.pop(name)
                if hasattr(obj, "destroy"):
                    obj.destroy()
        self.viewer.set_visible_worlds([0] if self.show_robot else [])

    def _update_highlight_colors(self):
        """Recolor robot shapes so the active link (flash or hover) is shown in orange."""
        if self.base_shape_colors is None:
            return
        colors = self.base_shape_colors.copy()
        active_link = self._flash_link or self.hovered_link
        if active_link:
            body_idx = self.body_name_to_model_idx.get(active_link, -1)
            if body_idx >= 0:
                colors[self.shape_body_np == body_idx] = [1.0, 0.6, 0.0]
        try:
            wp.copy(self.model.shape_color, wp.array(colors, dtype=wp.vec3))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def gui(self, ui):
        """Build the ImGui panel: robot selector, scale controls, tab bar, and Add-Robot popup."""
        viewport = ui.get_main_viewport()
        _s = _ui_scale(ui)
        _panel_w  = int(_UI_PANEL_WIDTH  * _s)
        _margin   = int(_UI_PANEL_MARGIN * _s)
        fps_panel_height = 290
        panel_height = viewport.size.y - fps_panel_height - _margin * 2

        ui.set_next_window_pos(ui.ImVec2(
            viewport.size.x - _margin - _panel_w,
            fps_panel_height + _margin),
            cond=ui.Cond_.once)
        ui.set_next_window_size(ui.ImVec2(_panel_w, panel_height),
                                cond=ui.Cond_.once)
        ui.set_next_window_bg_alpha(_UI_PANEL_ALPHA)

        ui.begin("Robot Configurator", flags=ui.WindowFlags_.no_collapse)

        ui.separator()
        ui.text("Registered Robots")

        combo_width = _panel_w - int(300 * _s)
        ui.set_next_item_width(combo_width)
        changed, self.robot_idx = ui.combo("##robot", self.robot_idx, self.robot_list)
        if changed and self.robot_list:
            self._load_robot(self.robot_list[self.robot_idx])

        ui.same_line()
        if ui.button("Add Robot..."):
            self._browse_and_prepare_popup()
        ui.same_line()
        if ui.button("Save Configs"):
            self._save_configs()

        if self.current_robot:
            ui.separator()
            ui.text(f"Robot: {self.current_robot}  "
                    f"DOFs: {self.model.joint_dof_count}  "
                    f"Links: {len(self.body_link_names)}")

        # Scale + Visibility (only when a robot is loaded)
        if self.model is not None:
            ui.separator()
            ui.push_id("scale_map")
            ui.align_text_to_frame_padding()
            ui.text("Scale:")
            ui.same_line()
            _slider_x = ui.get_cursor_pos_x()
            ui.set_next_item_width(-80)
            changed, self.human_scale_ratio = ui.slider_float("##scale", self.human_scale_ratio, 0.1, 5.0, "%.4f")
            if changed:
                self._apply_scale(self.human_scale_ratio)
            # Yellow marker at the saved/loaded scale value
            _lo, _hi = 0.1, 5.0
            _t = (max(_lo, min(_hi, self._saved_scale)) - _lo) / (_hi - _lo)
            _rmin = ui.get_item_rect_min()
            _rmax = ui.get_item_rect_max()
            _mx = _rmin.x + _t * (_rmax.x - _rmin.x)
            _dl = ui.get_window_draw_list()
            _yellow = ui.color_convert_float4_to_u32(ui.ImVec4(1.0, 0.85, 0.0, 0.9))
            _dl.add_line(ui.ImVec2(_mx, _rmin.y + 1), ui.ImVec2(_mx, _rmax.y - 1), _yellow, 2.0)
            ui.same_line()
            if ui.button("Reset##scale"):
                self._apply_scale(self._saved_scale)

            _set_to_options = ["Bounding Box", "Pelvis"]
            ui.set_cursor_pos_x(_slider_x)
            if ui.button("Set To...##scale"):
                if self._scale_set_to_idx == 0:
                    ratio = self._default_scale_from_bbox()
                    if ratio is None:
                        self.show_message("Cannot compute Bounding Box scale: robot or SOMA data is unavailable.")
                    else:
                        self._apply_scale(ratio)
                else:
                    ratio = self._default_scale_from_pelvis()
                    if ratio is None:
                        self.show_message("Cannot compute Pelvis scale: Hips joint is not mapped or SOMA/robot data is unavailable.")
                    else:
                        self._apply_scale(ratio)
            ui.same_line()
            ui.set_next_item_width(120)
            _, self._scale_set_to_idx = ui.combo("##scale_set_to_mode", self._scale_set_to_idx, _set_to_options)
            ui.pop_id()

            ui.separator()
            ui.push_id("soma_vis")
            # Row 1: mesh / robot / skeleton checkboxes
            changed, self.show_soma_mesh = ui.checkbox("SOMA Mesh", self.show_soma_mesh)
            if changed and self.soma_mesh_renderer is not None:
                self.soma_mesh_renderer.clear(self.viewer)
            ui.same_line()
            changed, self.show_robot = ui.checkbox("Robot", self.show_robot)
            if changed:
                self._apply_robot_visibility()
            ui.same_line()
            changed, self.show_soma_skeleton = ui.checkbox("Skeleton", self.show_soma_skeleton)
            if changed and self.soma_skeleton_renderer is not None:
                self.soma_skeleton_renderer.clear(self.viewer)
            ui.same_line()
            _, self.show_gizmos = ui.checkbox("Root Gizmos", self.show_gizmos)
            ui.same_line()
            if ui.button("Reset Gizmos"):
                self.robot_offset = wp.transform_identity()
                self.soma_offset  = wp.transform_identity()
            # Row 2: SOMA→Robot map + blend slider
            changed, self.show_remapped_coords = ui.checkbox("SOMA to Robot Map", self.show_remapped_coords)
            if changed and self.coordinate_renderer is not None:
                self.coordinate_renderer.clear(self.viewer)
            ui.same_line()
            ui.set_next_item_width(-1)
            _, self.remapped_coords_blend = ui.slider_float(
                "##blend", self.remapped_coords_blend, 0.0, 1.0, "blend %.2f")
            ui.pop_id()

        # Tabs (only when a robot is loaded)
        if self.model is not None:
            ui.separator()
            if ui.begin_tab_bar("##tabs"):
                if ui.begin_tab_item_simple("Zero Pose"):
                    self._ui_dof_editor_tab(ui)
                    ui.end_tab_item()
                if ui.begin_tab_item_simple("Mapping"):
                    self._ui_mapping_tab(ui)
                    ui.end_tab_item()
                if ui.begin_tab_item_simple("Smooth Filter"):
                    self._ui_smooth_filter_tab(ui)
                    ui.end_tab_item()
                ui.end_tab_bar()

        # Popup triggers
        if self._pending_popup:
            ui.open_popup("Add Robot##dlg")
            self._pending_popup = False
        if self._message_queue:
            ui.open_popup("##msg")

        self._ui_add_robot_popup(ui)
        self._ui_message_popup(ui)

        ui.end()

    # ------------------------------------------------------------------
    # Mapping tab
    # ------------------------------------------------------------------

    def _ui_mapping_tab(self, ui):
        """Render the Mapping tab: SOMA joint → robot link table with per-joint offset display."""
        ui.push_id("mapping_tab")
        ui.spacing()

        table_flags = (ui.TableFlags_.borders_inner_v |
                       ui.TableFlags_.row_bg |
                       ui.TableFlags_.sizing_fixed_fit |
                       ui.TableFlags_.scroll_y)

        COL_EDIT_W       = 22   # width of the checkbox column
        COL_JOINT_W      = 100  # width of the SOMA joint name column
        OFFSET_PANEL_H   = 92   # height of the per-joint offset editor shown below the table
        TABLE_BOTTOM_PAD = 4    # extra gap between table bottom and window edge

        avail        = ui.get_content_region_avail()
        has_sel      = bool(self.selected_mapping_joint)
        bottom_h     = OFFSET_PANEL_H if has_sel else 0
        table_height = avail.y - bottom_h - TABLE_BOTTOM_PAD

        if ui.begin_table("##mapping_table", 3, table_flags, ui.ImVec2(0, table_height)):
            ui.table_setup_column("Edit",       ui.TableColumnFlags_.width_fixed,   COL_EDIT_W)
            ui.table_setup_column("SOMA Joint", ui.TableColumnFlags_.width_fixed,   COL_JOINT_W)
            ui.table_setup_column("Body Link",  ui.TableColumnFlags_.width_stretch)
            ui.table_setup_scroll_freeze(0, 1)
            ui.table_headers_row()

            # Links already chosen by other joints (exclude "(none)")
            used_links = {
                self.body_link_names[self.mapping_body_idx[j]]
                for j in self.config["soma_joints"]
                if self.body_link_names[self.mapping_body_idx[j]] != "(none)"
            }

            hips_mapped = self.body_link_names[self.mapping_body_idx.get("Hips", 0)] != "(none)"

            new_hovered   = ""
            combo_hovered = ""
            for i, joint in enumerate(self.config["soma_joints"]):
                ui.table_next_row()
                current_link = self.body_link_names[self.mapping_body_idx[joint]]
                can_edit = hips_mapped

                ui.table_set_column_index(0)
                is_sel = self.selected_mapping_joint == joint
                if not can_edit and is_sel:
                    self._select_mapping_joint("")
                    is_sel = False
                if not can_edit:
                    ui.begin_disabled()
                chk_changed, new_sel = ui.checkbox(f"##chk{i}", is_sel)
                if not can_edit:
                    ui.end_disabled()
                    if ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled):
                        ui.set_tooltip("Map Hips first to set scale")
                elif chk_changed:
                    self._select_mapping_joint(joint if new_sel else "")

                ui.table_set_column_index(1)
                ui.selectable(f"{joint}##sel{i}", False)
                if ui.is_item_hovered():
                    new_hovered = joint

                ui.table_set_column_index(2)
                ui.set_next_item_width(-1)
                filtered = [lnk for lnk in self.body_link_names
                            if lnk not in used_links or lnk == current_link]
                if ui.begin_combo(f"##body{i}", current_link):
                    for lnk in filtered:
                        is_selected = (lnk == current_link)
                        changed, _ = ui.selectable(lnk, is_selected)
                        if ui.is_item_hovered():
                            combo_hovered = lnk if lnk != "(none)" else ""
                        if changed:
                            self.mapping_body_idx[joint] = self.body_link_names.index(lnk)
                            if lnk == "(none)":
                                self.human_offset_transforms.pop(joint, None)
                                self.human_offset_eulers_deg.pop(joint, None)
                            self._compute_offsets()
                            if lnk != "(none)":
                                self._flash_link  = lnk
                                self._flash_until = self.time + 1.0
                                self._update_highlight_colors()
                                mirror_joint = self.config["mirror_soma_joints"].get(joint)
                                if mirror_joint:
                                    mirror_cur = self.body_link_names[self.mapping_body_idx.get(mirror_joint, 0)]
                                    if mirror_cur == "(none)":
                                        mirror_lnk = _mirror_link_name(lnk)
                                        if mirror_lnk and mirror_lnk in self.body_link_names:
                                            self.mapping_body_idx[mirror_joint] = self.body_link_names.index(mirror_lnk)
                                            self._compute_offsets()
                        if is_selected:
                            ui.set_item_default_focus()
                    ui.end_combo()

            ui.end_table()

            if combo_hovered:
                self.hovered_link = combo_hovered
            elif new_hovered:
                link = self.body_link_names[self.mapping_body_idx[new_hovered]]
                self.hovered_link = link if link != "(none)" else ""
            else:
                self.hovered_link = ""

        # Offset editor shown below table when a joint is selected
        if has_sel:
            joint  = self.selected_mapping_joint
            link   = self.body_link_names[self.mapping_body_idx.get(joint, 0)]
            ui.separator()
            ui.text_disabled(f"{joint}  →  {link}")
            offset_tx  = self.human_offset_transforms.get(joint, wp.transform_identity())
            euler_deg  = self.human_offset_eulers_deg.get(joint, wp.vec3(0.0, 0.0, 0.0))
            p = [float(offset_tx.p[0]), float(offset_tx.p[1]), float(offset_tx.p[2])]
            r = [float(euler_deg[0]),   float(euler_deg[1]),   float(euler_deg[2])]
            ui.begin_disabled()
            ui.align_text_to_frame_padding()
            ui.text("Translation Offset:")
            ui.same_line()
            ui.set_next_item_width(-1)
            ui.input_float3("##off_t", p, "%.3f")
            ui.align_text_to_frame_padding()
            ui.text("Rotation Offset:   ")
            ui.same_line()
            ui.set_next_item_width(-1)
            ui.input_float3("##off_r", r, "%.2f")
            ui.end_disabled()

        ui.pop_id()

    def _ui_smooth_filter_tab(self, ui):
        """Render the Smooth Filter tab: per-link smooth filter mask table with add/remove controls."""
        ui.push_id("smooth_tab")
        ui.spacing()
        ui.text_wrapped("Per-link smoothing weights")
        ui.text_wrapped("W: Weight Multiplier | L: Lower Limit Offset | U: Upper Limit Offset")
        ui.spacing()

        available = [lnk for lnk in self.body_link_names
                     if lnk != "(none)" and lnk not in self.smooth_filter_masks]
        to_add = None
        new_hovered = ""
        if available:
            self.smooth_filter_add_idx = min(self.smooth_filter_add_idx, len(available) - 1)
            ui.set_next_item_width(-60)
            current_add = available[self.smooth_filter_add_idx]
            if ui.begin_combo("##sm_add", current_add):
                for ai, lnk in enumerate(available):
                    is_sel = (ai == self.smooth_filter_add_idx)
                    changed, _ = ui.selectable(f"{lnk}##sm_add_item{ai}", is_sel)
                    if ui.is_item_hovered():
                        new_hovered = lnk
                    if changed:
                        self.smooth_filter_add_idx = ai
                    if is_sel:
                        ui.set_item_default_focus()
                ui.end_combo()
            ui.same_line()
            if ui.button("Add", ui.ImVec2(-1, 0)):
                to_add = available[self.smooth_filter_add_idx]
        else:
            ui.text_disabled("All links added.")

        ui.separator()
        to_remove = None
        table_flags = (ui.TableFlags_.borders_inner_v | ui.TableFlags_.row_bg |
                       ui.TableFlags_.sizing_fixed_fit | ui.TableFlags_.scroll_y)
        avail = ui.get_content_region_avail()
        if ui.begin_table("##smooth_table", 5, table_flags, ui.ImVec2(0, avail.y)):
            ui.table_setup_column("Link",  ui.TableColumnFlags_.width_stretch)
            ui.table_setup_column("W",    ui.TableColumnFlags_.width_fixed, 56)
            ui.table_setup_column("L",    ui.TableColumnFlags_.width_fixed, 56)
            ui.table_setup_column("U",    ui.TableColumnFlags_.width_fixed, 56)
            ui.table_setup_column("##rm",  ui.TableColumnFlags_.width_fixed, 22)
            ui.table_setup_scroll_freeze(0, 1)
            ui.table_headers_row()

            for j, (link, vals) in enumerate(list(self.smooth_filter_masks.items())):
                ui.push_id(f"smf{j}")
                ui.table_next_row()
                ui.table_set_column_index(0)
                ui.selectable(f"{link}##smfsel{j}", False)
                if ui.is_item_hovered():
                    new_hovered = link
                for k in range(3):
                    ui.table_set_column_index(k + 1)
                    ui.set_next_item_width(-1)
                    changed, new_val = ui.input_float(f"##v{k}", vals[k], 0.0, 0.0, "%.2f")
                    if changed:
                        self.smooth_filter_masks[link][k] = new_val
                    if ui.is_item_hovered():
                        new_hovered = link
                ui.table_set_column_index(4)
                if ui.button("X"):
                    to_remove = link
                ui.pop_id()

            ui.end_table()

        if to_add:
            self.smooth_filter_masks[to_add] = [1.0, 1.0, 1.0]
        if to_remove:
            del self.smooth_filter_masks[to_remove]
        self.hovered_link = new_hovered

        ui.pop_id()

    def _ui_dof_editor_tab(self, ui):
        """Render the Zero Pose tab: per-DOF sliders with individual and global reset buttons."""
        ui.push_id("dof_tab")
        ui.spacing()
        ui.text_wrapped("Modify the robot pose to match the yellow SOMA skeleton reference(zero pose).")
        ui.spacing()
        _s = _ui_scale(ui)
        btn_w = (int(_UI_PANEL_WIDTH * _s) - int(_UI_PANEL_MARGIN * _s) * 2 - ui.get_style().item_spacing.x * 2) / 4

        if self.loaded_reference_joint_q is None:
            store_col  = (ui.ImVec4(0.45, 0.45, 0.45, 1.0),
                          ui.ImVec4(0.55, 0.55, 0.55, 1.0),
                          ui.ImVec4(0.35, 0.35, 0.35, 1.0))
        elif np.array_equal(self.current_joint_q, self.loaded_reference_joint_q):
            store_col  = (ui.ImVec4(0.18, 0.55, 0.18, 1.0),
                          ui.ImVec4(0.25, 0.65, 0.25, 1.0),
                          ui.ImVec4(0.12, 0.45, 0.12, 1.0))
        else:
            store_col  = (ui.ImVec4(0.65, 0.55, 0.05, 1.0),
                          ui.ImVec4(0.75, 0.65, 0.10, 1.0),
                          ui.ImVec4(0.55, 0.45, 0.02, 1.0))

        ui.push_style_color(ui.Col_.button,         store_col[0])
        ui.push_style_color(ui.Col_.button_hovered, store_col[1])
        ui.push_style_color(ui.Col_.button_active,  store_col[2])
        if ui.button("Store", ui.ImVec2(btn_w, 0)):
            self.loaded_reference_joint_q = self.current_joint_q.copy()
        ui.pop_style_color(3)
        ui.same_line()
        if self.loaded_reference_joint_q is not None and ui.button("Restore", ui.ImVec2(btn_w, 0)):
            self.current_joint_q = self.loaded_reference_joint_q.copy()
            self._compute_offsets()
        ui.same_line()
        if self.default_joint_q is not None and ui.button("Reset All to Defaults", ui.ImVec2(btn_w*2, 0)):
            self.current_joint_q          = self.default_joint_q.copy()
            self._compute_offsets()
        if not self.dof_joints:
            ui.text_disabled("No revolute joints found.")
            ui.pop_id()
            return

        avail = ui.get_content_region_avail()
        table_flags = (ui.TableFlags_.sizing_fixed_fit |
                       ui.TableFlags_.scroll_y |
                       ui.TableFlags_.no_borders_in_body)
        if ui.begin_table("##dof_table", 2, table_flags, ui.ImVec2(0, avail.y - 4)):
            ui.table_setup_column("##dof_slider", ui.TableColumnFlags_.width_stretch)
            ui.table_setup_column("##dof_reset",  ui.TableColumnFlags_.width_fixed, 28)
            new_hovered = ""
            for i, (name, q_idx, lo, hi) in enumerate(self.dof_joints):
                ui.table_next_row()
                ui.table_set_column_index(0)
                ui.text_disabled(name)
                ui.table_next_row()
                ui.table_set_column_index(0)
                val = float(self.current_joint_q[q_idx])
                ui.set_next_item_width(-1)
                changed, new_val = ui.slider_float(f"##dof{i}", val, lo, hi, "%.3f")
                if changed:
                    self.current_joint_q[q_idx] = new_val
                    self._compute_offsets()
                if self.loaded_reference_joint_q is not None:
                    ref_val = float(self.loaded_reference_joint_q[q_idx])
                    t = (max(lo, min(hi, ref_val)) - lo) / (hi - lo) if (hi - lo) > 1e-9 else 0.0
                    rmin = ui.get_item_rect_min()
                    rmax = ui.get_item_rect_max()
                    marker_x = rmin.x + t * (rmax.x - rmin.x)
                    dl = ui.get_window_draw_list()
                    yellow = ui.color_convert_float4_to_u32(ui.ImVec4(1.0, 0.85, 0.0, 0.9))
                    dl.add_line(ui.ImVec2(marker_x, rmin.y + 1), ui.ImVec2(marker_x, rmax.y - 1), yellow, 2.0)
                if ui.is_item_hovered():
                    new_hovered = name
                ui.table_set_column_index(1)
                if ui.button(f"R##rst{i}", ui.ImVec2(28, 0)):
                    self.current_joint_q[q_idx] = float(self.default_joint_q[q_idx])
                    self._compute_offsets()
            self.hovered_link = new_hovered
            ui.end_table()
        ui.pop_id()

    # ------------------------------------------------------------------
    # Add Robot popup UI
    def show_message(self, text: str, buttons: MsgButtons = MsgButtons.OK,
                     callback: Optional[Callable[[bool], None]] = None):
        """Queue a modal message dialog.

        Args:
            text:     Message body. Use newlines for multi-line content.
            buttons:  ``MsgButtons.OK`` for a single OK button;
                      ``MsgButtons.YES_NO`` for Yes / No buttons.
            callback: Called with ``True`` on OK/Yes, ``False`` on No.
        """
        self._message_queue.append(_PendingMessage(text, buttons, callback))

    def _ui_message_popup(self, ui):
        """Render the front message in the queue as a modal dialog."""
        if not self._message_queue:
            return
        result = ui.begin_popup_modal("##msg", None,
                                      ui.WindowFlags_.always_auto_resize |
                                      ui.WindowFlags_.no_title_bar)
        popup_open = result[0]
        if not popup_open:
            return

        msg = self._message_queue[0]
        ui.text(msg.text)
        ui.spacing()
        btn_w = 120

        def _dismiss(confirmed: bool):
            self._message_queue.pop(0)
            ui.close_current_popup()
            if msg.callback:
                msg.callback(confirmed)

        if msg.buttons == MsgButtons.OK:
            avail = ui.get_content_region_avail().x
            ui.set_cursor_pos_x(ui.get_cursor_pos_x() + (avail - btn_w) / 2)
            if ui.button("OK", ui.ImVec2(btn_w, 0)):
                _dismiss(True)
        else:
            if ui.button("Yes", ui.ImVec2(btn_w, 0)):
                _dismiss(True)
            ui.same_line()
            if ui.button("No", ui.ImVec2(btn_w, 0)):
                _dismiss(False)

        ui.end_popup()

    # ------------------------------------------------------------------

    def _ui_add_robot_popup(self, ui):
        """Render the modal confirmation popup for registering a new robot."""
        result = ui.begin_popup_modal("Add Robot##dlg", None,
                                      ui.WindowFlags_.always_auto_resize)
        popup_open = result[0]
        if not popup_open:
            return

        src_name = pathlib.Path(self._popup_src_path).name if self._popup_src_path else ""
        ui.text("Confirm details before registering:")
        ui.separator()
        ui.text_disabled(f"Source: {src_name}")
        ui.separator()

        ui.text("Vendor:")
        ui.set_next_item_width(280)
        _, self._popup_vendor = ui.input_text("##vendor", self._popup_vendor)

        ui.text("Robot Name:")
        ui.set_next_item_width(280)
        _, self._popup_name = ui.input_text("##rname", self._popup_name)

        ui.spacing()
        ui.separator()

        vendor = self._popup_vendor.strip()
        name   = self._popup_name.strip()
        valid_vendor = _is_safe_path_component(vendor)
        valid_name = _is_safe_path_component(name)
        dest_root = (self.assets_root / vendor / name).resolve() if valid_vendor and valid_name else None
        assets_root = self.assets_root.resolve()
        dest_safe = dest_root is not None and _is_relative_to(dest_root, assets_root)
        dest_exists = bool(dest_safe and dest_root.exists())
        existing_manifest = bool(dest_exists and (dest_root / "manifest.json").exists())
        if vendor and not valid_vendor:
            ui.text_disabled("Vendor may contain only letters, numbers, '.', '_', or '-'.")
        if name and not valid_name:
            ui.text_disabled("Robot name may contain only letters, numbers, '.', '_', or '-'.")
        if dest_exists:
            ui.text_disabled("⚠ Folder already exists.")
            if existing_manifest:
                _, self._popup_overwrite = ui.checkbox("Overwrite existing", self._popup_overwrite)
            else:
                ui.text_disabled("Existing folder has no manifest.json; overwrite is blocked.")
            ui.spacing()
        else:
            self._popup_overwrite = False

        can_confirm = (valid_vendor and valid_name and dest_safe and
                       (not dest_exists or (existing_manifest and self._popup_overwrite)))
        if not can_confirm:
            ui.begin_disabled()
        if ui.button("Register", ui.ImVec2(130, 0)):
            ui.close_current_popup()
            self._execute_register()
        if not can_confirm:
            ui.end_disabled()

        ui.same_line()
        if ui.button("Cancel", ui.ImVec2(130, 0)):
            ui.close_current_popup()

        ui.end_popup()

    # ------------------------------------------------------------------
    # Add Robot logic
    # ------------------------------------------------------------------

    def _browse_and_prepare_popup(self):
        """Open a file-picker for a robot description, pre-fill popup fields, and queue it to open."""
        import tkinter as tk
        from tkinter import filedialog as tk_fd

        root = tk.Tk()
        root.withdraw()
        path = tk_fd.askopenfilename(
            title="Select Robot Description File",
            filetypes=[
                ("Robot files", "*.urdf *.xml"),
                ("URDF",        "*.urdf"),
                ("MJCF",        "*.xml"),
                ("All files",   "*.*"),
            ])
        root.destroy()

        if not path:
            return

        src = pathlib.Path(path)
        ext = src.suffix.lower()
        if ext not in (".urdf", ".xml"):
            self.show_message(f"Unsupported file type '{ext}'.\nOnly .urdf and .xml (MJCF) files are supported.")
            return

        self._popup_src_path = path
        suggested = _parse_robot_name(path)

        self._popup_vendor = _sanitize_path_component(src.parent.name, "vendor")
        base_name = _sanitize_path_component(suggested or src.stem, "robot")
        name, i = base_name, 1
        while (self.assets_root / self._popup_vendor / name).exists():
            name = f"{base_name}_{i}"
            i += 1
        self._popup_name      = name
        self._popup_overwrite = False
        self._pending_popup   = True

    def _execute_register(self):
        """Copy robot description + meshes into the assets tree and write a manifest.json."""
        src    = pathlib.Path(self._popup_src_path)
        vendor = self._popup_vendor.strip()
        name   = self._popup_name.strip()

        if not _is_safe_path_component(vendor) or not _is_safe_path_component(name):
            print("[ERROR] Vendor and robot name may contain only letters, numbers, '.', '_', or '-'.")
            return

        assets_root = self.assets_root.resolve()
        dest_root = (self.assets_root / vendor / name).resolve()
        if not _is_relative_to(dest_root, assets_root):
            print(f"[ERROR] Refusing to register robot outside assets root: {dest_root}")
            return
        if dest_root.exists():
            if not self._popup_overwrite:
                print(f"[ERROR] Robot folder already exists: {dest_root}")
                return
            if not (dest_root / "manifest.json").exists():
                print(f"[ERROR] Refusing to overwrite folder without manifest.json: {dest_root}")
                return
            shutil.rmtree(dest_root)

        dest_desc    = dest_root / "desc"
        dest_meshes  = dest_desc / "meshes"
        dest_configs = dest_root / "configs"
        for d in (dest_desc, dest_meshes, dest_configs):
            d.mkdir(parents=True, exist_ok=True)

        ext = src.suffix.lower()
        desc_file = dest_desc / src.name
        shutil.copy2(str(src), str(desc_file))
        desc_key = "urdf_path" if ext == ".urdf" else "xml_path"

        self._repath_and_copy_meshes(desc_file, src.parent, dest_meshes)

        manifest = {
            "name": name,
            "desc": {desc_key: f"desc/{desc_file.name}"},
            "retarget_configs": {},
        }
        save_json(dest_root / "manifest.json", manifest)

        print(f"[Robot Configurator] Registered '{name}' → {dest_root}")

        robot_registry.add_search_path(str(self.assets_root))
        self._refresh_robot_list()
        if name in self.robot_list:
            self.robot_idx = self.robot_list.index(name)
            self._load_robot(name)

    def _repath_and_copy_meshes(self, desc_file: pathlib.Path,
                                 src_dir: pathlib.Path,
                                 dest_meshes: pathlib.Path):
        """Copy mesh files referenced in *desc_file* into *dest_meshes* and update paths in-place.

        DAE files are converted to STL via trimesh when available; other formats are copied as-is.
        Package-relative paths (``package://...``) are resolved by searching near *src_dir*.
        """
        try:
            content = desc_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not read {desc_file}: {e}")
            return

        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            print(f"[WARN] Could not parse XML in {desc_file}: {e}")
            return

        def _local_tag_name(tag):
            return tag.rsplit("}", 1)[-1] if "}" in tag else tag

        refs = []
        for elem in root.iter():
            tag_name = _local_tag_name(elem.tag)
            attr_names = ["filename"]
            if tag_name in ("mesh", "texture", "hfield"):
                attr_names.append("file")
            for attr in attr_names:
                ref = elem.get(attr)
                if ref and ref not in refs:
                    refs.append(ref)

        def _candidate_roots():
            start = src_dir.resolve()
            yield start
            yield from start.parents

        def _resolve_mesh_ref(ref: str) -> pathlib.Path | None:
            if ref.startswith("package://"):
                package_ref = ref[len("package://"):]
                package_name, _, package_path = package_ref.partition("/")
                for base in _candidate_roots():
                    candidates = []
                    if base.name == package_name:
                        candidates.append(base / package_path)
                    candidates.extend([
                        base / package_name / package_path,
                        base / package_path,
                    ])
                    for candidate in candidates:
                        candidate = candidate.resolve()
                        if candidate.exists():
                            return candidate
                return None

            ref_path = pathlib.Path(ref)
            if ref_path.is_absolute():
                return ref_path if ref_path.exists() else None

            for base in _candidate_roots():
                candidate = (base / ref_path).resolve()
                if candidate.exists():
                    return candidate
            return None

        replacements = {}

        for ref in refs:
            src_mesh = _resolve_mesh_ref(ref)
            if src_mesh is None:
                print(f"[WARN] Mesh not found, skipping: {ref}")
                continue

            dest_name = src_mesh.name

            shutil.copy2(str(src_mesh), str(dest_meshes / dest_name))

            replacements[ref] = f"meshes/{dest_name}"

        for old, new in replacements.items():
            pattern = re.compile(r'(?P<prefix>\b(?:filename|file)\s*=\s*["\'])' +
                                 re.escape(old) +
                                 r'(?P<suffix>["\'])')
            content = pattern.sub(rf'\g<prefix>{new}\g<suffix>', content)
        desc_file.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Robot loading
    # ------------------------------------------------------------------

    def _refresh_robot_list(self):
        """Reload the robot registry and rebuild the dropdown list."""
        robot_registry.add_search_path(str(self.assets_root))
        names = list_available_targets()
        self.robot_list = names if names else ["(no robots registered)"]
        self.robot_idx  = min(self.robot_idx, len(self.robot_list) - 1)

    def _load_robot(self, name: str):
        """Build a Newton model for *name*, run FK, and initialize mapping/DOF state.

        Args:
            name: Robot name as registered in the robot registry. No-op for placeholder strings.
        """
        if not name or name.startswith("("):
            return
        try:
            self.robot_builder = pipeline_utils.create_robot_builder(name)
        except Exception as e:
            print(f"[ERROR] Failed to load robot '{name}': {e}")
            return

        camera_state = _capture_camera_state_safe(self.viewer)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_world(self.robot_builder, wp.transform_identity())
        self.model  = builder.finalize()
        self.state  = self.model.state()
        self.default_joint_q = self.model.joint_q.numpy().copy()

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets([0, 0, 0])
        self.viewer.register_ui_callback(lambda ui: self.gui(ui), position="free")
        self._apply_robot_visibility()

        _apply_camera_state_safe(self.viewer, camera_state)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.robot_offset  = wp.transform_identity()
        self.current_robot = name
        self._init_dof_and_highlight()
        self._init_mapping(name)
        self._sync_robot_fk()
        print(f"[Robot Configurator] Loaded '{name}' — {self.model.joint_dof_count} DOFs")

    def _init_dof_and_highlight(self):
        """Populate DOF slider list, shape-color arrays, and reset joint-q to robot defaults."""
        # Shape color for hover highlight
        self.base_shape_colors = self.model.shape_color.numpy().copy()
        self.shape_body_np = self.model.shape_body.numpy()
        self.body_name_to_model_idx = {
            newton_utils.get_name_from_label(l): i
            for i, l in enumerate(self.model.body_label)
        }
        self.hovered_link       = ""
        self._prev_hovered_link = "__reset__"  # force color refresh on first frame

        # DOF editor joints
        self.current_joint_q          = self.default_joint_q.copy()
        self.loaded_reference_joint_q = None
        self.dof_joints = []
        q_start_np    = self.model.joint_q_start.numpy()
        joint_type_np = self.model.joint_type.numpy()
        lo_arr = self.model.joint_limit_lower.numpy() \
            if getattr(self.model, "joint_limit_lower", None) is not None else None
        hi_arr = self.model.joint_limit_upper.numpy() \
            if getattr(self.model, "joint_limit_upper", None) is not None else None

        for i, label in enumerate(self.model.body_label):
            if joint_type_np[i] != newton.JointType.REVOLUTE:
                continue
            name   = newton_utils.get_name_from_label(label)
            q_idx  = int(q_start_np[i])
            lo, hi = -3.14159, 3.14159
            if lo_arr is not None and q_idx < len(lo_arr):
                raw_lo, raw_hi = float(lo_arr[q_idx]), float(hi_arr[q_idx])
                if abs(raw_lo) < 1e5: lo = max(raw_lo, -6.28)
                if abs(raw_hi) < 1e5: hi = min(raw_hi,  6.28)
            self.dof_joints.append((name, q_idx, lo, hi))

    def _default_scale_from_bbox(self) -> "float | None":
        """Compute an initial human_scale_ratio from robot bbox height / SOMA skeleton height.

        Returns the ratio on success, or ``None`` if the computation cannot proceed.
        """
        def _fail(msg: str):
            print(f"[WARN] Bounding Box scale: {msg}")
            return None

        if self.soma_instance is None:
            return _fail("SOMA instance is not available.")
        saved_scale = self.soma_instance.scale
        self.soma_instance.scale = 1.0
        try:
            np_soma_tx = self.soma_instance.compute_global_transforms()  # (num_joints, 7)
            joint_z = np_soma_tx[:, 2]
            soma_height = float(joint_z.max() - joint_z.min())
            if soma_height < 1e-6:
                return _fail("SOMA skeleton Z extent is near zero.")
            robot_height = self._compute_robot_bbox_height()
            if robot_height is None:
                return _fail("Could not compute robot bounding box height.")
            return robot_height / soma_height
        finally:
            self.soma_instance.scale = saved_scale

    def _compute_robot_bbox_height(self) -> float | None:
        """Return the world-space Z extent (max_z - min_z) of all robot shapes, or None on failure."""
        if (self.model is None or self.state is None or
                self.model.shape_collision_aabb_lower is None):
            return None

        aabb_lower   = self.model.shape_collision_aabb_lower.numpy()  # (N, 3)
        aabb_upper   = self.model.shape_collision_aabb_upper.numpy()  # (N, 3)
        shape_bodies = self.model.shape_body.numpy()                  # (N,)
        shape_tfs    = self.model.shape_transform.numpy()             # (N, 7)
        body_q       = self.state.body_q.numpy()                      # (B, 7)

        min_z = float('inf')
        max_z = float('-inf')
        for i in range(self.model.shape_count):
            body_idx = int(shape_bodies[i])
            if body_idx < 0:
                continue
            lo = aabb_lower[i]
            hi = aabb_upper[i]
            shape_tx = wp.transform(*shape_tfs[i])
            body_tx  = wp.transform(*body_q[body_idx])
            world_tx = wp.transform_multiply(body_tx, shape_tx)
            for cx in (lo[0], hi[0]):
                for cy in (lo[1], hi[1]):
                    for cz in (lo[2], hi[2]):
                        z = float(wp.transform_point(world_tx, wp.vec3(cx, cy, cz))[2])
                        min_z = min(min_z, z)
                        max_z = max(max_z, z)

        return (max_z - min_z) if max_z > min_z else None

    def _init_mapping(self, name: str):
        """Load retargeter and scaler configs for *name* and populate all mapping UI state.

        If no saved config exists, all joints are reset to ``(none)`` with default weights.
        The ``loaded_reference_joint_q`` is updated from the scaler config when present.
        """
        # Reset scale before anything else so bbox and offset calculations use the original skeleton.
        self.human_scale_ratio = 1.0
        if self.soma_instance is not None:
            self.soma_instance.scale = 1.0

        # Build sorted link name list from the robot builder (no ground plane)
        raw_names = [newton_utils.get_name_from_label(l) for l in self.robot_builder.body_label]
        self.body_link_names = ["(none)"] + sorted(set(raw_names))

        # Defaults
        self.mapping_body_idx        = {j: 0 for j in self.config["soma_joints"]}
        self.mapping_weights         = {j: list(self.config["default_soma_weights"][j]) for j in self.config["soma_joints"]}
        self.retargeter_config       = {}
        self.retargeter_config_path  = None
        self._saved_scale            = 1.0
        self.human_offset_transforms  = {}
        self.human_offset_eulers_deg  = {}
        self._loaded_joint_offsets    = {}  # snapshot from disk, used as save fallback
        self.selected_mapping_joint   = ""
        self.edit_offset_tx           = wp.transform_identity()
        self.smooth_filter_masks      = {}
        self.smooth_filter_add_idx    = 0

        entry = robot_registry.get(name)

        # Load retargeter config (ik_match_table + pointer to scaler config)
        rel = entry.retarget_configs.get("soma")
        if rel:
            cfg_path = pathlib.Path(entry.manifest_dir) / rel
            try:
                cfg = load_json(cfg_path)
                self.retargeter_config      = cfg
                self.retargeter_config_path = cfg_path
                ik_table = cfg.get("ik_match_table", {})
                for joint, entry_data in ik_table.items():
                    if joint not in self.config["soma_joints"]:
                        continue
                    link = entry_data.get("t_body", "")
                    if link in self.body_link_names:
                        self.mapping_body_idx[joint] = self.body_link_names.index(link)
                    self.mapping_weights[joint] = [
                        float(entry_data.get("t_weight", 1.0)),
                        float(entry_data.get("r_weight", 1.0)),
                    ]
                for link, vals in cfg.get("smooth_joint_filter_objective_body_masks", {}).items():
                    self.smooth_filter_masks[link] = [float(v) for v in vals[:3]]
                print(f"[Robot Configurator] Loaded mapping from {cfg_path.name}")
            except Exception as e:
                print(f"[WARN] Could not load retargeter config: {e}")

        # Load scaler config (scale ratio, reference pose, joint offsets)
        scaler_rel = self.retargeter_config.get("human_robot_scaler_config")
        if scaler_rel:
            scaler_path = pathlib.Path(entry.manifest_dir) / scaler_rel
            try:
                scaler = load_json(scaler_path)

                scale = scaler.get("human_scale_ratio")
                if scale is not None:
                    self.human_scale_ratio = float(scale)
                    self._saved_scale      = float(scale)

                ref_q = scaler.get("reference_joint_q")
                if ref_q is not None:
                    ref_arr = np.array(ref_q, dtype=np.float32)
                    if len(ref_arr) == len(self.current_joint_q):
                        self.current_joint_q          = ref_arr.copy()
                        self.loaded_reference_joint_q = ref_arr.copy()

                for joint, data in scaler.get("human_joint_offsets", {}).items():
                    if joint not in self.config["soma_joints"]:
                        continue
                    t = data[0]
                    q = data[1]
                    self.human_offset_transforms[joint] = wp.transform(
                        wp.vec3(float(t[0]), float(t[1]), float(t[2])),
                        wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                    self._sync_euler_cache(joint)

                print(f"[Robot Configurator] Loaded scaler config from {scaler_path.name}")
            except Exception as e:
                print(f"[WARN] Could not load scaler config: {e}")

    # ------------------------------------------------------------------
    # SOMA loading
    # ------------------------------------------------------------------

    def _load_soma(self):
        """Load the SOMA zero-pose BVH and build the skeleton/mesh renderers."""
        try:
            converter = SpaceConverter(FacingDirectionType.MUJOCO)
            bvh_path = pipeline_utils.get_source_zero_pose_asset_path(pipeline_utils.SourceType.SOMA)
            self.soma_skeleton, animation = bvh_utils.load_bvh(bvh_path)
            z_rot = wp.transform(wp.vec3(0, 0, 0),
                                    wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.radians(90.0)))
            soma_xform = wp.mul(z_rot, converter.transform(wp.transform_identity()))
            self.soma_instance = SkeletonInstance(self.soma_skeleton, _SOMA_COLOR, soma_xform)
            self.soma_instance.set_local_transforms(animation.get_local_transforms(0))
            self.soma_skeleton_renderer = SkeletonRenderer(self.soma_skeleton, [0])
            soma_mesh = pipeline_utils.get_source_model_mesh(pipeline_utils.SourceType.SOMA, self.soma_skeleton)
            self.soma_mesh_renderer = SkeletalMeshRenderer(soma_mesh)
            print("[Robot Configurator] SOMA reference mesh loaded")
        except Exception as e:
            print(f"[WARN] Could not load SOMA mesh: {e}")

    # ------------------------------------------------------------------
    # Auto Scale / Create Offset
    # ------------------------------------------------------------------

    def _default_scale_from_pelvis(self) -> "float | None":
        """Compute scale as the ratio of the robot Hips link height to SOMA Hips height in Z.

        Returns the ratio on success, or ``None`` if the computation cannot proceed.
        """
        def _fail(msg: str):
            print(f"[WARN] Pelvis scale: {msg}")
            return None

        if self.soma_instance is None or self.state is None:
            return _fail("SOMA instance or robot state is not available.")
        self._sync_robot_fk()
        hips_link = self.body_link_names[self.mapping_body_idx.get("Hips", 0)]
        if hips_link == "(none)":
            return _fail("Hips joint is not mapped to any robot link.")
        robot_body_idx = self.body_name_to_model_idx.get(hips_link, -1)
        if robot_body_idx < 0:
            return _fail(f"Robot body '{hips_link}' not found in model.")
        soma_hips_idx = self.soma_skeleton.joint_index("Hips")
        if soma_hips_idx < 0:
            return _fail("'Hips' joint not found in SOMA skeleton.")

        prev_xform = wp.transform(self.soma_instance.xform)
        self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
        np_soma_tx = self.soma_instance.compute_global_transforms(scale=1.0)
        self.soma_instance.xform = prev_xform

        np_body_q = self.state.body_q.numpy()
        q_inv = wp.quat_inverse(self.robot_offset.q)
        robot_offset_inv = wp.transform(wp.quat_rotate(q_inv, -self.robot_offset.p), q_inv)
        robot_hip_tx = wp.mul(robot_offset_inv, wp.transform(*np_body_q[robot_body_idx]))

        robot_hip_z = float(robot_hip_tx.p[2])
        soma_hip_z  = float(np_soma_tx[soma_hips_idx][2])
        if abs(soma_hip_z) < 1e-6:
            return _fail("SOMA Hips Z is near zero.")
        ratio = robot_hip_z / soma_hip_z
        print(f"[Robot Configurator] Pelvis scale: ratio = {ratio:.4f}")
        return ratio

    def _apply_scale(self, ratio: float) -> None:
        """Set human_scale_ratio, propagate to the SOMA skeleton, and recompute offsets."""
        self.human_scale_ratio = ratio
        if self.soma_instance is not None:
            self.soma_instance.scale = ratio
        self._compute_offsets()

    def _compute_offsets(self):
        """Recompute local offset transforms for every mapped SOMA joint.

        Gizmo offsets are handled identically to :meth:`_compute_scale_ratio`: ``soma_offset``
        is temporarily applied to the SOMA instance and ``robot_offset`` is stripped from
        ``state.body_q`` before sampling positions.
        """
        if self.soma_instance is None or self.state is None:
            return
        self._sync_robot_fk()

        # Apply soma_offset so SOMA world positions match the viewport.
        # compute_global_transforms() applies self.scale so positions are already in robot scale.
        prev_xform = wp.transform(self.soma_instance.xform)
        self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
        np_soma_tx = self.soma_instance.compute_global_transforms()
        self.soma_instance.xform = prev_xform

        # Strip robot_offset from body_q (FK bakes it into the root transform)
        np_body_q = self.state.body_q.numpy()
        q_inv = wp.quat_inverse(self.robot_offset.q)
        robot_offset_inv = wp.transform(wp.quat_rotate(q_inv, -self.robot_offset.p), q_inv)

        count = 0
        for joint in self.config["soma_joints"]:
            link = self.body_link_names[self.mapping_body_idx.get(joint, 0)]
            if link == "(none)":
                continue
            robot_body_idx = self.body_name_to_model_idx.get(link, -1)
            soma_joint_idx = self.soma_skeleton.joint_index(joint)
            if robot_body_idx < 0 or soma_joint_idx < 0:
                continue
            soma_tx  = wp.transform(*np_soma_tx[soma_joint_idx])
            robot_tx = wp.mul(robot_offset_inv, wp.transform(*np_body_q[robot_body_idx]))
            self.human_offset_transforms[joint] = _compute_local_offset(
                soma_tx, robot_tx)
            self._sync_euler_cache(joint)
            count += 1
        print(f"[Robot Configurator] Create Offset: computed {count} offsets")

    def _sync_euler_cache(self, joint: str):
        """Update the cached Euler-degree representation for *joint*'s offset quaternion."""
        if joint in self.human_offset_transforms:
            euler_rad = wp.quat_to_euler(self.human_offset_transforms[joint].q, 2, 1, 0)
            self.human_offset_eulers_deg[joint] = wp.vec3(
                float(wp.degrees(euler_rad[0])),
                float(wp.degrees(euler_rad[1])),
                float(wp.degrees(euler_rad[2])))

    def _select_mapping_joint(self, joint: str):
        """Select a SOMA joint for offset editing and position the edit gizmo at its mapped location."""
        self.selected_mapping_joint = joint
        if not joint or self.soma_instance is None or self.soma_skeleton is None:
            return
        soma_joint_idx = self.soma_skeleton.joint_index(joint)
        if soma_joint_idx < 0:
            return
        prev_xform = wp.transform(self.soma_instance.xform)
        self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
        np_soma_tx = self.soma_instance.compute_global_transforms()
        self.soma_instance.xform = prev_xform
        soma_tx    = wp.transform(*np_soma_tx[soma_joint_idx])
        offset_tx  = self.human_offset_transforms.get(joint, wp.transform_identity())
        self.edit_offset_tx = _apply_local_offset(soma_tx, offset_tx)

    def _render_remapped_coordinates(self):
        """Draw blended coordinate-axis gizmos at each mapped SOMA joint's remapped world position."""
        if self.soma_instance is None:
            return
        # Temporarily apply soma_offset so axes follow the gizmo position.
        # compute_global_transforms() applies self.scale so soma positions are already in robot scale.
        prev_xform = wp.transform(self.soma_instance.xform)
        self.soma_instance.xform = wp.mul(self.soma_offset, self.soma_instance.xform)
        np_soma_tx = self.soma_instance.compute_global_transforms()
        self.soma_instance.xform = prev_xform

        coords = []
        for joint, offset_tx in self.human_offset_transforms.items():
            if self.body_link_names[self.mapping_body_idx.get(joint, 0)] == "(none)":
                continue
            soma_joint_idx = self.soma_skeleton.joint_index(joint)
            if soma_joint_idx < 0:
                continue
            soma_tx   = wp.transform(*np_soma_tx[soma_joint_idx])
            mapped_tx = _apply_local_offset(soma_tx, offset_tx)
            blended   = wp.transform(
                wp.lerp(soma_tx.p, mapped_tx.p, self.remapped_coords_blend),
                wp.quat_slerp(soma_tx.q, mapped_tx.q, self.remapped_coords_blend))
            coords.append(blended)

        if coords:
            self.coordinate_renderer.draw(self.viewer, coords, 0.08, 200)

    # ------------------------------------------------------------------
    # Save mapping
    # ------------------------------------------------------------------

    def _save_mapping(self):
        """Write the retargeter config JSON (ik_match_table + smooth filter masks) to disk.

        Creates the file and updates the manifest if no config existed yet.

        Returns:
            pathlib.Path: Path of the written config file.
        """
        # Build updated ik_match_table from current UI state (skip unmapped joints)
        ik_table = {}
        for joint in self.config["soma_joints"]:
            link = self.body_link_names[self.mapping_body_idx[joint]]
            if link == "(none)":
                continue
            ik_table[joint] = {
                "t_body":   link,
                "r_body":   link,
                "t_weight": round(self.mapping_weights[joint][0], 4),
                "r_weight": round(self.mapping_weights[joint][1], 4),
            }

        smooth_masks = {link: [round(v, 4) for v in vals]
                        for link, vals in self.smooth_filter_masks.items()}

        if self.retargeter_config_path is not None:
            # Update existing config in place (preserve all other fields)
            self.retargeter_config["ik_match_table"] = ik_table
            if smooth_masks:
                self.retargeter_config["smooth_joint_filter_objective_body_masks"] = smooth_masks
                self.retargeter_config["smooth_joint_filter_weight"] = self.config["default_ik_params"]["smooth_joint_filter_weight"]
            else:
                self.retargeter_config.pop("smooth_joint_filter_objective_body_masks", None)
                self.retargeter_config["smooth_joint_filter_weight"] = 0.0
            cfg_path = self.retargeter_config_path
        else:
            # Create a new retargeter config file for this robot
            robot_name = self.current_robot
            entry = robot_registry.get(robot_name)
            configs_dir = pathlib.Path(entry.manifest_dir) / "configs"
            configs_dir.mkdir(exist_ok=True)
            cfg_path = configs_dir / f"soma_to_{robot_name}_retargeter_config.json"

            self.retargeter_config = {
                **self.config["default_ik_params"],
                "smooth_joint_filter_weight": self.config["default_ik_params"]["smooth_joint_filter_weight"] if smooth_masks else 0.0,
                "human_robot_scaler_config": f"configs/soma_to_{robot_name}_scaler_config.json",
                "post_processing": {
                    "robot_config": f"configs/{robot_name}_post_processing_config.json"
                },
                "ik_match_table": ik_table,
            }
            if smooth_masks:
                self.retargeter_config["smooth_joint_filter_objective_body_masks"] = smooth_masks
            self.retargeter_config_path = cfg_path

            # Update manifest to point at the new config
            manifest_path = pathlib.Path(entry.manifest_dir) / "manifest.json"
            try:
                manifest = load_json(manifest_path)
                # Handle both flat and targets-array manifests
                targets = manifest.get("targets", [manifest])
                for t in targets:
                    if t.get("name") == robot_name:
                        t.setdefault("retarget_configs", {})["soma"] = \
                            f"configs/soma_to_{robot_name}_retargeter_config.json"
                save_json(manifest_path, manifest)
            except Exception as e:
                print(f"[WARN] Could not update manifest: {e}")

        save_json(cfg_path, self.retargeter_config)
        print(f"[Robot Configurator] Saved retargeter config → {cfg_path}")
        return cfg_path

    def _save_configs(self):
        """Save all three config files: retargeter, scaler, and post-processing."""
        saved = []
        cfg_path = self._save_mapping()
        saved.append(cfg_path.name)

        # Derive scaler config path from retargeter config
        robot_name = self.current_robot
        entry = robot_registry.get(robot_name)
        configs_dir = pathlib.Path(entry.manifest_dir) / "configs"
        configs_dir.mkdir(exist_ok=True)

        scaler_rel = self.retargeter_config.get(
            "human_robot_scaler_config",
            f"configs/soma_to_{robot_name}_scaler_config.json")
        scaler_path = pathlib.Path(entry.manifest_dir) / scaler_rel

        # robot_root = body link mapped to Hips (fall back to "(none)")
        robot_root = self.body_link_names[self.mapping_body_idx.get("Hips", 0)]
        if robot_root == "(none)":
            robot_root = ""

        # Build human_joint_offsets from all known transforms (includes loaded values for
        # unmapped joints, so previously saved offsets are preserved across save cycles).
        joint_offsets = {}
        for joint, tx in self.human_offset_transforms.items():
            t, q = tx.p, tx.q
            joint_offsets[joint] = [
                [round(float(t[0]), 4), round(float(t[1]), 4), round(float(t[2]), 4)],
                [round(float(q[0]), 4), round(float(q[1]), 4),
                 round(float(q[2]), 4), round(float(q[3]), 4)],
            ]

        self._saved_scale = self.human_scale_ratio

        scaler_data = {
            "human_type":        "soma",
            "human_root":        "Hips",
            "robot_type":        robot_name,
            "robot_root":        robot_root,
            "human_scale_ratio": float(self.human_scale_ratio),
            "reference_joint_q": [round(float(v), 10) for v in self.current_joint_q],
            "human_joint_offsets": joint_offsets,
        }

        save_json(scaler_path, scaler_data)
        print(f"[Robot Configurator] Saved scaler config        → {scaler_path}")
        saved.append(scaler_path.name)

        # Post-processing config: only generated for robots that don't have one yet.
        # Once a config file exists on disk it is left untouched, since it may contain
        # hand-tuned values that this tool has no UI to edit.
        pp_rel = (self.retargeter_config
                  .get("post_processing", {})
                  .get("robot_config", f"configs/{robot_name}_post_processing_config.json"))
        pp_path = pathlib.Path(entry.manifest_dir) / pp_rel
        if pp_path.exists():
            print(f"[Robot Configurator] Post-processing config already exists, leaving untouched → {pp_path}")
        else:
            pp_data = self._build_post_processing_config()
            save_json(pp_path, pp_data)
            print(f"[Robot Configurator] Saved post-processing config → {pp_path}")
            saved.append(pp_path.name)

        robot_registry.reload()
        self.show_message("Config files saved:\n\n" + "\n".join(f"  {n}" for n in saved))

    def _build_post_processing_config(self):
        """Build the post-processing config dict from the current mapping state.

        Effector positions within the dict are computed dynamically so that ``ik_limbs``
        indices stay correct even when some SOMA joints are left unmapped.

        Returns:
            dict: Post-processing config with ``contact_correction`` and ``limb_stabilizer`` sections.
        """
        _EFFECTOR_JOINTS = [
            ("Hips",      [30.0,  8.0]),
            ("LeftLeg",   [1.5,  0.15]),
            ("LeftShin",  [1.0,   1.0]),
            ("LeftFoot",  [10.0,  2.0]),
            ("RightLeg",  [1.5,  0.15]),
            ("RightShin", [1.0,   1.0]),
            ("RightFoot", [10.0,  2.0]),
        ]

        # Build effectors dict and track which SOMA joints were included (preserving order)
        effectors = {}
        effector_order = []
        for joint, weights in _EFFECTOR_JOINTS:
            link = self.body_link_names[self.mapping_body_idx.get(joint, 0)]
            if link and link != "(none)":
                effectors[link] = weights
                effector_order.append(joint)

        # Compute ik_limbs indices as positions within the effectors dict (what limb_stabilizer expects)
        def _idx(joint):
            return effector_order.index(joint) if joint in effector_order else -1

        ik_limbs = {}
        left = [_idx("LeftLeg"), _idx("LeftShin"), _idx("LeftFoot")]
        if all(i >= 0 for i in left):
            ik_limbs["LeftFoot"] = {
                "effectors":      left,
                "hint_reference": left[1],
                "hint_offset":    [0.25, 0.0, 0.0],
            }
        right = [_idx("RightLeg"), _idx("RightShin"), _idx("RightFoot")]
        if all(i >= 0 for i in right):
            ik_limbs["RightFoot"] = {
                "effectors":      right,
                "hint_reference": right[1],
                "hint_offset":    [0.25, 0.0, 0.0],
            }

        return {
            "contact_correction": {
                "transition_frames":           4,
                "propagation_ratio":           0.4,
                "rotation_propagation_ratio":  0.15,
                "enable_flatten_foot_plant":   True,
            },
            "limb_stabilizer": {
                "ik_iterations":        20,
                "joint_limit_weight":   10.0,
                "effectors":            effectors,
                "ik_root":              0,
                "root_correction_ratio": 1.0,
                "root_smooth_alpha":    0.8,
                "ik_limbs":             ik_limbs,
            },
        }


# ---------------------------------------------------------------------------
# Camera helpers (safe: no-op when camera isn't ready yet)
# ---------------------------------------------------------------------------

def _capture_camera_state_safe(viewer):
    """Snapshot the viewer's camera pose; returns ``None`` if the camera isn't ready yet."""
    try:
        cam = viewer.camera
        return {
            "pos":   [float(cam.pos[i])   for i in range(3)],
            "pivot": [float(cam.pivot[i]) for i in range(3)],
            "pitch": float(cam.pitch),
            "yaw":   float(cam.yaw),
            "fov":   float(cam.fov),
            "near":  float(cam.near),
            "far":   float(cam.far),
        }
    except Exception:
        return None


def _apply_camera_state_safe(viewer, state):
    """Restore a camera snapshot produced by :func:`_capture_camera_state_safe`; no-op on failure."""
    if state is None:
        return
    try:
        cam = viewer.camera
        vec3_type = type(cam.pos)
        cam.pos   = vec3_type(*state["pos"])
        cam.pivot = vec3_type(*state["pivot"])
        cam.pitch = state["pitch"]
        cam.yaw   = state["yaw"]
        cam.fov   = state["fov"]
        cam.near  = state["near"]
        cam.far   = state["far"]
    except Exception:
        pass


# ---------------------------------------------------------------------------

def main():
    """Entry point: parse CLI args, create the Newton viewer, and run the Robot Configurator."""
    import newton.examples
    parser = newton.examples.create_parser()
    parser.set_defaults(viewer="gl")
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "assets" / "robot_configurator" / "robot_config_generator_config.json"),
        help="Path to the robot_config_generator config JSON file.")
    viewer, args = newton.examples.init(parser)
    if not pathlib.Path(args.config).exists():
        print(f"[ERROR]: Config file not found: {args.config}")
        exit(1)
    if args.viewer != "null" and hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    config = load_json(args.config)
    with wp.ScopedDevice(args.device):
        app = Viewer(viewer, config)
        app.run()


if __name__ == "__main__":
    main()
