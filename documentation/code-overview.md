# Code Overview

This page summarizes the main application entry points and the importable `soma_retargeter` package.

## `app/`

| File | Description |
|------|-------------|
| `bvh_to_csv_converter.py` | Main Retargeting application. Runs the interactive OpenGL viewer for live retargeting preview and handles headless batch BVH-to-CSV conversion. |
| `tools/robot_config_generator.py` | Robot Configurator tool. Loads a URDF or MJCF, copies robot assets into `assets/robotics/`, creates the robot manifest, defines SOMA-to-robot link mappings, and writes the scaler and retargeter config files. |
| `tools/ik_weight_optimizer.py` | IK Weight Optimizer tool. Runs repeated retargeting passes over a BVH corpus, scores tracking and smoothness metrics, and writes an optimized `ik_match_table` config. |

## `soma_retargeter/`

### `pipelines/`

The core retargeting pipeline. `SomaRetargetingPipeline` in `soma_retargeting_pipeline.py` orchestrates the full run:

1. Scales human effector targets to robot proportions (`HumanToRobotScaler`).
2. Prepends initialization and stabilization frames.
3. Runs a **base IK pass** (Newton Levenberg–Marquardt) with position, rotation, joint-limit, and smooth-filter objectives.
4. Optionally runs a **post-process pass** — two-bone limb stabilization and foot-plant correction blending.
5. Clamps all joint coordinates to limits before output.

| File | Role |
|------|------|
| `soma_retargeting_pipeline.py` | Top-level pipeline: config loading, pass construction, frame loop, output. |
| `ik_solver.py` | `IKSolver` / `BaseIKSolver` — own the Newton IK solver, joint_q buffer, and objective factories. |
| `ik_pass.py` | `BasePass` / `PostProcessPass` — per-frame execution with CUDA-graph capture. |
| `ik_objectives.py` | `IKSmoothJointFilter` — custom smooth joint-limit objective. |
| `limb_stabilizer.py` | Two-bone IK for feet and hands during the post-process pass. |
| `joint_limit_clamper.py` | Hard clamp of joint coordinates to model limits after the solve. |
| `plant_subsegment_detector.py` | Detects foot-plant sub-segments from contact results. |
| `plant_correction_blender.py` | Blends foot-plant corrections into effector targets. |
| `utils.py` | Config loading, source-type helpers, robot builder construction. |

### `animation/`

Core data structures and algorithms operating on source-skeleton data.

| File | Role |
|------|------|
| `skeleton.py` | `Skeleton` and `SkeletonInstance` — joint hierarchy, local/global transforms. |
| `animation_buffer.py` | `AnimationBuffer` — frame-indexed local transforms with sample-rate metadata. |
| `contact_detection.py` | Velocity/jerk-based foot contact detection (`detect_contacts_velocity_jerk`). |
| `contact_phase.py` | Foot landmark models (authored and heuristic) used by plant detection. |
| `ik.py` | Skeleton-space IK helpers. |
| `mesh.py` | Skinned mesh data structures. |

### `robotics/`

Robot-side data and registry.

| File | Role |
|------|------|
| `robot_registry.py` | Registers robot manifests and resolves robot asset roots by name. |
| `human_to_robot_scaler.py` | Computes scaled effector targets from source-skeleton poses. |
| `csv_animation_buffer.py` | `CSVAnimationBuffer` — wraps raw joint-coordinate arrays for CSV output. |

### `io/`

File I/O for animation and configuration formats.

| File | Role |
|------|------|
| `bvh.py` | BVH parser — returns `(Skeleton, AnimationBuffer)`. |
| `csv.py` | CSV writer for retargeted joint-coordinate sequences. |
| `usd.py` | USD skeletal mesh loader. |
| `utils.py` | Shared path and format helpers. |

### `renderers/`

OpenGL visualization used by the interactive applications.

| File | Role |
|------|------|
| `skeleton_renderer.py` | Draws joint hierarchies as lines and spheres. |
| `mesh_renderer.py` | Draws skinned robot and human meshes. |
| `coordinate_renderer.py` | Draws remapped coordinate axis overlays. |
| `base_renderer.py` | Shared VAO/shader setup. |

### `utils/`

Math and integration helpers.

| File | Role |
|------|------|
| `newton_utils.py` | Joint coordinate masks, initialization frame construction, Newton/Warp helpers. |
| `space_conversion_utils.py` | `SpaceConverter` — converts source-space root transforms to Newton world space. |
| `pose_utils.py` | Batched global pose computation. |
| `math_utils.py` | Quaternion, vector, and interpolation utilities. |
| `maya_utils.py` | Maya coordinate-space helpers. |
| `time_utils.py` | Timing utilities. |

### `assets/`

Bundled runtime assets: SOMA source skeleton BVH, robot description manifests, scaler and retargeter configs, and USD skeletons.

---

## Example

Retarget one bundled SOMA BVH to `unitree_h2`, save the result as CSV, then open the viewer to play it back.

```python
from pathlib import Path

import newton
import newton.examples
import warp as wp

from app.bvh_to_csv_converter import Viewer
from soma_retargeter.io import bvh as bvh_io
from soma_retargeter.io import csv as csv_io
from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
from soma_retargeter.utils.newton_utils import get_filtered_joint_names
from soma_retargeter.utils.space_conversion_utils import (
    SpaceConverter,
    get_facing_direction_type_from_str,
)

bvh_path = Path("assets/motions/bvh/Neutral_walk_forward_002__A057.bvh")
csv_path = Path("output/examples/unitree_h2_walk.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

# Load the SOMA-skeleton BVH animation.
skeleton, animation = bvh_io.load_bvh(str(bvh_path))

# Match the default converter orientation (Mujoco convention).
space_converter = SpaceConverter(get_facing_direction_type_from_str("Mujoco"))
source_offset = space_converter.transform(wp.transform_identity())

# Build and run the retargeting pipeline.
pipeline = SomaRetargetingPipeline(
    skeleton=skeleton,
    source_type="soma",
    robot_type="unitree_h2",
)
pipeline.add_input_motions([animation], [source_offset], scale_animation=True)
csv_buffers = pipeline.execute()

# CSV columns are the robot's revolute joints in model order.
joint_names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
csv_io.save_csv(str(csv_path), joint_names, csv_buffers[0])
print(f"Saved retargeted H2 motion to {csv_path}")

# Open the viewer and play back the CSV on the robot.
parser = newton.examples.create_parser()
parser.set_defaults(viewer="gl")
viewer, _ = newton.examples.init(parser)

viewer_app = Viewer(viewer, {
    "retarget_source_facing_direction": "Mujoco",
    "retarget_target": "unitree_h2",
})
viewer_app._reset_viewer_model()
viewer_app.load_csv_file(str(csv_path))
viewer_app.run()
```

---

[Documentation home](../README.md)
