# Robot configuration reference

A robot target is a description file plus a connected set of configuration JSON files. The manifest gives the target a registry name. The retargeter uses the scaler and post-processing files. Those files must all agree on source-joint names, robot body-link names, and robot coordinate layout.

Use [Add a robot with the configurator](robot-configurator.md) for a guided H1 example. 

Use this reference page when reviewing generated files, creating an override, or diagnosing configurations.

## Robot descriptions prerequisites

### supported formats

Robot configuration files can register robots from .urdf and mjcf .xml descriptions. If a robot is defined by a .xacro file, it needs to be converted to .urdf or mjcf .xml files prior to being registered in the Robot Configurator. S

Supported formats for robot meshes are STL or DAE

### Units

No units conversions are done when registering robots. The SOMA Retargeter assumes that all unites provided from the robot description are:
	- Lengths and positions: meters
	- Angles: radians
	- Mass: kilograms

### Robot transformations

Robot definitions are assumed to be oriented as:

- +Z up
- +X forward
- +Y left

No offset is applied to the robots in the SOMA Retargeter. Robots are assumed to be at origin and feet on the ground. 

### Memory management

High-DOF hands are not supported yet. Avoid registering robot descriptions that include  articulated fingers or dense hand mechanisms. The current retargeting workflow is designed for humanoid body motion, not high-DOF dexterous hands. Finger links can add many extra joints and may cause memory or solver issues. If a manufacturer description includes hands, prefer a simplified version that stops at the wrist or palm, or remove links beyond the hand before registering the robot.

## Bundled and user robot targets

SOMA Retargeter has two distinct robot-asset locations:

- `soma_retargeter/assets/robotics/` contains targets bundled with the Python package. The registry scans this location automatically.
- `assets/robotics/` is the repository-level location used by the Robot Configurator for user-created targets. The converter discovers the user robots as an `extra_robot_paths` in its configuration file .

Do not edit a bundled robot target merely to test a change. Prefer to copy the complete robot target directory to a user search root included in the `extra_robot_paths`. Keep or deliberately change its manifest name. An extra robot target with the same registry `name` overrides the bundled target for that process.

A conventional user robot target looks like:

```text
assets/robotics/
└── <vendor>/
    └── <robot-directory>/
        ├── manifest.json
        ├── desc/
        │   ├── <robot>.urdf
        │   └── meshes/
        └── configs/
            ├── <source>_to_<robot>_scaler_config.json
            ├── <source>_to_<robot>_retargeter_config.json
            └── <robot>_post_processing_config.json
```

The vendor and directory levels organize files; discovery is recursive and does not derive the target name from those folder names. `manifest.json` is authoritative for the registered robot name.

## Discovery and `extra_robot_paths`

The registry recursively scans each search root for files named `manifest.json`. It scans bundled assets first, then additional roots. A later target with the same name replaces the earlier registry entry.

The converter registers extra roots from its JSON config file (assets\default_bvh_to_csv_converter_config.json) : 

```json
{
    "extra_robot_paths": ["assets/robotics"]
}
```

`extra_robot_paths` may be a JSON array or a single string. Values are converted to absolute paths from the process's current working directory, so run documented commands from the repository root or use an explicit path. A missing root is skipped; it does not create the directory.

The interactive converter reads these paths at startup. Its **Reload** button rescans for registered roots. The Robot Configurator is different: it always scans the bundled root and repository-level `assets/robotics/`, and it writes new targets to `assets/robotics/`.

In Python, applications can register another root before creating a pipeline:

```python
from soma_retargeter.robotics.robot_registry import register_robots_path

register_robots_path("assets/robotics")
```

Use a new name to keep two robot target variants selectable. Reuse a bundled name only when you intend the extra target to override it.

## `manifest.json`

The simplest manifest declares one target:

```json
{
    "name": "unitree_h2",
    "desc": {
        "urdf_path": "desc/H2.urdf"
    },
    "retarget_configs": {
        "soma": "configs/soma_to_h2_retargeter_config.json"
    }
}
```

All local paths in a manifest are relative to the directory containing that manifest.

### Manifest fields

- `name` — required registry key used by `retarget_target`, the UI target selector, and `robot_type` in the scaler.
- `desc` — required object describing how Newton loads the robot.
- `desc.urdf_path` — path to a URDF. Use either this or `xml_path`.
- `desc.xml_path` — path to an MJCF XML. Use either this or `urdf_path`.
- `desc.urdf_offset` — optional precomputed Z lift for a URDF. Without it, the loader builds the URDF once, measures the lowest collision geometry, and rebuilds it at ground level.
- `desc.newton_asset` — optional Newton asset identifier. When present, description paths are resolved below Newton's downloaded asset directory rather than the manifest directory.
- `retarget_configs` — source-name-to-retargeter-path map. The current public source key is `soma`.

If neither `urdf_path` nor `xml_path` is present, model loading fails. If `retarget_configs.soma` is absent, the model can still appear as registered, but SOMA retargeting fails because no source-to-target configuration can be resolved.

A manifest may group targets:

```json
{
    "targets": [
        {
            "name": "robot_variant_a",
            "desc": {"xml_path": "desc/robot.xml"},
            "retarget_configs": {
                "soma": "configs/soma_to_variant_a_retargeter_config.json"
            }
        },
        {
            "name": "robot_variant_b",
            "desc": {"urdf_path": "desc/robot.urdf"},
            "retarget_configs": {
                "soma": "configs/soma_to_variant_b_retargeter_config.json"
            }
        }
    ]
}
```

Each entry becomes a separate registry target and shares the manifest directory as its path base.

## Description and meshes

The description defines the robot's:

- body-link and joint names;
- parent-child topology;
- joint types, axes, and limits;
- inertial, visual, and collision geometry;
- default joint pose.

Configuration files refer to **body-link names**, not URDF joint names. For example, `left_ankle_link` is a body, while `left_ankle_joint` is the revolute joint connecting it.

Keep a portable description:

```text
desc/
├── robot.urdf
└── meshes/
    ├── pelvis.STL
    └── ...
```

Use description-relative mesh references such as `meshes/pelvis.STL`. Avoid absolute paths and machine-specific package locations in a distributable target.

Convert Xacro files to URDF or MJCF prior to register.  Xacro registration requires the external `xacro` executable and expands the source to a URDF before copying meshes. `xacro` is not installed as a core project dependency.

When the configurator registers a URDF or MJCF, it:

1. copies the selected description into `desc/`;
2. finds `filename="..."` mesh references;
3. copies referenced STL, DAE and OBJ files into `desc/meshes/`;
4. rewrites successful references to `meshes/<filename>`.

For `package://<package>/...` references, the importer removes the package prefix and resolves the remaining path relative to the selected description's directory. Stage the description and mesh tree accordingly, as shown in the [H1 tutorial](robot-configurator.md). Treat every `Mesh not found` warning as a failed portability check.

## The scaler config

The scaler converts source-skeleton transforms into robot-aligned IK effector transforms. The retargeter points to it through `human_robot_scaler_config`.

Example shape:

```json
{
    "human_type": "soma",
    "human_root": "Hips",
    "robot_type": "unitree_h2",
    "robot_root": "pelvis",
    "human_scale_ratio": 1.0,
    "reference_joint_q": [],
    "human_joint_offsets": {
        "Hips": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }
}
```

The numeric values above illustrate structure only; they are not an H2 calibration.

General note on human joint offsets:

- If a joint doesn't have a link mapping, do not include it in `human_joint_offsets`.
### Scaler fields

- `human_type` — source label written by the configurator; currently `soma`.
- `human_root` — source-root label written by the configurator; normally `Hips`.
- `robot_type` — required target registry name. It must identify the same target selected by the manifest and converter.
- `robot_root` — mapped root body recorded by the configurator, normally the body selected for `Hips`.
- `human_scale_ratio` — required scalar applied to source translations. In normal converter use, the motion is proportionally scaled about the source root; root height is scaled as well.
- `reference_joint_q` — optional complete Newton joint-coordinate vector used to initialize the base IK solve. For a floating-base robot it begins with root translation `[x, y, z]` and quaternion `[x, y, z, w]`, followed by articulated coordinates in model order.
- `human_joint_offsets` — required map from SOMA joint name to `[translation, quaternion]`. Quaternions use `[x, y, z, w]` and are normalized when loaded.

The configurator's auto scale uses the current SOMA Hips height and the mapped robot-root height. It computes offsets from the current zero pose, current scale, and each mapped link. Therefore `human_scale_ratio`, `reference_joint_q`, and `human_joint_offsets` form one calibration set; changing one without recomputing or validating the others can make the target inconsistent.

Only source joints present in `human_joint_offsets` become scaler effectors. Their nearest mapped source ancestors form the scaled effector hierarchy. This is why leaving unsupported anatomy unmapped is valid, but accidentally omitting a required foot or root mapping affects later IK and post-processing.

## The retargeter config

The retargeter config is the central runtime configuration. It selects the scaler, defines IK correspondences and weights, controls initialization and solver behavior, and optionally selects post-processing.

Representative structure:

```json
{
    "num_initialization_frames": 10,
    "num_stabilization_frames": 5,
    "human_robot_scaler_config": "configs/soma_to_h2_scaler_config.json",
    "ik_iterations": 24,
    "joint_limit_weight": 10.0,
    "smooth_joint_filter_weight": 5.5,
    "enable_post_processing": true,
    "enable_contact_processing": true,
    "post_processing": {
        "robot_config": "configs/h2_post_processing_config.json"
    },
    "ik_match_table": {
        "Hips": {
            "t_body": "pelvis",
            "r_body": "pelvis",
            "t_weight": 30.0,
            "r_weight": 2.0
        }
    }
}
```

These are configurator defaults that demonstrate field placement, not optimized H2 values.

### Initialization and solver fields

- `num_initialization_frames` — number of source-zero-pose transition frames prepended before solving. Half transition the root; the remainder blend joint rotations.
- `num_stabilization_frames` — additional first-pose hold frames after initialization. Both groups are removed from exported output.
- `human_robot_scaler_config` — required path to the scaler, relative to the manifest directory.
- `ik_iterations` — base IK iterations per frame; values below 1 are clamped to 1.
- `joint_limit_weight` — global joint-limit objective weight; negative values are clamped to zero.
- `smooth_joint_filter_weight` — global weight for the smooth joint-limit avoidance objective; negative values are clamped to zero.

More iterations can improve convergence but increase processing cost. Higher objective weights do not guarantee better motion; evaluate tracking and joint-limit behavior together.

### `ik_match_table`

Each key is a mapped SOMA joint. Its object contains:

- `t_body` — robot body whose position follows the scaled source effector;
- `r_body` — robot body whose rotation follows it;
- `t_weight` — position objective weight;
- `r_weight` — rotation objective weight.

The configurator assigns the same selected link to `t_body` and `r_body`, but hand-edited configs may use different bodies. Every source name must exist in the SOMA skeleton and in the scaler's mapped effectors. Every target body must exactly match a body-link name in the loaded description. A typo fails during pipeline construction rather than being silently ignored.

Mappings and weights solve different problems:

- correct the link mapping, scale, zero pose, and offsets first;
- tune `t_weight` and `r_weight` only after the structural calibration works.

The [IK weight optimizer](ik-weight-optimizer.md) changes these two weights. It does not change the scaler or body-link choices.

General notes on `ik_match_table`:

- Always choose the root link of the robot for the hips joint. 
- If a joint doesn't have a link mapping, do not include it in the `ik_match_table`.

### Smooth joint-limit filter

`smooth_joint_filter_objective_body_masks` is optional:

```json
{
    "smooth_joint_filter_objective_body_masks": {
        "left_shoulder_pitch_link": [1.0, 1.0, 1.0]
    }
}
```

Each entry is `[mask, lower_offset, upper_offset]`:

| Value | Meaning | Units |
|---|---|---|
| `mask` | Scales the residual; `0` disables it. | Unitless |
| `lower_offset` | Moves the ramp start inward from the lower DOF limit. | Radians |
| `upper_offset` | Moves the ramp start inward from the upper DOF limit. | Radians |

The filter uses a smooth residual curve to discourage a joint from approaching either limit. For joint coordinate `q` and limits `L` and `U`:

```text
center      = (L + U) / 2
distance    = max((L + lower_offset) - q, 0, q - (U - upper_offset))
ramp        = smooth_exponential_ramp(distance)
residual    = (q - center) * ramp * smooth_joint_filter_weight * mask
```

The residual is zero between the two ramp starts. On either side, the ramp has an inverted Gaussian-like shape: it stays near zero at first, then rises smoothly and steeply toward `1`. `mask` and `smooth_joint_filter_weight` scale its strength without changing its shape.

`lower_offset` controls how far before the lower DOF limit the penalty begins, and `upper_offset` does the same for the upper DOF limit. Larger offsets start penalizing the DOF earlier. As the DOF approaches its limit, the objective gradually makes the pose more costly. When the kinematic chain has enough freedom, the IK solver can then distribute more of the motion across less-penalized joints while still trying to match the effector targets.

The arrows measure each offset from its DOF limit to its ramp start, while the shaded band marks the zero-residual region between the starts. The animation changes each offset, then moves the lower DOF limit from `-1` to `-2` to show that the offsets follow their corresponding limits.

![Effect of changing joint-limit offsets and DOF limits](images/smooth-joint-limit-moving-limits.gif)

Newton stores revolute DOF coordinates and limits in radians. Each offset is therefore a distance in radians measured inward from its corresponding limit. Unlisted bodies and entries with `mask = 0` are disabled. Start with `lower_offset` and `upper_offset` at `1 rad`, adjust where each ramp begins, then tune `mask` gradually. The **Smooth Filter** tab edits these values; body names must match the robot model.

### Post-Processing switches

- `enable_post_processing` — enables the limb-stabilizer pass. If true, `post_processing.robot_config` must resolve to a file containing `limb_stabilizer`.
- `enable_contact_processing` — enables contact detection and plant correction within post-processing. (see important note below)
- `post_processing.robot_config` — robot post-processing path relative to the manifest directory.

**Important note : Experimental feature**

Contact-guided foot stabilization is an **experimental** post-processing feature for motion retargeting. It reduces visible foot drift and abrupt motion when a source foot is detected as planted.

**To enable:** 

The feature runs only when both runtime settings are `true`:

```json
{
  "enable_post_processing": true,
  "enable_contact_processing": true
}
```

H2 configuration already sets both values to `true`.

**To disable:**

To reproduce the non-stabilized baseline, set both flags to `false`:

```json
{
  "enable_post_processing": false,
  "enable_contact_processing": false
}
```

Turning off only `enable_contact_processing` disables plant detection and correction but leaves the general limb stabilizer active.

**Limitations**

This is not a collision or scene-interaction feature. It does not guarantee that a foot will avoid penetration or floating, and it is not intended to solve stairs, uneven terrain, props, or handrails.

**Advanced tuning**

The source-contact thresholds and H2 post-processing settings are shared advanced defaults. Do not change them for a single motion. If you find a repeatable issue, compare the same input with the feature on and off, then report the source BVH, target configuration, output CSV/USD, and affected frame range to the retargeting team.

To learn more about configuring foot contacts, check the [Contact detection & foot-plant correction configuration](foot-contact.md) page.

## The post-processing config

The robot post-processing file supplies robot-specific foot correction and limb stabilization. 

Generated structure:

```json
{
    "contact_correction": {
        "transition_frames": 4,
        "propagation_ratio": 0.4,
        "rotation_propagation_ratio": 0.15,
        "enable_flatten_foot_plant": true,
        "sole_normal_local": [0.0, 0.0, 1.0]
    },
    "limb_stabilizer": {
        "ik_iterations": 20,
        "joint_limit_weight": 10.0,
        "effectors": {
            "pelvis": [30.0, 8.0],
            "left_hip_roll_link": [1.5, 0.15],
            "left_knee_link": [1.0, 1.0],
            "left_ankle_link": [10.0, 2.0]
        },
        "ik_root": 0,
        "root_correction_ratio": 1.0,
        "root_smooth_alpha": 0.8,
        "ik_limbs": {
            "LeftFoot": {
                "effectors": [1, 2, 3],
                "hint_reference": 2,
                "hint_offset": [0.25, 0.0, 0.0]
            }
        }
    }
}
```

Again, this shows dependencies, not validated H2 tuning.

### Contact-correction fields

- `transition_frames` — becomes the plant-correction blend transition length.
- `propagation_ratio` — positional correction propagated around a planted interval.
- `rotation_propagation_ratio` — rotational correction propagated around a planted interval.
- `enable_flatten_foot_plant` — allows planted-foot orientation flattening.
- `sole_normal_local` — sole-up direction, expressed in the **local frame of the foot**  (`LeftFoot`/`RightFoot` in the retargeter's `ik_match_table`). Used when flattening a planted foot.

### Limb-stabilizer fields

- `ik_iterations` — iterations for the stabilizer solve.
- `joint_limit_weight` — stabilizer joint-limit objective weight.
- `effectors` — ordered map from robot body to `[translation_weight, rotation_weight]`.
- `ik_root` — zero-based index into the ordered `effectors` map, normally the pelvis/root entry.
- `root_correction_ratio` — amount of computed vertical root correction.
- `root_smooth_alpha` — root-correction smoothing coefficient, clamped to `[0, 1]`.
- `ik_limbs` — source limb targets and their robot effector chains.
- `ik_limbs.<name>.effectors` — three indices into the ordered `effectors` map for the limb chain.
- `hint_reference` — index into `effectors` used to orient the two-bone IK hint *(the knee)*.
- `hint_offset` — local offset from that hint body *(in front of the knee)*.

The indices depend on JSON object insertion order. If you add, remove, or reorder `effectors`, update `ik_root`, every `effectors` index list, and every `hint_reference`. The configurator calculates these indices dynamically from its current Hips, leg, shin, and foot mappings; manual edits must preserve the same consistency.

For contact-aware feet, scaler offsets for `LeftFoot` and `RightFoot`, retargeter mappings for those source joints, and corresponding post-processing limb definitions must all exist and agree.

## Dependency chain

Runtime resolution follows this order:

```text
converter config
 ├─ extra_robot_paths ─► registry search roots
 ├─ retarget_source ───► manifest retarget_configs key
 └─ retarget_target ───► manifest name
                            ├ desc ─► URDF/MJCF ─► meshes
                            └ retargeter config
                                ├── human_robot_scaler_config ─► scaler
                                ├── ik_match_table ─► source joints + robot bodies
                                └── post_processing.robot_config ─► post-processing
```

All robot-local config references are resolved from the target's manifest directory, even when one config points to another.

## Validation checklist

Review a target in this order:

1. **Discovery:** the intended search root is listed, the manifest is found, and the target name appears once.
2. **Manifest:** one valid description path exists and `retarget_configs.soma` resolves.
3. **Description:** all mesh references are portable, geometry loads, joint limits are credible, and body-link names are stable.
4. **Scaler:** `robot_type` matches the manifest name; scale is measured; `reference_joint_q` has the exact coordinate count; offsets cover every intended source effector.
5. **Retargeter:** its scaler and post-processing paths exist; source names and target body names are exact; weights are finite and intentionally chosen.
6. **Post-processing:** effectors exist in the robot; all indices refer to the intended ordered entries; foot limbs agree with the scaler and retargeter.
7. **Converter:** `retarget_target` uses the manifest name and representative motions run in both interactive and headless modes.

## Common failures

- **Unknown target:** the extra search root was not registered, `name` differs from `retarget_target`, or the process needs a reload.
- **No retargeter config for source:** `retarget_configs` lacks the `soma` key or its path is wrong.
- **Description must specify** `urdf_path` **or** `xml_path`**:** `desc` is present but has no supported description key.
- **A body name is not in the model:** a retargeter, smooth mask, or post-processing entry uses a joint name instead of a body link, contains a typo, or belongs to another robot variant.
- **Reference joint shape mismatch:** `reference_joint_q` came from a different description or omitted the floating-root coordinates.
- **Post-processing is enabled but missing:** set a valid `post_processing.robot_config` with `limb_stabilizer`, or explicitly disable post-processing for diagnosis.
- **Foot correction affects the wrong bodies:** recheck ordered effector indices, foot mappings, and scaler offsets.
- **Bundled changes seem to have no effect:** an extra target with the same registry name is overriding it.
- **User override seems to have no effect:** another later extra search root defines the same target name.

## Tips to solve common issues

### Tune IK weights in config file

Many motion-quality issues can be reduced by tuning the IK weights directly in the `*_retargeter_config.json` file. Default weights may work well for one robot but produce unstable or exaggerated motion on another. Change one group of values at a time and review several representative animations before keeping the result. Best values are often a compromise between values validated on different animations. 

Each `ik_match_table` entry can contain translation and rotation weights:

- `t_weight` controls how strongly the solver matches the source position.
- `r_weight` controls how strongly it matches the source orientation.

Higher values give that objective more influence relative to the other IK objectives. Excessively high values can make the solve stiff or transfer problems elsewhere.

Common adjustments include:

- **Excessive torso rotation**: If the robot torso rotates more strongly than the source chest, increase `Chest.r_weight`. This makes the torso orientation follow the human chest more closely. Increase it in steps of `1.0` or `2.0`. `Chest.t_weight` can also help when the chest position drifts, but it does not directly correct orientation.
- **Incorrect arm angles or popping**: An arm can flip or take an undesirable direction while reaching its end effector. Increase `LeftArm.r_weight` and `RightArm.r_weight` to preserve more of the source upper-arm orientation. Use small increments such as `0.1` or `0.5`, then review several animations. Values that are too high can restrict the solve and make the overall result worse.
- **Pelvis popping**: Fast pelvis motion, jumps, and strong hips movement during walking can cause the pelvis to pop when its objective is too weak relative to the other effectors. Increase `Hips.t_weight` in steps of approximately `10.0` and `Hips.r_weight` in steps of approximately `2.0`. This gives the pelvis position and orientation more influence during the solve.

These adjustments only help when the robot diverges from the source objective. They do not remove motion that already exists in the human animation.

### Unmapped joints

Source joints such as `Head`, `LeftToeBase`, and `RightToeBase` may not correspond to dedicated robot links. Approximately placing these landmarks relative to the robot can still improve the solve.

A toe-base landmark provides a directional reference beyond the foot, which can improve foot and ankle orientation. A head landmark can similarly provide a reference above the torso and help orient the upper body. These landmarks do not create new robot links; they only provide additional spatial references for the mapped IK hierarchy.

Place them approximately where the corresponding anatomical feature would be on the robot. Exact placement is less important than maintaining a sensible direction and distance from the nearest mapped body.

### Redistribute shoulder motion near joint limits

If a shoulder coordinate approaches its limit and absorbs too much of the required rotation, increase the first value, `mask`, in that link's `smooth_joint_filter_objective_body_masks` entry. This makes the solver favor other coordinates while the configured ramp is active.

If avoidance begins too late, increase `lower_offset` or `upper_offset` on the affected side. These values are distances in radians measured inward from the corresponding limit.

Increase `mask` gradually, typically in steps of `0.1`, and review the result across multiple animations. Values that are too high can reduce reach or move the problem to another joint. Changes have no effect while the coordinate remains entirely inside the zero-contribution region.

## Where to go next

Use [Interactive retargeting](interactive-retargeting.md) to validate one motion visually. Once the topology and scaler are correct, use [IK weight optimizer](ik-weight-optimizer.md) and [Compare and tune](compare-and-tune.md) to evaluate weight changes without overwriting the baseline.

[← Previous: Add a robot with the configurator](robot-configurator.md) · [Documentation home](../README.md) · [Next: Interactive retargeting →](interactive-retargeting.md)