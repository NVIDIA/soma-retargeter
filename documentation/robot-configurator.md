# Add a robot with the configurator

This tutorial uses the **Robot Configurator** application to register the public, handless Unitree H1 robot and create a full SOMA-to-H1 configuration. 

The source model is Unitree's [`h1_description`](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/h1_description), specifically [`urdf/h1.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/h1_description/urdf/h1.urdf). That file describes the basic H1 without dexterous hands.

**Important notes about robot descriptions:** 
- The Robot Configurator can register .urdf and mjcf .xml description files. If a robot is defined by a .xacro file, it needs to be converted to .urdf or mjcf .xml files prior to being registered in the Robot Configurator. 
- Supported formats for robot meshes are STL and DAE. 
- No units conversions are done when registering robots. Here are the units that the SOMA Retargeter needs:
	- Lengths and positions: meters
	- Angles: radians
	- Mass: kilograms
- Robot definitions are assumed to be oriented as:
	- +Z up
	- +X forward
	- +Y left
- High-DOF hands are not supported yet. Avoid registering robot descriptions that include  articulated fingers or dense hand mechanisms. The current retargeting workflow is designed for humanoid body motion, not high-DOF dexterous hands. Finger links can add many extra joints and may cause memory or solver issues. If a manufacturer description includes hands, prefer a simplified version that stops at the wrist or palm, or remove links beyond the hand before registering the robot.

## What this tutorial produces

The configurator creates the following artifacts for a given robot:

```text
assets/robotics/
└── unitree/
    └── unitree_h1/
        ├── manifest.json
        ├── desc/
        │   ├── h1.urdf
        │   └── meshes/
        └── configs/
            ├── soma_to_unitree_h1_scaler_config.json
            ├── soma_to_unitree_h1_retargeter_config.json
            └── unitree_h1_post_processing_config.json
```

`assets/robotics/` is for robot assets you create or override. Do not add H1 under `soma_retargeter/assets/robotics/`; that directory contains the robot targets bundled with the package.

## Before you start

Manufacturer robot repositories should be cloned **outside** the `motion-transfer-oss` repository. Treat them as source data repositories: they are not part of this project, and they should not be copied into `assets/`, `documentation/`, or any other tracked folder in the main repo. The configurator imports the selected robot description and copies only the registered robot assets into `assets/robotics/`.

The examples below assume your terminal is open in the `motion-transfer-oss` repository root. They clone Unitree's repository into a sibling folder named `../robot-sources/`, next to the main repo:

```text
Gitlab/
├── motion-transfer-oss/
└── robot-sources/
    └── unitree_ros/
```

The public Unitree repository keeps `urdf/` and `meshes/` as sibling directories, while the configurator resolves mesh paths relative to the selected file's directory. Make a small staging directory with the layout the importer expects. This leaves the upstream clone unchanged and keeps temporary source data outside the main repo.

### Linux shell

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git ../robot-sources/unitree_ros
mkdir -p ../robot-sources/h1-import/meshes
cp ../robot-sources/unitree_ros/robots/h1_description/urdf/h1.urdf ../robot-sources/h1-import/h1.urdf
cp ../robot-sources/unitree_ros/robots/h1_description/meshes/*.STL ../robot-sources/h1-import/meshes/
```

### Windows PowerShell

```powershell
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git ../robot-sources/unitree_ros
New-Item -ItemType Directory -Force ../robot-sources/h1-import/meshes | Out-Null
Copy-Item ../robot-sources/unitree_ros/robots/h1_description/urdf/h1.urdf ../robot-sources/h1-import/h1.urdf
Copy-Item ../robot-sources/unitree_ros/robots/h1_description/meshes/*.STL ../robot-sources/h1-import/meshes/
```

The configurator rewrites copied mesh references to `meshes/<file>`. Keep Unitree's license and attribution requirements in mind when using the resulting robot asset.

## Understand the H1 model

The basic H1 URDF has **19 actuated revolute joints**:

- Left leg: hip yaw, hip roll, hip pitch, knee, and ankle — 5 DOFs.
- Right leg: hip yaw, hip roll, hip pitch, knee, and ankle — 5 DOFs.
- Torso: torso yaw — 1 DOF.
- Left arm: shoulder pitch, shoulder roll, shoulder yaw, and elbow — 4 DOFs.
- Right arm: shoulder pitch, shoulder roll, shoulder yaw, and elbow — 4 DOFs.

The kinematic branches are:

```text
pelvis
├── left_hip_yaw_link
│   └── left_hip_roll_link
│       └── left_hip_pitch_link
│           └── left_knee_link
│               └── left_ankle_link
├── right_hip_yaw_link
│   └── right_hip_roll_link
│       └── right_hip_pitch_link
│           └── right_knee_link
│               └── right_ankle_link
└── torso_link
    ├── left_shoulder_pitch_link
    │   └── left_shoulder_roll_link
    │       └── left_shoulder_yaw_link
    │           └── left_elbow_link
    └── right_shoulder_pitch_link
        └── right_shoulder_roll_link
            └── right_shoulder_yaw_link
                └── right_elbow_link
```

The URDF also has fixed sensor and decorative links attached to the torso. They do not add articulated DOFs. In particular, this H1 model has:

- no articulated head;
- no wrist or hand links after either elbow;
- no toe links after either ankle.

## Start the configurator

```bash
uv run python app/tools/robot_config_generator.py --viewer gl
```

The window is titled **Robot Configurator**. Its right panel contains:

- a **Registered Robots** selector and **Add Robot...** button;
- scale, SOMA visibility, coordinate-map, and gizmo controls;
- **Zero Pose**, **Mapping**, and **Smooth Filter** tabs;
- a **Save Configs** button.

![Robot Configurator viewer](images/robot-configurator.png)

## Register Unitree H1

1. Click **Add Robot...**.

![Add Robot button](images/robot-configurator-add-robot.png)

2. Select `../robot-sources/h1-import/h1.urdf`.
3. In the confirmation dialog, set **Vendor** to `unitree`.
4. Set **Robot Name** to `unitree_h1`.
5. Click **Register**.

Registration copies the URDF and meshes, writes a minimal manifest, rescans `assets/robotics/`, and loads the new robot. Watch the terminal while it runs. Resolve every `Mesh not found` warning before continuing; missing meshes usually mean the staging directory was not prepared as shown above.

![H1 robot registered in the viewer](images/robot-configurator-h1-registered.png)

If registration created the wrong vendor or name, remove only the newly created `assets/robotics/<vendor>/<name>/` directory, restart the configurator, and register again. Do not overwrite another target unless you intend to replace it.
### Define the human-to-robot scale

The human-to-robot scale controls how source motion maps onto the robot. Choose it based on the intent of the retarget. The same animation can favor overall body overlap, match leg length, or match stride depending on the scale you set.

In most cases, matching the robot's overall size and keeping the human and robot overlapping gives the best general results. To do this quickly, scale the SOMA rig to the robot's bounding box by using the `Set To...` button and set the dropdown next to it to `Bounding Box`.

![Scale to Bounding Box option](images/robot-configurator-scale-to-boundingbox.png)

For H1, this produces a scale of `1.0167`.

![Scale adjusted to bounding box](images/robot-configurator-scale-done.png)

You can also set the scale by other references, depending on which body region matters most for your motion:

- Match the pelvis height to the robot's Hips link to keep the robot's legs closer to the human animation scale.
- Match the elbow height manually when a robot's proportions place its arms further from the SOMA elbows.

When matching a specific robot feature, enable the `SOMA Mesh` checkbox to display the SOMA mesh. The mesh makes it easier to judge overlap while you adjust the scale.

![SOMA mesh displayed in the viewer](images/robot-configurator-soma-mesh-display-scale.png)

For a more extreme option, scale the human so the feet distance matches the robot. This keeps the robot in the human's footsteps, but can produce large steps.

![Extreme feet-distance scale](images/robot-configurator-extreme-scale.png)

Turn off the `SOMA Mesh` display before continuing this tutorial.
## Establish a reference Zero Pose

The purpose of the reference **Zero Pose** is to make the robot's calibration pose correspond sensibly to the SOMA zero pose before offsets are computed for the configuration files. 

The SOMA reference is an upright calibration pose: legs straight, arms down, elbows bent about 90 degrees so the forearms point forward, and palms facing one another. H1 has no wrists or hands, so match the pelvis, legs, upper arms, and elbows while treating the forearm direction as an approximation at each terminal elbow link.

Many robots, like the H1, are already in the required Zero Pose. So in the case of the H1, the Zero Pose is the default pose when the robot is registered. No further editing is needed in this case. But you can familiarize yourself with the Zero Pose tools so that you can prepare your robot adequately in the future. 

1. Open the **Zero Pose** tab. The sliders use the DOF limits from the URDF. 

![Zero Pose tab in the Robot Configurator](images/robot-configurator-zeropose-tab.png)

2. Adjust the joint sliders while checking pelvis, torso, knees, ankles, shoulders, and elbows.
	   *Tip: every numerical slider can be edited manually. Ctrl-click the value area, not the slider handle, then type the exact value you want.*
3. Keep the pose inside the displayed joint limits and avoid visually collapsed or crossed limbs.
4. Click **Store** when you have a defensible candidate. 

The **Store** button becomes green to indicate that a Zero Pose is now in memory, ready to be restored if needed. 

![Zero Pose buttons in the Robot Configurator](images/robot-configurator-zeropose-buttons.png)
 
**Restore** button returns to the last **Zero Pose** captured with the **Store** button. When the **Store** button is yellow, it means that the current pose differs from the Zero Pose. Storing a new Zero Pose will set the **Store** button to green again. 

**Reset All** returns to the URDF **defaults** pose, which may be different from the Zero Pose. If you need to set the Zero Pose to the URDF default pose, press **Reset All to Defaults** button first, then press the **Store** button. 

Note that **Save Configs saves the pose currently shown**. Make sure to restore the **Zero Pose** prior to saving and do not move sliders away from the pose before saving.
## Create the initial mapping

Open **Mapping** tab. 

![Mapping tab in the Robot Configurator](images/robot-configurator-mapping-tab.png)

Each SOMA joint can select one robot body link or `(none)`. A body link already used by another row is removed from the other dropdowns, so the UI does not permit two SOMA joints to share one robot link.

Start with this topology-based mapping:

| SOMA joint     | Initial H1 link            | Reason                                                       |
| -------------- | -------------------------- | ------------------------------------------------------------ |
| `Hips`         | `pelvis`                   | Floating robot root and leg-branch origin                    |
| `Chest`        | `torso_link`               | Only articulated torso body                                  |
| `LeftArm`      | `left_shoulder_roll_link`  | Stable upper-arm/shoulder landmark used by similar humanoids |
| `LeftForeArm`  | `left_elbow_link`          | Terminal articulated forearm body                            |
| `RightArm`     | `right_shoulder_roll_link` | Symmetric upper-arm/shoulder landmark                        |
| `RightForeArm` | `right_elbow_link`         | Terminal articulated forearm body                            |
| `LeftLeg`      | `left_hip_roll_link`       | Stable upper-leg/hip landmark                                |
| `LeftShin`     | `left_knee_link`           | Lower-leg landmark                                           |
| `LeftFoot`     | `left_ankle_link`          | Terminal left foot/ankle body                                |
| `RightLeg`     | `right_hip_roll_link`      | Symmetric upper-leg/hip landmark                             |
| `RightShin`    | `right_knee_link`          | Lower-leg landmark                                           |
| `RightFoot`    | `right_ankle_link`         | Terminal right foot/ankle body                               |

Notice that, as you search in the dropdown menus for the robot's link, the links are highlighted on the robot. 

![Robot links highlighted in the viewer](images/robot-configurator-link-highlight-elbow.png)

Leave these rows unmapped:

- `Head`, because H1 has no articulated head.
- `LeftHand` and `RightHand`, because H1 has no wrist or hand bodies.
- `LeftToeBase` and `RightToeBase`, because H1 has no toe bodies.

General notes on mapping:

- Always choose the root link of the robot for the hips joint. It is important for IK solving.
- If a joint doesn't have a link mapping, leave the link to (none). 

### Unmapped manual placement

The UI supports a limited editing workflow for joints transformations. To edit mapped (or unmapped) joints, do the following: 

- Select the checkbox at the left of that mapping row.

![Selecting a row for manual edits](images/robot-configurator-manual-edit.png)

- use the transform gizmo in the viewport to position and rotate the joint on the robot.

![Transform gizmos in the viewer for manual edits](images/robot-configurator-transformgizmo.png)

Map effectors to robots link and move their landmark toward an estimated endpoint on the robot. These choices do **not** create a head, wrist, hand, or toe joint. They only place an IK target relative to an existing body. Because one body link cannot be used twice, using `left_elbow_link` for `LeftHand` means it cannot simultaneously serve `LeftForeArm`.

The configurator cannot create a new virtual body link or an independent landmark unattached to a mapped body. For the first H1 configuration, keep unsupported landmarks unmapped. 

You can use the visibility tools to hide and unhide SOMA and robots components. This can be useful when joints are difficult to see on the robot.

![Visibility options in the Robot Configurator](images/robot-configurator-visibiliy.png)

Use the **SOMA to Robot Map** blend slider to visualize how SOMA joints are being mapped to the robot. 

![Blend slider effect in the viewer](images/robot-configurator-blend.gif)

## Smooth-filter masks

The **Smooth Filter** tab supports optional per-link smooth-filter entries. Each entry has `W`, `L`, and `U`, and new entries start at `[1, 1, 1]`. `W` is the mask weight, while `L` and `U` represent lower and upper joint-limit normalized offsets. An unlisted joint has a zero smooth mask; a listed joint with `[1, 1, 1]` entry activates its mask without tightening the default offsets. Lower `W` values favours this joint over others to achieve a reach pose, reducing locking DOF issues.  

Use the Add button to include links in the smooth filter masks list:

![Smooth Filter - Add links button](images/robot-configurator-add-smoothfilter.png)


For the initial configuration, set the following: 

![Smooth Filter values table for Unitree H1](images/robot-configurator-smoothfilter-values.png)

For a detailed explanation on smooth-filter masks, refer to the [Robot configuration reference](robot-configuration-reference.md).

## Save and inspect the generated files

Click **Save Configs**. The configurator writes the scaler, retargeter, and post-processing files and updates the manifest's `retarget_configs.soma` entry.

> **Runtime checkpoint — saved configuration**
>
> Confirm all three JSON files exist under `assets/robotics/unitree/unitree_h1/configs/`. Open them and verify:
>
> - `robot_type` is `unitree_h1`;
> - `robot_root` is `pelvis`;
> - `reference_joint_q` has 26 values for this floating-base 19-DOF model;
> - only intended source joints appear in `human_joint_offsets` and `ik_match_table`;
> - all referenced body-link names exactly match the H1 URDF;
> - the manifest points `retarget_configs.soma` to the generated retargeter config.

The configurator seeds mapping weights and post-processing settings with general defaults. They are scaffolding, not optimized H1 parameters. Do not publish or deploy the configuration as “finished” solely because it saves successfully.

## Verify that the converter discovers H1

Open the interactive converter:

```bash
uv run python app/bvh_to_csv_converter.py --viewer gl
```

The converter includes every manifest found recursively below `assets/robotics`, so `unitree_h1` should now appear in **Robot Options**.

![Robot Options in the retargeting viewer](images/viewer-robot-options-h1.png)

## Validate before tuning

Use [Interactive retargeting](interactive-retargeting.md) to test at least:

- neutral standing and ordinary walking;
- deep knee and hip motion;
- turns and torso rotation;
- broad arm movement and elbow flexion;
- foot lift, landing, and planted intervals;
- first and last frames, where initialization behavior is easiest to spot.

Check for:

- mirrored or rotated limbs, indicating a bad correspondence offset;
- persistent height error, indicating a poor scale or reference pose;
- feet penetrating or hovering above the ground;
- joints repeatedly hitting URDF limits;
- unsupported head, hand, or toe targets distorting the torso or limbs;
- oscillation introduced by an unnecessary smooth-filter mask.

Return to the configurator for topology, pose, scale, or offset corrections. This is an iterative process. It's a good practice to leave the 2 applications opened (Robot Configurator and Converter) simultaneously and to go back and forth to test and update the configuration. 

Check the **Tips to solve common issues** section in the [Robot configuration reference](robot-configuration-reference.md) to address common issues like popping, fast shoulder rotations and more. 

Use [IK weight optimizer](ik-weight-optimizer.md) only after the mapping is structurally correct. The optimizer does not repair a wrong link mapping, scale, zero pose, or proxy choice.

## Troubleshooting registration

- **Meshes are missing:** rebuild `../robot-sources/h1-import` with `h1.urdf` beside a `meshes/` directory, then register again.
- **The robot is not in the converter:** include `"assets/robotics"` in `extra_robot_paths` and restart, or click **Reload** in an already-open converter.
- **Auto Scale prints that Hips is unmapped:** assign `Hips → pelvis` first.
- **A desired body link is absent from a dropdown:** it may already be assigned to another SOMA joint.
- **Restore returns to an unexpected pose:** click **Set** only after selecting the intended reference pose.
- **Save produces a target that fails on startup:** inspect every relative path and body-link name using [Robot configuration reference](robot-configuration-reference.md).
- **A proxy makes the result worse:** return that source joint to `(none)`.
  Switch to another registered target and back, then reapply the intended
  mappings before the first save. For an already saved scaler config, delete
  its stale `human_joint_offsets` entry. Then restart and retest.

## Where to go next

Read [Robot configuration reference](robot-configuration-reference.md) before hand-editing any generated JSON. It explains which file owns each value and how the paths depend on one another.

[← Previous: First retarget](first-retarget.md) · [Documentation home](../README.md) · [Next: Robot configuration reference →](robot-configuration-reference.md)
