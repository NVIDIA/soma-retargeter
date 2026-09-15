# Interactive retargeting

Use the interactive viewer when you want to inspect a SOMA BVH motion, retarget it to a robot, review the result, and save a robot-motion CSV.

![SOMA BVH-to-robot retargeting pipeline](images/retargeting-pipeline.png)

## Before you start

Run commands from the repository root after completing the project installation:

```bash
git lfs pull
uv sync
```

The GUI requires a desktop display. On Linux it also requires Tk:

```bash
sudo apt-get install python3.12-tk
```

## Open the viewer

```bash
uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl
```

The window is titled **BVH to CSV Converter**. It opens with:

- **Scene Options** on the right, containing robot, motion, and visibility controls.
- **Playback Controls** along the bottom.
- A 3D viewport showing the selected robot.

The default config selects `unitree_g1`. The target list comes from the robot manifests bundled with the installed package, plus any paths listed in `extra_robot_paths`.

![Converter viewer opened and default screen](images/converter-viewer-opened.png)

## Retarget and save one motion

Follow these steps in order:

1. Under **Robot Options**, select the target robot. If robot files were added while the viewer was open, click **Reload** to rescan registered robot paths.
	1. for this tutorial, choose unitree_h2

![Unitree H2 robot in Converter viewer](images/converter-unitree-h2.png)

2. Under **Motion**, beside **BVH Motion**, click **Load**.

![Converter load BVH button](images/converter-bvh-load.png)

3. Select a SOMA-compatible `.bvh` file. For a first run, use:
   `assets/motions/bvh/dance_hiphop_shuffle_square_R_fast_002__A318.bvh`.
4. Inspect the human motion in the viewport. Playback starts automatically.

![BVH playback in converter viewer](images/converter-bvh-playback.png)

5. Click **Retarget**. This runs the retargeting pipeline in memory and loads the resulting robot motion into the viewer.

![Retarget button in converter viewer](images/converter-retarget-button.png)

5. Scrub the time slider and inspect difficult frames such as foot plants, turns, arm reaches, and the first and last frames.

![BVH animation retargeted in the converter viewer](images/converter-bvh-retargeted.png)

6. Beside **CSV Motion**, click **Save** and choose a new `.csv` path. The viewer does not save automatically.

![BVH save button in converter viewer](images/converter-save-csv.png)

Expected result: the saved CSV contains one row per robot-motion frame and can be loaded again with **CSV Motion → Load**.

## Playback and visibility controls

![Converter playback controls](images/converter-playback-controls.png)

- **Time (s)** scrubs to a specific time.
- **Pause** and **Play** stop and resume playback.
- **Speed** ranges from `-2.0` to `2.0`; negative values play backward.
- **Loop** controls whether playback wraps at the end.

![Converter visibility options](images/converter-visibility-options.png)

- **Show Mesh** displays the skinned human mesh.
- **Show Skeleton** displays the source skeleton.
- **Show Joint Axes** displays source-joint coordinate axes.
- **Show Gizmos** displays transform gizmos for staging the source and robot in the viewport.
- **Reset** restores the robots offsets in the scene.

Changing visibility does not change the exported joint values. Use gizmos for inspection and reset them before making repeatable visual comparisons.

## Load an existing CSV without retargeting

To inspect an earlier result:

1. Select the robot target that produced the CSV.
2. Click **CSV Motion → Load** and select the CSV.

![Converter load CSV button](images/converter-csv-load-button.png)

Do not click **Retarget** unless you intend to replace the loaded robot motion with a loaded BVH animation.

The CSV loader assumes a playback rate of **120 frames per second**. The CSV itself does not store a frame rate.

## CSV layout and units

The exported header is:

```text
Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,<robot joint names...>
```

- `Frame` is zero-based.
- Root translations are written in centimeters.
- Root rotations are XYZ Euler angles in degrees.
- Actuated robot-joint values are written in degrees.
- Joint columns and their order depend on the selected target robot.

Do not compare columns from different robot targets by position alone; compare their header names first.

## Use a personal config

The interactive viewer only reads a few converter settings, but it uses the same config as batch mode. Keep the tracked default unchanged so updates and examples remain reproducible.

Create `local-configs/` with your file manager or editor if needed, then create
`local-configs/interactive-retargeting.json`:

```json
{
    "import_folder": "assets/motions/bvh",
    "export_folder": "output/interactive",
    "batch_size": 1,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "unitree_h2",
    "retarget_source_facing_direction": "Mujoco",
    "extra_robot_paths": []
}
```

Then run:

```bash
uv run python app/bvh_to_csv_converter.py --config local-configs/interactive-retargeting.json --viewer gl
```

Converter paths, including custom robot paths, are interpreted from the process working directory rather than the config file’s directory. Use repository-root-relative paths and keep the command’s working directory at the repository root.

## If something goes wrong

- If the config is not found, verify the `--config` path and run from the repository root.
- If a target is unknown, remove stale `extra_robot_paths`, click **Reload**, and select a name present in the target list.
- If the BVH loads but retargeting fails, confirm it uses the SOMA skeleton expected by this project.
- If no window opens, confirm `--viewer gl` was used and that the machine has a desktop display. For non-interactive processing, use [batch retargeting](batch-retargeting.md).
- If the file dialog fails on Linux, install Tk and start the viewer again.
- If a saved CSV looks wrong in another tool, check the units above and preserve the header’s joint order.

For more diagnoses, see [Troubleshooting](troubleshooting.md).

## Where to go next

After a new robot retargets representative motions without structural mapping or
scale errors, continue to [IK weight optimizer](ik-weight-optimizer.md). If the
current result is not yet stable enough to tune, return to
[Robot configuration reference](robot-configuration-reference.md) and correct
the underlying configuration first.

[← Previous: Robot configuration reference](robot-configuration-reference.md) · [Documentation home](../README.md) · [Next: Optimize IK weights →](ik-weight-optimizer.md)
