# Compare and tune

SOMA Retargeter does not include a dedicated “compare” command. Comparison is a repeatable workflow. This page continues the H1 tutorial and uses an optimized `unitree_h1` retargeter as the candidate. 

Substitute the robot target and filenames produced by your optimizer run when working with another robot.

## Keep the experiment controlled

Change only the retargeter config being tested. Keep these fixed:

- Target robot and model files.
- Human-to-robot scaler config.
- Input BVH corpus.
- Source facing direction.
- Converter and batch settings.
- Software revision and GPU environment.

Use separate output folders. Never overwrite the baseline config file with the optimized candidate.

## 1. Generate a baseline

Create `local-configs/` with your file manager or editor if needed, then create
`local-configs/compare-baseline.json`:

```json
{
    "import_folder": "assets/motions/bvh",
    "export_folder": "output/compare/unitree_h1-baseline",
    "batch_size": 10,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "unitree_h1",
    "retarget_source_facing_direction": "Mujoco",
    "extra_robot_paths": ["assets/robotics"]
}
```

Run:

```bash
uv run python app/bvh_to_csv_converter.py --config local-configs/compare-baseline.json
```

Expected output is one CSV per input BVH under:

```text
output/compare/unitree_h1-baseline/
```

The input folder hierarchy and base filenames are preserved.

## 2. Copy the new optimized config file

Do not replace the baseline H1 retargeter config. The `manifest.json` file can choose which retargeter config file to use and you can have multiple versions in the robot's `configs` folder. 

1. Copy the new optimized config file :
   from: `output/ik-optimizer/optimized_unitree_h1/optimized_soma_to_unitree_h1_retargeter_config.json`
   to: `assets/robotics/unitree/unitree_h1/configs/optimized_soma_to_unitree_h1_retargeter_config.json`
2. Modify the `assets/robotics/unitree/unitree_h1/manifest.json` file: 

```json
{
    "name": "unitree_h1",
    "desc": {
        "urdf_path": "desc/h1.urdf"
    },
    "retarget_configs": {
        "soma": "configs/optimized_soma_to_unitree_h1_retargeter_config.json"
    }
}
```

## 3. Generate candidate outputs

Create `local-configs/compare-candidate.json`:

```json
{
    "import_folder": "assets/motions/bvh",
    "export_folder": "output/compare/unitree_h1-candidate",
    "batch_size": 10,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "unitree_h1",
    "retarget_source_facing_direction": "Mujoco",
    "extra_robot_paths": ["assets/robotics"]
}
```

`extra_robot_paths` is registered in list order from the repository-root working directory. `assets/robotics` supplies the baseline H1 target.

Run:

```bash
uv run python app/bvh_to_csv_converter.py --config local-configs/compare-candidate.json
```

Expected output is the same set of relative CSV paths under:

```text
output/compare/unitree_h1-candidate/
```

Before reviewing quality, compare the two output file lists. A missing candidate file means the runs are not comparable yet.

## 4. Compare the same motion visually

Start the viewer:

```bash
uv run python app/bvh_to_csv_converter.py --viewer gl
```

Load one source BVH and its matching CSV from `output/compare/unitree_h1-baseline/`. Do **not** click **Retarget**, because that would replace the loaded CSV with a newly computed in-memory result. 

Record the motion name and frame or playback time for every observation.

Keep the same BVH and load the matching CSV from `output/compare/unitree_h1-candidate/`. 

Use the same playback time, speed, visibility, and camera view to compare with previous recording. 

Inspect:

- Foot sliding during planted phases.
- Ground penetration and floating feet.
- Root height, lean, and heading.
- Knee and elbow hyperextension or joint-limit clamping.
- Arm and elbow endpoint placement during reaches and contacts.
- Sudden joint changes around turns or impacts.
- First-frame settling and end-of-motion behavior.

## 6. Decide and iterate

Accept a candidate only when:

- Aggregate metrics improve enough to matter.
- Regressed clips are understood and acceptable.
- Visual inspection confirms the intended improvement.
- No new severe jitter, penetration, or implausible posture appears.

If the candidate is not acceptable:

1. Keep the baseline unchanged.
2. Change one optimizer setting or corpus choice.
3. Write to a new optimizer output folder.
4. Build a new local robot override.
5. Repeat the exact baseline/candidate workflow.

Useful single-variable experiments include adjusting effector priorities, reducing the learning rate, changing smoothness balance, or adding motions that expose the observed failure. Avoid changing the scaler, robot model, corpus, and IK priorities in the same comparison; the result would not identify which change helped.

## Comparison limitations

- CSV files contain joint values, not quality metrics.
- CSV playback assumes 120 FPS because frame rate is not stored in the file.
- A textual CSV diff shows that values changed, not whether motion quality improved.
- The interactive viewer loads one robot CSV at a time; side-by-side assessment requires repeatable camera and time notes or separately captured images.
- Optimizer loss measures kinematic tracking and smoothness. It does not prove physical feasibility or controller performance.

## Where to go next

Once one configuration passes the comparison checklist, use it for a small
[batch retargeting](batch-retargeting.md) trial. Preserve the accepted robot
asset directory and optimizer evidence so the larger batch is reproducible.

[← Previous: IK weight optimizer](ik-weight-optimizer.md) · [Documentation home](../README.md) · [Next: Batch retargeting →](batch-retargeting.md)
