# Batch retargeting

Batch mode recursively converts a folder of SOMA BVH motions to robot CSV files without opening a window. It uses the same converter as the interactive workflow but without the viewer. It uses the current `manifest.json` file for the target robot to define which retargeting configuration will be used. 

## Create a batch config

Keep the tracked default as a known-good example. Create `local-configs/` with your file manager or editor if needed, then create `local-configs/batch-retargeting.json`:

```json
{
    "import_folder": "assets/motions/bvh",
    "export_folder": "output/batch/unitree_h2",
    "batch_size": 10,
    "retargeter": "Newton",
    "retarget_source": "soma",
    "retarget_target": "unitree_h1",
    "retarget_source_facing_direction": "Mujoco",
    "extra_robot_paths": []
}
```

The converter interprets these paths from the process working directory, not relative to the config file. This differs from the optimizer.

## Run the batch

From the repository root:

```bash
uv run python app/bvh_to_csv_converter.py --config local-configs/batch-retargeting.json
```

Expected console progress includes:

```text
[INFO]: Processing batch 1 of ...
[INFO]: Loading ... animations...
[INFO]: Retargeting...
[INFO]: Exporting CSV Files
```

On success, the final message reports the number of animations, total elapsed time, and average seconds per motion.

## How files are discovered and written

Batch mode:

1. Recursively finds files matching `*.bvh` below `import_folder`.
2. Sorts files by file size, largest first.
3. Splits them into groups of `batch_size`.
4. Uses the first motion’s skeleton as the reference skeleton.
5. Retargets each group together based on current configuration from `manifest.json`.
6. Writes a `.csv` for each input while preserving its relative subdirectory and filename stem.

For example:

```text
input:
assets/motions/bvh/Neutral_walk_forward_002__A057.bvh

output:
output/batch/unitree_g1/Neutral_walk_forward_002__A057.csv
```

The export directory and any required subdirectories are created automatically.

## Config fields

- `import_folder` — existing folder to scan recursively. A single file is not accepted by batch mode.
- `export_folder` — non-empty destination folder.
- `batch_size` — positive number of motions retargeted together. Smaller values generally use less GPU memory.
- `retargeter` — currently must be `"Newton"`.
- `retarget_source` — currently `"soma"`.
- `retarget_target` — a target name discovered from robot manifests, such as `unitree_g1`.
- `retarget_source_facing_direction` — coordinate convention used to orient the source; the bundled default is `"Mujoco"`.
- `extra_robot_paths` — optional robot-manifest directory or list of directories. Entries discovered there override bundled targets with the same name.

## Input requirements

All BVH files in one batch run are expected to use the same skeleton topology. The converter compares each animation’s joint count with the first file’s skeleton. A different joint count stops the run with an assertion.

For predictable results:

- Use SOMA-compatible BVH files.
- Separate different skeleton types into different import folders.
- Keep file extensions lowercase `.bvh`, especially on case-sensitive systems.
- Run a small representative folder before processing a large collection.

## Choose a batch size

Start with `10` for a new machine or target. Increase it only after a successful run.

- Lower `batch_size` if the GPU runs out of memory.
- Higher values can improve throughput by processing more motions together.
- `batch_size` must be greater than zero.
- Batch boundaries do not change output paths.

## Output CSV format

Each CSV contains:

```text
Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,<robot joint names...>
```

Translations are centimeters. Root XYZ rotations and robot-joint values are degrees. The CSV does not store frame rate; the project loader plays CSV files at 120 FPS by default.

## Reruns and failure recovery

Batch mode has no resume database and does not skip existing CSV files. A rerun writes the same paths again.

If a run fails:

1. Read the first error above the stack trace.
2. Keep already written CSV files for inspection, but treat the output set as incomplete.
3. Fix the input, config, custom robot asset, or batch size.
4. Rerun the same command. Existing files at matching paths are overwritten.
5. Confirm the final success message and compare the number of input BVHs with output CSVs.

If you need to preserve a partial or previous result, change `export_folder` before rerunning.

Typical recovery choices:

- **No BVH files found:** correct `import_folder` or extension case.
- **Unexpected number of joints:** move the incompatible BVH to a separate corpus.
- **GPU memory error:** lower `batch_size`.
- **Unknown target:** correct `retarget_target` or `extra_robot_paths`.
- **Invalid solver:** set `retargeter` to `"Newton"`.
- **Bad output from every motion:** verify `retarget_source_facing_direction` and target selection before running again.

## Validate a batch

After completion:

1. Compare recursive input and output file counts.
2. Check that nested directories were mirrored.
3. Open several CSVs from the beginning, middle, and end of the sorted workload.
4. Include small, large, and difficult motions in the sample.
5. Inspect them with the [interactive viewer](interactive-retargeting.md).

## Where to go next

If the complete output count and visual spot checks pass, archive the exact converter config and robot asset version with the batch results. If any check fails, continue to [Troubleshooting](troubleshooting.md) before rerunning the full dataset.

[← Previous: Compare and tune](compare-and-tune.md) · [Documentation home](../README.md) · [Next: Troubleshooting →](troubleshooting.md)
