# Troubleshooting

Start with the exact command for the workflow you are running:

```bash
# Interactive
uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl

# Batch
uv run python app/bvh_to_csv_converter.py --config local-configs/batch-retargeting.json 

# Optimizer
uv run --with matplotlib python app/tools/ik_weight_optimizer.py --config local-configs/ik-weight-optimizer.json
```

Run these commands from the repository root.

## Installation and startup

### Git LFS assets are pointers or fail to load

Symptoms include tiny asset files, parser errors in BVH or robot files, or unexpected text beginning with a Git LFS URL.

Recover with:

```bash
git lfs pull
uv sync
```

Then rerun the original command.

### `uv` cannot find or start Python

The project requires Python 3.12 or newer. Refresh the environment:

```bash
uv sync
uv run python --version
```

The version command should report Python 3.12 or newer.

### GPU or Warp initialization fails

Confirm the machine has a supported NVIDIA GPU and current NVIDIA driver. The project requires driver 545 or newer for its CUDA 12 runtime. A local CUDA Toolkit is not required.

Close other GPU-heavy applications and retry with a smaller batch. If initialization fails before any motion loads, changing the motion batch will not help; fix the driver or GPU environment first.

### `imgui-bundle` installation fails on Windows

Install the current Microsoft Visual C++ Redistributable, then rerun:

```bash
uv sync
```

## Config and path errors

### Main config not found

The converter prints:

```text
[ERROR]: Main config json file not found: ...
```

Check spelling, use a `.json` file, and run from the repository root. Converter config paths and values such as `import_folder`, `export_folder`, and `extra_robot_paths` are interpreted from the process working directory.

The optimizer behaves differently: `input_folder`, `output_folder`, and optimizer `extra_robot_paths` are resolved relative to the optimizer config file’s directory.

### Optimizer rejects `input_folder`

The bundled optimizer config intentionally sets `input_folder` to an empty string. The optimizer requires a non-empty path to a `.bvh` file or a directory containing at least one `.bvh`.

Create a personal config and set, for example:

```json
"input_folder": "../assets/motions/optimizer"
```

This value is correct for a config stored in `local-configs/`.

### No BVH files found

The converter batch scanner and optimizer search recursively for `*.bvh`. Confirm:

- The folder exists.
- It contains real Git LFS content.
- Extensions are lowercase on case-sensitive systems.
- The converter path is repository-root-relative.
- The optimizer path is config-file-relative.

### Export folder error

The converter requires a non-empty `export_folder`; it creates a missing folder automatically. Verify the parent is writable and the value is not `""`.

The optimizer similarly creates:

```text
<output_folder>/optimized_<target_type>/
```

## Interactive viewer

### No window appears

Confirm the command uses:

```bash
--viewer gl
```

The GUI requires a desktop session. Remote or headless machines should use `--viewer null` or no `--viewer` argument for batch conversion.

### File dialog does not open on Linux

Install Tk:

```bash
sudo apt-get install python3.12-tk
```

Restart the viewer afterward.

### Retarget or Save is disabled

- **Retarget** is disabled until a BVH has been loaded.
- **Save** is disabled until a robot CSV has been loaded or a BVH has been retargeted.

### Unknown robot target

The error lists currently available target names. Check `retarget_target`, then check every `extra_robot_paths` entry.

Custom paths are scanned recursively for `manifest.json`. A custom target with the same name overrides the bundled one. If an override is accidental, remove that extra path and restart the process.

### Source and robot point in unexpected directions

Use the bundled `"retarget_source_facing_direction": "Mujoco"` as the baseline. Reset viewer gizmos before judging the result. If a custom source convention is required, change only that field and compare the same motion and frame.

### CSV plays at the wrong speed

CSV files do not store sample rate. The loader assumes 120 FPS. If the source or consuming application expects another rate, handle resampling explicitly outside this file format rather than changing joint values.

## Batch processing

### Unexpected number of joints

All motions in one run are loaded against the first BVH skeleton. A motion with a different joint count stops the run.

Separate incompatible skeletons into different input folders. Do not delete joints or reorder channels merely to bypass the assertion.

### Out of GPU memory

Lower `batch_size` in the converter config, for example:

```json
"batch_size": 2
```

Already written CSVs can remain, but rerunning will overwrite matching output paths.

### Some outputs exist after failure

This is expected: CSVs are written batch by batch. There is no transactional rollback or resume index.

Fix the failure and rerun the full command. Verify the final success message and output count before treating the dataset as complete.

### Existing outputs are overwritten

This is expected. Batch mode does not skip existing files. Use a new `export_folder` when preserving an earlier run matters.

## IK optimizer

### Matplotlib is missing

Optimization can continue and still writes JSON diagnostics and the optimized config. Plot rendering is skipped.

Rerender later without changing project dependencies:

```bash
uv run --with matplotlib python app/tools/ik_weight_optimizer.py --plot output/ik-optimizer/optimized_unitree_h1/optimizer_metrics.json
```

### Optimizer runs out of GPU memory

Add smaller values to the personal optimizer config:

```json
{
    "motion_batch_size": 10,
    "fk_frame_batch_size": 25
}
```

These settings control evaluation batching; they do not change the optimized parameter type.

### Every update is rejected or held

Read the `REJECT - PROBE` and `HOLD` messages:

- `objective_not_improved` means the candidate did not clear the line-search improvement threshold.
- Guard messages identify the motion and metric that regressed too far.
- In `auto` mode, strict gating may transition to penalty mode after repeated holds.

Enable probe diagnostics in a new run:

```json
"save_probe_diagnostics": true
```

Then inspect `optimizer_probe_diagnostics.csv`. Do not immediately disable guards; first determine whether a hard motion is exposing a real regression.

### Optimization stops before `outer_iterations`

Early convergence is expected after too many consecutive iterations without an accepted improvement. The optimizer still writes its final accepted config and diagnostics.

### One motion makes the optimizer fail

The optimizer does not skip malformed or incompatible clips. Move the clip out of the corpus, record the reason, and rerun. Keep corpus changes explicit because changing the corpus changes the optimization problem.

### Output config does not change motion

The optimizer only writes a candidate retargeter JSON. The converter continues
using the baseline target from its registered search roots until a later custom
robot path overrides that target name with the candidate. Follow
[Compare and tune](compare-and-tune.md) without editing the baseline assets.

## Output quality

### CSV columns look unfamiliar

The fixed leading columns are frame, root translation, and root rotation. Remaining columns are actuated joints for the selected robot, in target-specific order.

Translations are centimeters; rotations and joint values are degrees. Always retain and read the header.

### Metrics improve but motion looks worse

Optimizer metrics cover solved-IK tracking and smoothness. They do not establish controller stability, physical feasibility, or subjective quality.

Review regressed rows in `optimizer_improvement_ranking.csv`, then compare the exact same source motion, output frame, robot override, and camera view. Reject the candidate if severe visual regressions remain.

### Baseline and candidate cannot be compared

Confirm both runs used:

- The same input file list.
- The same target model and scaler.
- The same source facing direction.
- Different output folders.
- Exactly one intended difference: the candidate retargeter config.

If more than one variable changed, regenerate the pair before drawing conclusions.

## Where to go next

Return to the workflow that failed after applying one correction at a time:

1. [Interactive retargeting](interactive-retargeting.md)
2. [IK weight optimizer](ik-weight-optimizer.md)
3. [Compare and tune](compare-and-tune.md)
4. [Batch retargeting](batch-retargeting.md)
5. Troubleshooting

[← Previous: Batch retargeting](batch-retargeting.md) · [Documentation home](../README.md)
