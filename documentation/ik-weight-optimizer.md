# IK weight optimizer

**Important note: Experimental feature**

The IK weight optimizer is an experimental calibration tool. It writes a new retargeter configuration with candidate IK weights based on kinematic tracking and smoothness metrics; compare the result visually with the baseline before adoption.

The IK weight optimizer tunes how strongly a robot retargeter tries to match each source effector. Use it after the standard retargeter works but consistently favors the wrong body parts—for example, good hands but poor feet, or good root tracking but weak limb tracking.

The optimizer is a corpus-level calibration tool, not a single-motion converter. It repeatedly retargets a set of BVH motions, measures solved robot poses against the scaler targets, proposes new IK weights, and accepts only updates that satisfy its objective and clip policy.

## The three inputs

A useful optimization run depends on three groups of repository content:

1. **Motions:** `assets/motions/optimizer/` is the representative BVH corpus. It should contain varied motions that exercise locomotion, turns, reaches, impacts, and full-body movement. Or, if you need to train a robot for specific motions, you can optimize for another set of representative motions.
2. **Robot assets:** the selected target’s discovered manifest, model description, scaler config, and base retargeter config. For the H1 tutorial these are under `assets/robotics/unitree/unitree_h1/`.
3. **Optimizer:** `app/tools/ik_weight_optimizer.py` and its JSON run config at `app/tools/assets/ik_optimizer/ik_weight_optimizer_config.json`.

The bundled optimizer config has:

```json
"input_folder": ""
```

The optimizer validates this field as a non-empty string, so the default config **cannot run unchanged**. The 15 motions under `assets/motions/optimizer/` are a useful first corpus for general H1 movement; replace or extend them later with motions representative of the robot's intended use.

## What changes and what stays fixed

The optimizer changes only:

- `t_weight` values in the target retargeter’s `ik_match_table`.
- `r_weight` values in the same table.

The final values are the base weights multiplied by learned per-effector scales and rounded to three decimal places. By default, matching `Left*` and `Right*` effectors are constrained to remain symmetric.

The optimizer does **not** change:

- The robot model or kinematic structure.
- The human-to-robot scaler config or its correspondence offsets.
- Joint limits.
- The source BVH files.
- The base retargeter file in the bundled robot assets.
- Post-processing settings in the deployable output.

`retargeter_overrides` can temporarily disable processing stages while candidates are evaluated. Those evaluation-only overrides are not written into the final deployable retargeter config. For best results in the optimisation process, it is recommended to set the overrides to false.

## Create a run config

Do not edit the tracked default for an experiment. Create `local-configs/` with
your file manager or editor if needed, then create
`local-configs/ik_weight_optimizer_config.json`:

```json
{
    "target_type": "unitree_h1",
    "input_folder": "../assets/motions/optimizer",
    "output_folder": "../output/ik-optimizer",
    "extra_robot_paths": ["../assets/robotics"],
    "outer_iterations": 50,
    "ik_weight_update": {
        "learning_rate": 0.075,
        "max_scale": 4.0,
        "protect_root_position": true,
        "protect_root_rotation": false,
        "root_rotation_min_scale": 0.7,
        "hard_clip_emphasis": 1.0
    },
    "clip_guard": {
        "mode": "auto",
        "penalty_balance": 1.0,
        "penalty_tail_weight": 0.5,
        "max_clip_frame_loss_p95_regression": 0.15,
        "max_aligned_frame_loss_p95_regression": 0.30,
        "catastrophic_accel_spike_regression": 1.5
    },
    "objective": {
        "smoothness_balance": 0.08
    },
    "plot": {
        "enabled": true,
        "live": false
    },
    "effector_priorities": {
        "position_weights": {
            "Hips": 5.0,
            "Chest": 1.2,
            "LeftFoot": 3.0,
            "RightFoot": 3.0
        },
        "rotation_weights": {
            "Hips": 1.2,
            "Chest": 1.0
        }
    },
    "retargeter_overrides": {
        "enable_post_processing": false,
        "enable_contact_processing": false,
        "enable_post_smoothing": false
    }
}
```

Optimizer paths are resolved relative to the optimizer config’s directory. In this example:

- `../assets/motions/optimizer` resolves to the bundled corpus.
- `../output/ik-optimizer` keeps generated artifacts separate from source assets.
- `../assets/robotics` lets the optimizer discover the H1 target created by the Robot Configurator.

For a smoke test, set `outer_iterations` to `2`. Restore the planned iteration count for a real calibration run.

## Run optimization

There are 2 ways to run the optimization process. One with no graphic reports and one with graphic reports. In both cases, a set of report output files are generated as diagnostic files. 

If you only want to optimize IK weights and write the JSON/CSV diagnostics only, run the following:  

```bash
uv run python app/tools/ik_weight_optimizer.py --config local-configs/ik_weight_optimizer_config.json
```

For a strictly no-graph run, set the run config's plot block to:

```json
"plot": {
    "enabled": false,
    "live": false
}
```

With plotting disabled, the optimizer still writes the optimized retargeter config, `optimizer_metrics.json`, and the motion improvement ranking. It skips the dashboard PNG and the flattened iteration-metrics CSV.

Expected startup messages report:

- The number of BVH motions found recursively.
- Skeleton loading.
- Whether target symmetry is enabled.
- The initial tracking, smoothness, frame-loss, position, rotation, and reachability metrics.

Each iteration then reports `ACCEPT`, `REJECT - PROBE`, or `HOLD`:

- **ACCEPT** means a candidate improved the objective and passed the active clip policy.
- **REJECT - PROBE** means the line search will try a smaller update.
- **HOLD** means every probe was rejected and the current accepted weights were retained.

An early `CONVERGED` message is normal when repeated probes cannot improve the accepted state. In `auto` clip-guard mode, the optimizer can first transition from strict `gate` behavior to softer `penalty` behavior.

### Generate the dashboard graph

Graphs are useful once the basic command runs successfully. The dashboard makes it much easier to see whether the optimizer is improving steadily, getting stuck, or trading one kind of accuracy for another.

![Example IK optimizer dashboard](images/ik-optimizer-dashboard-example.png)

The dashboard shows:

- optimization loss, tracking loss, and smoothness penalty;
- IK weight update size per accepted iteration;
- position error in meters;
- rotation error in radians;
- motion smoothness and acceleration signals;
- mean, p95, and worst-frame tracking loss.

To generate the dashboard during a run, keep plotting enabled in the config:

```json
"plot": {
    "enabled": true,
    "live": false
}
```

Then run with Matplotlib available only for this command:

```bash
uv run --with matplotlib python app/tools/ik_weight_optimizer.py --config local-configs/ik_weight_optimizer_config.json
```

`--with matplotlib` does not modify the project dependency files. It only makes Matplotlib available to this `uv run` invocation. The optimizer writes `optimizer_dashboard.png` beside the optimizer metrics when plotting succeeds.

Keep `"live": false` for most runs. Set `"live": true` only when you want the dashboard image refreshed during optimization; that adds repeated plot rendering and file-writing overhead.

## Expected output files

For `target_type: "unitree_h1"` and the config above, outputs are written under:

```text
output/ik-optimizer/optimized_unitree_h1/
```

The directory contains:

- `optimized_soma_to_unitree_h1_retargeter_config.json` — deployable retargeter config with optimized IK weights.
- `optimizer_metrics.json` — complete run metadata, accepted history, per-motion metrics, final scales, and artifact paths.
- `optimizer_improvement_ranking.csv` — each corpus motion ranked by baseline-to-final tracking-loss change.
- `optimizer_iteration_metrics.csv` — flat per-iteration metrics, when plotting is enabled.
- `optimizer_dashboard.png` — six-panel dashboard, when plotting is enabled and Matplotlib is available.
- `optimizer_probe_diagnostics.csv` — optional probe-level accept/reject details, only when `"save_probe_diagnostics": true`.

The optimizer does not produce retargeted motion CSV files. Use the optimized config in a custom robot override, then run the converter as described in [Compare and tune](compare-and-tune.md).

Alternatively, you can copy the `optimized_soma_to_unitree_h1_retargeter_config.json` optimized config file into the robot's `configs` folder and update the `manifest.json` file of the robot to point to the `optimized_soma_to_unitree_h1_retargeter_config.json` retargeter file. Reloading robots in the converter will use the updated manifest and retarget with the optimized config file. 

## Read the results

Start with these checks:

1. In `optimizer_dashboard.png`, confirm tracking loss trends downward rather than improving once and then flattening through repeated holds.
2. Check position and rotation error separately; a lower combined objective can still trade one against the other.
3. Check smoothness and worst-frame signals for spikes.
4. Open `optimizer_improvement_ranking.csv`. Review both the best improvements and every row marked `regressed`.
5. Inspect the optimized config’s `ik_match_table`; confirm weights remain plausible and symmetric where expected.
6. Validate representative motions visually before adopting the config.

Lower loss is evidence, not proof that the motion is better for control or deployment.

## Render a dashboard later

If the run completed without Matplotlib, the JSON and CSV diagnostics still remain. Render from the full-precision JSON:

```bash
uv run --with matplotlib python app/tools/ik_weight_optimizer.py --plot output/ik-optimizer/optimized_unitree_h1/optimizer_metrics.json
```

Or choose the output PNG explicitly:

```bash
uv run --with matplotlib python app/tools/ik_weight_optimizer.py --plot output/ik-optimizer/optimized_unitree_h1/optimizer_metrics.json output/ik-optimizer/optimized_unitree_h1/review-dashboard.png
```



## IK Optimizer Best Practices

The motions in `assets/motions/optimizer/` are a good place to start because they exercise a range of full-body behaviors. They are not guaranteed to be the best corpus for every robot. Each robot has a different body shape, DOF layout, joint range, end-effector placement, and intended task. A motion set that is useful for one humanoid can be too easy, too difficult, or simply not representative for another.

Good optimizer inputs should be close to what the robot is expected to do. For a robot that needs stable walking and turning, choose motions with gait changes, direction changes, and foot plants. For a robot that needs manipulation, include motions with reaches, arm coordination, and torso motion. Avoid optimizing around motions the robot cannot realistically perform, because the optimizer may learn weights that chase impossible targets and make ordinary motions worse.

After you understand the basic optimizer workflow, use motion selection as a second-stage refinement step. If the bundled corpus does not improve results for your robot, test the motions one at a time before running another large optimization:

1. Create a temporary input folder with one BVH from `assets/motions/optimizer/`.
2. Run a short optimization for that single motion.
3. Review `optimizer_dashboard.png` and `optimizer_improvement_ranking.csv`.
4. Visually compare the baseline and optimized retargeting in the converter.
5. Repeat with the next motion and keep notes about which motions improve, stay neutral, or regress.

This exercise helps you build a better robot-specific corpus. Keep motions that produce meaningful improvements, remove motions that consistently regress or force unnatural poses, and then run one combined optimization on the curated set. The goal is not to find one perfect motion. The goal is to collect a small set of representative motions that pull the weights in a useful direction for the robot's real use case.

You can use the same approach with your own SOMA-compatible BVH motions. Prepare or source motions that match the robot's intended tasks, test them individually, then combine the strongest examples into a single optimization run. If you need a broader library of SOMA-ready motions, the [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed) provides a large collection of annotated humanoid motions in SOMA formats. The dataset page may require accepting its access terms before downloading files. The [SEED interactive viewer](https://seed-viewer.bones.studio/) is useful when you want to browse motions visually before choosing candidates.

Treat IK optimization as an iterative calibration process. Run an experiment, inspect the metrics, watch the robot, adjust the corpus, and rerun with a clean output folder. Lower loss is useful evidence, but the final decision should always include visual validation on meaningful motions for the robot.

## Recover from a failed run

- Keep the failed output directory until you have read the last console error.
- Fix the personal run config or corpus; do not patch the bundled robot config to make the run continue.
- Use a new `output_folder` for materially different experiments so results are not confused.
- If GPU memory is exhausted, lower `motion_batch_size` and `fk_frame_batch_size` in the run config.
- If one malformed motion stops loading, move it out of the corpus, record why, and rerun the complete corpus. The optimizer does not skip failed clips.
- If every iteration holds, inspect guard messages and probe diagnostics before relaxing thresholds. A hold can be correctly preventing regression.



## Where to go next

Keep the generated optimizer directory unchanged as evidence, then use [Compare and tune](compare-and-tune.md) to install the candidate in a non-destructive robot override and compare it with the baseline on identical motions.

[← Previous: Interactive retargeting](interactive-retargeting.md) · [Documentation home](../README.md) · [Next: Compare and tune →](compare-and-tune.md)
