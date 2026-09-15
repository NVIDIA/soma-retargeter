[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Version](https://img.shields.io/github/v/tag/NVIDIA/soma-retargeter?sort=date&label=version)](https://github.com/NVIDIA/soma-retargeter/tags)

# SOMA Retargeter

![SOMA Retargeter banner](documentation/images/banner_allrobots.gif)

SOMA Retargeter converts human motion in a SOMA-skeleton BVH file into joint animation for a humanoid robot. The result is a CSV file that can be inspected, compared, or passed to downstream robot-control and simulation tools.

Start with the bundled motion and robot, confirm that the software works on your computer, and then move on to your own files and tuning.

> **Important:** SOMA Retargeter is in active beta, so APIs and features may change without notice. Generated motion is kinematic; validate its safety, feasibility, and controller compatibility in simulation before hardware use.

## What the retargeter does

The human source and target robot have different proportions and joints. SOMA  
Retargeter does the following:

1. reads a SOMA-base skeleton BVH animation;
2. retargets human scale joints to the robot's proportions;
3. solves inverse kinematics for each frame;
4. stabilizes contacts and clamps configured joint limits;
5. writes the robot root pose and actuated joint values to CSV.

![Retargeting pipeline from BVH to robot CSV](documentation/images/retargeting-pipeline.png)


The repository includes:

- 10 example BVH motions and matching CSV results for all five bundled robots;
- an interactive viewer for loading, retargeting, inspecting, and saving one motion;
- a headless mode for converting folders of motions;
- configuration and model assets for `unitree_g1`, `unitree_h2`, `booster_t1`, `agibot_x2ultra`, and `agibot-a3t3`;
- tools for configuring another robot and tuning and optimizing inverse-kinematics weights.
- 15 BVH motions for inverse-kinematics weights optimization

## Motion data and credit
The bundled BVH motion examples and optimizer inputs are provided with permission from [Bones](https://bones.studio/) and are drawn from the [BONES-SEED dataset](https://bones.studio/datasets/seed). We thank the Bones team for supporting this release. See BONES-SEED for the broader motion collection and licensing options.


## Documentation structure

Documentation from setup through batch processing:

![Documentation workflow from setup through batch processing](documentation/images/workflow-overview.png)

1. **[Installation](documentation/installation.md)** — install prerequisites, create the environment, download assets, and verify the setup.
2. **[First retarget](documentation/first-retarget.md)** — convert one bundled BVH in the interactive viewer and save a CSV.
3. **[Robot configurator](documentation/robot-configurator.md)** — begin the guided process of adding a robot. The later tutorial uses Unitree H1 as the new-robot example;
4. **[Robot configuration reference](documentation/robot-configuration-reference.md)** — understand manifests, model descriptions, retargeting objectives, scaling, and post-processing settings.
5. **[Contact detection & foot-plant correction configuration](documentation/foot-contact.md)** — detailed description of contact detection and foot planting settings.
6. **[Interactive retargeting](documentation/interactive-retargeting.md)** — learn viewer controls and work with SOMA BVH files.
7. **[IK weight optimizer](documentation/ik-weight-optimizer.md)** — optimize for better objective weights.
8. **[Compare and tune](documentation/compare-and-tune.md)** — evaluate results and refine a robot configuration.
9. **[Batch retargeting](documentation/batch-retargeting.md)** — convert a directory tree without opening a viewer.
10. **[Code overview](documentation/code-overview.md)** — review the main application entry points and `soma_retargeter` package modules.
11. **[Troubleshooting](documentation/troubleshooting.md)** — diagnose installation, viewer, input, GPU, and output problems.

## Inputs and outputs

### Input: SOMA BVH

A `.bvh` file contains a skeleton hierarchy and one pose per animation frame. The retargeter currently expects the source hierarchy and naming used by the SOMA base skeleton. An arbitrary BVH exported from another character or motion capture package may load incorrectly or fail because its skeleton is different.

For the first run, use `assets/motions/bvh/Neutral_walk_forward_002__A057.bvh`. Keeping the input known removes one source of incertainty while you test the installation.

### Output: robot CSV

The `.csv` output stores the root pose and actuated robot joint values over time. Joint columns correspond to the selected robot model. Two target robots can have different columns, even when they were generated from the same BVH.

The bundled CSV files under `assets/motions/csv/` are organized by robot and provide reference results for all five bundled robots. Select the folder matching your target robot. Your result should have the same general motion, but exact values can change when configuration or dependency versions change.

## Bundled robot targets

Use these exact target names when using bundled robots:

- `unitree_g1` — the default target and the best choice for the first tutorial;
- `unitree_h2`;
- `booster_t1`;
- `agibot_x2ultra`;
- `agibot-a3t3`.

The interactive viewer lists discovered targets in its robot selector. Headless conversion reads the target from the `retarget_target` field in the converter configuration.

## Command convention in this guide

Run commands from the repository root. The documentation uses `uv` because it creates and manages the project environment without requiring you to activate it manually:

```bash
uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl
```

Paths with forward slashes work in PowerShell and in Linux shells. If a path you provide contains spaces, enclose it in quotes.

## Related Work

SOMA Retargeter is a support tool within the SOMA ecosystem for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

## Acknowledgments

This project draws inspiration and builds upon excellent open-source work, including:
* [GMR](https://github.com/YanjieZe/GMR) - General Motion Retargeting
* [PyRoki](https://pyroki-toolkit.github.io/) - A Modular Toolkit for Robot Kinematic Optimization

## License

NVIDIA-authored source code is licensed under [Apache-2.0](LICENSE).

The bundled motion data under `assets/motions/bvh/`, `assets/motions/optimizer/`, and `assets/motions/csv/` is licensed separately under the [NVIDIA Sample Data Evaluation License](assets/motions/LICENSE.txt). It is not covered by Apache-2.0.

The repository directly includes third-party robot-description and mesh assets:

- Unitree G1 and Unitree H2 assets are licensed under BSD-3-Clause. See [`licenses/unitree-LICENSE.txt`](licenses/unitree-LICENSE.txt).
- Booster Robotics T1 assets are licensed under BSD-3-Clause. See [`licenses/booster-LICENSE.txt`](licenses/booster-LICENSE.txt).
- AGIBot X2 and AGIBot A3 assets are licensed under Mulan PSL v2. See [`licenses/agibot-LICENSE.txt`](licenses/agibot-LICENSE.txt) and the pinned upstream [X2 source](https://github.com/AgibotTech/agibot_x2_urdf/tree/77f43eb0904dae4c48ccd9154fee824f8ffd4d38/X2_URDF-v1.4.0) and [A3 source](https://github.com/AgibotTech/A3-A3U-robot-model/tree/589f508ff357447c610a3f3004419035ddc8f153/a3_t3d0).

The applicable robot-asset license texts and path scopes are also included in the top-level [LICENSE](LICENSE). License copies for Python dependencies are available in [`licenses/`](licenses/). Python dependencies are installed from package registries rather than copied into this repository.

Some manifest-driven workflows can ask NVIDIA Newton to retrieve a separately hosted robot asset at runtime. Those assets remain subject to their upstream terms and are not bundled by SOMA Retargeter unless they are present in this repository.

## Where to go next

Go to **[Installation](documentation/installation.md)** to prepare Python, Git LFS, the GPU driver, and the project environment. If you already completed installation, run the verification command there before continuing to **[First retarget](documentation/first-retarget.md)**.

---

[Next: Installation →](documentation/installation.md)
