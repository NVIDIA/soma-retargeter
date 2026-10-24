# Optional joint continuity

Add `joint_continuity_weights` to a robot's retargeter configuration to penalize
large changes in selected scalar joint coordinates between input frames:

```json
{
  "joint_continuity_weights": {
    "left_shoulder_pitch_joint": 2.0,
    "left_shoulder_roll_joint": 2.0,
    "left_shoulder_yaw_joint": 2.0,
    "left_elbow_joint": 2.0,
    "left_wrist_roll_joint": 2.0,
    "left_wrist_pitch_joint": 2.0,
    "left_wrist_yaw_joint": 2.0,
    "right_shoulder_pitch_joint": 2.0,
    "right_shoulder_roll_joint": 2.0,
    "right_shoulder_yaw_joint": 2.0,
    "right_elbow_joint": 2.0,
    "right_wrist_roll_joint": 2.0,
    "right_wrist_pitch_joint": 2.0,
    "right_wrist_yaw_joint": 2.0
  }
}
```

Names must match the target's revolute/prismatic joint names. Omit this field or
use `{}` to keep the existing behavior. No bundled robot preset is changed.
Nonpositive/nonfinite weights, ambiguous/unknown names, and non-scalar joints
are rejected when the objective is constructed.

For each selected joint, the additional residual is
`weight * (q_current - q_previous_base_solution)`. The least-squares cost squares
this residual. The reference is copied once before each frame's base IK solve
and held fixed throughout all solver iterations and candidate evaluations.
Each animation in the batch has its own reference; expanded candidate batches
use Newton's `problem_idx` mapping. Initialization/stabilization and the first
output frame run without this prior. Every `execute()` builds a new objective,
so state is not carried from an earlier animation batch.

This is a soft **per-frame** prior, not a hard speed limit. Its effect depends
on the input sample rate: retune the weights when resampling. Coordinates are
in radians (revolute) or metres (prismatic); this API intentionally excludes
floating and quaternion joints. Bounded joint coordinates are not wrapped at
2π. The existing joint-limit, contact and limb-stabilization stages remain in
place; the new prior does not add collision avoidance or dynamic feasibility.
It references the previous **base-pass** solution, not the final post-processed
or clamped output. Select joints unaffected by downstream corrections (such as
arms in the current A3 setup); it is not an output-continuity guarantee for feet
or other coordinates subsequently changed by post-processing.

Larger weights reduce abrupt changes but can introduce lag and tracking error.
Validate these tradeoffs on representative motions before changing a preset.

## A3 single-clip experiment

This reproduces the issue with the **bundled upstream `agibot-a3t3` URDF and
SOMA calibration**, unchanged from `cbdb2b4`, using Newton 1.3.0 and Warp 1.16.0.
Input: BONES-SEED SOMA Uniform `221129/reach_jump_R_004__A077.bvh`,
424 source frames at 120 Hz. Both runs use the upstream defaults, including
24 IK iterations, the static smooth filter, joint limits and contact/post-processing.
The only retargeting change is adding the above 14 arm residuals with weight 2.

To reproduce with a licensed local copy of the input and downloaded upstream
assets (Git LFS):

```bash
PYTHONPATH=. python docs/examples/compare_joint_continuity.py \
  --input /path/to/reach_jump_R_004__A077.bvh \
  --output /path/to/new-comparison
```

The script copies configuration files into a new comparison directory and points
at the original bundled URDF. It refuses to overwrite an existing output. Use
`--device cpu` if CUDA is unavailable; the reported single-clip run used CUDA.

Peak absolute adjacent-frame joint change in **0.90–1.05 seconds**, measured
between source frames 108 through 126 (inclusive), in degrees per input frame:

| Right joint | Baseline | Continuity |
|---|---:|---:|
| Shoulder pitch | 11.18° | 2.93° |
| Shoulder roll | 14.70° | 7.07° |
| Shoulder yaw | 17.44° | 6.15° |
| Wrist roll | 12.41° | 4.27° |
| Wrist pitch | 10.55° | 2.53° |
| Wrist yaw | 6.14° | 1.17° |

The abrupt shoulder/wrist changes occur during fast arm lowering. The existing
`smooth_joint_filter` penalizes proximity to limits; it is not a previous-frame
continuity objective. Adding the prior reduces these coordinate changes but
introduces visible lag. It does not establish improved target tracking,
calibration, reachability or dynamics. Both outputs have 424 finite frames.
Existing collision-related settings remain unchanged; no claim of collision
freedom is made. The weight is a single-clip example, not a recommended default
for all motions, robots or sample rates.

[Before/after videos, numeric measurements and credits](https://github.com/huster-wgm/soma-retargeter/blob/c1dee3746b72537d8c83de0c3181127f02a2c228/continuity-evidence/README.md)

Each full video shows original SOMA on the left and the robot on the right,
107 frames at 30 fps (3.57 seconds). The GIF compares baseline and continuity
robot renders side by side. The numerical audit uses the 120 Hz output.

**Motion Data by [Bones Studio](https://bones.studio/).** The underlying data
is subject to the [BONES-SEED license](https://bones.studio/info/seed-license).
Raw BVH, motion CSV and human/robot asset files are not redistributed here.
SOMA/renderer: NVIDIA soma-retargeter; robot: Agibot A3.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests exercise a floating-root model (different coordinate/DoF indices),
expanded candidate batches, finite-difference Jacobian agreement, reference
snapshot isolation, warmup/frame lifecycle, invalid joint selections, and an
actual Newton IK solve with a known tracking/continuity optimum. The five tests
pass on CPU; the 424-frame bundled-A3 comparison also completes
on CUDA through the existing graph-captured base pass.
