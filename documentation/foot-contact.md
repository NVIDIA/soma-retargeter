# Contact detection & foot-plant correction configuration

All configs under `soma_retargeter/assets/`.

There are two sides to the config:
 - Human (source) side : configures the contact detection and foot-plant acceptance gates
 - Robot side : configures the foot-plant correction


## Human side

`assets/<source>/contact_processing/*_contact_config.json` 

For SOMA : `soma_retargeter/assets/soma/contact_processing/soma_contact_config.json` 

One file per source, used by all robots. Resolved by source type, so a change here affects every robot.

**Contact detection** — finds coarse contact segments from foot speed and jerk.

| Param | Default | |
|---|---|---|
| `joint_names` | `LeftFoot, LeftToeBase, RightFoot, RightToeBase` | contact joints |
| `foot_channel_map` | `{0:0, 2:1}` | index into `joint_names` → foot index (LeftFoot=0→left, RightFoot=2→right) |
| `velocity_contact` | 0.1 m/s | speed below this → contact starts |
| `jerk_contact` | -0.05 m/s³ | jerk below this opens the contact window |
| `velocity_uncontact` | 0.2 m/s | speed above this for 2 frames → contact ends |
| `velocity_probably_lift_off` | 0.05 m/s | speed above this marks the candidate lift-off frame contact is rolled back to |
| `post_contact_jerk_window` | 5 frames | how long a jerk trigger stays open |
| `transition_frames` | 4 frames | ramp added on each side of a contact region |
| `edge_propagation_frames` | 8 frames | max extension of a contact at clip start/end |

**Plant acceptance gates** — `plant_subsegment.*`, promotes a contact segment to a
correctable flat plant. All are code defaults; only present keys override.

| Param | Default | |
|---|---|---|
| `max_flatness_deg` | 25° | max sole-normal tilt from world up |
| `plant_speed_threshold` | 0.15 m/s | max foot speed; also normalizes the speed score |
| `max_height_spread` | 0.05 m | max heel-to-toe height difference |
| `max_position_delta` | 0.01 m | max per-frame foot-centre movement |
| `min_plant_frames` | 3 frames | shorter runs are dropped |
| `max_gap_frames` | 2 frames | bridges noise gaps inside a plant run |
| `min_acceptance_confidence` | 0.5 | reject the run below this confidence |
| `ground_height_tolerance` | 0.03 m | foot centre within this of z=0 counts as on the ground |
| `ground_confidence_boost` | 1.15 | confidence multiplier when on the ground |
| `elevated_confidence_scale` | 1.0 | confidence multiplier when off the ground |

Confidence is the mean per-frame `flatness_score × speed_score` over the run, then scaled by
whichever of the two multipliers applies, capped at 1.0.

**Landmarks**

| Param | Default | |
|---|---|---|
| `foot_landmarks` | — | per-side foot geometry in the **source skeleton's** joint frames, see *Foot landmarks* below |


## Robot side

`assets/robotics/<vendor>/<robot>/configs/`

**1. Enable — `*_retargeter_config.json`**
- `enable_post_processing`
- `enable_contact_processing`
- `post_processing.robot_config` → path to the post-processing config below

**2. Tune — `*_post_processing_config.json`, key `contact_correction`**

| Param | Default | |
|---|---|---|
| `transition_frames` | 4 frames | blend length |
| `propagation_ratio` | 0.4 | fraction (0–1) of adjacent swing taking position correction |
| `rotation_propagation_ratio` | 0.15 | fraction (0–1) of adjacent swing taking rotation correction |
| `enable_flatten_foot_plant` | true | **not an on/off switch** — `true` flattens the sole to horizontal and may shift heading (yaw); `false` still flattens, but preserves the original heading |
| `sole_normal_local` | *(unset)* | sole-up direction in the **robot foot effector link's** frame; defaults to `+Z` |

Flattening only fires when plant tilt exceeds 5° and subsegment confidence is >= 0.5.


## Configuring foot plant correction for a new robot

Contact detection runs entirely on the human skeleton, so the robot supplies no foot
geometry — no heel, no toe, no tip.

1. Set the two enable flags and `post_processing.robot_config` in `*_retargeter_config.json`.
2. Tweak the `contact_correction` parameters in `*_post_processing_config.json` as needed.
3. Set `sole_normal_local` only if the foot effector link's local `+Z` is not sole-up at the
   zero pose.


## Source foot landmarks

The `foot_landmarks` block of `*_contact_config.json`, under `rigs.<source>.left` / `.right`.
Four joint names plus three vectors per side:

```jsonc
{
  // Names in the source skeleton, define the frame for the vectors below
  "foot_joint": "LeftFoot",         // ankle joint. Frame for heel_local_in_foot
  "toe_joint": "LeftToeBase",       // toe base joint. Frame for toe_pivot_local_in_toe
  "toe_end_joint": "LeftToeEnd",    // optional, may be null. Only used by the heuristic
                                    // fallback to find the foot's forward axis; ignored
                                    // when the vectors below are authored
  "sole_normal_space": "foot",      // "foot" or "toe" — local frame for the sole_normal_local
                                    // Default "foot"

  // Relative to the joint named in the key, in metres
  "heel_local_in_foot":     [x, y, z],  // heel contact point, offset from foot_joint's origin
  "toe_pivot_local_in_toe": [x, y, z],  // toe/ball contact point, offset from toe_joint's origin
  "sole_normal_local":      [x, y, z]   // unit DIRECTION, points UP out of the sole
}
```

However, registering a **new source type** is a code change, not a config change: `SourceType` is an
enum, and the contact-config path, zero-pose asset and model mesh are each resolved by a
switch on it. The skeleton key must also be added to the heuristic-config table.
