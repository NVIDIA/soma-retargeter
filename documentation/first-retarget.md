# Your first retarget

In this tutorial you will convert a bundled SOMA walking animation to Unitree G1 joint motion. You will use the interactive viewer to confirm the motion before saving a CSV.

## What you will produce

Input:

```text
assets/motions/bvh/Neutral_walk_forward_002__A057.bvh
```

Target robot:

```text
unitree_g1
```

Suggested output:

```text
assets/motions/test-export/my_first_retarget.csv
```

The output directory does not have to exist in advance when you create it in the Save dialog. If you prefer not to write under `assets`, choose any writable folder and use the `.csv` extension.

![SOMA BVH is scaled, solved, post-processed, and exported as robot CSV](images/retargeting-pipeline.png)

## 1. Start the interactive viewer

Open PowerShell on Windows or a terminal on Linux in the repository root. Run:

```bash
uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl
```

Expected result: a window titled **BVH to CSV Converter** opens. The Scene Options panel appears on the right, playback controls appear along the bottom, and the selected robot is visible. The default configuration selects `unitree_g1`.

![Interactive Viewer](images/interactive-viewer.png)

If no window appears:

- Wait during the first launch and watch the terminal for progress or an error.
- If the terminal says the configuration file is missing, close it and rerun the command from the directory containing `pyproject.toml`.
- If `uv` or an import is missing, return to [Installation](installation.md) and rerun `uv sync`.
- On Linux, an error mentioning `_tkinter` means you need the Python 3.12 Tk package described on the installation page.
- For a graphics or CUDA error, confirm that `nvidia-smi` works and that the NVIDIA driver is version 545 or newer.

## 2. Confirm the robot target

In **Scene Options**, find the robot selector near the top. Select `unitree_g1` if it is not already selected.

The viewer also includes `unitree_h2` and `booster_t1`. Stay with Unitree G1 for this tutorial because the repository includes matching reference CSV files for the sample BVH motions.

If the list is empty or Unitree G1 is absent, close the viewer, run:

```bash
git lfs pull
uv run python -c "from soma_retargeter.robotics.robot_registry import list_available_targets; print(list_available_targets())"
```

The second command should include `unitree_g1`. Then reopen the viewer. The **Reload** button beside the selector rescans robot manifests, but it cannot restore assets that were never downloaded.

## 3. Load the sample BVH

In **Scene Options**:

![Load BVH in Scene Options](images/scene-options-loadBVH.png)

1. Expand **Motion** if it is collapsed.
2. On the **BVH Motion** row, select **Load**.
3. Browse to `assets/motions/bvh/Neutral_walk_forward_002__A057.bvh`.
4. Select **Open**.

Expected result: the human source motion appears in the viewport. The total duration at the bottom becomes greater than zero, and the source begins playing when playback is enabled.

![BVH loaded in viewer](images/viewer-BVH-loaded.png)

If the file browser opens somewhere unexpected, navigate to the checkout and then through `assets`, `motions`, and `bvh`. On Linux, hidden or sandboxed directories may not appear in a graphical file dialog; type or paste the full path into the dialog when available.

If loading fails with a skeleton or joint-count error, verify that you selected the exact bundled BVH above. The converter expects a SOMA-skeleton BVH; an arbitrary BVH is not a valid substitute for this installation test.

## 4. Retarget the motion

On the **BVH Motion** row, select **Retarget**.

![Retarget button in BVH Motion](images/viewer-retarget-button.png)

The retargeting process can take a few seconds to a few minutes, depending on the length of the animation and your hardware. Expect a delay before seeing the retargeted animation.

During the retargeting process, the pipeline scales source targets to Unitree G1, solves the configured inverse-kinematics objectives frame by frame, applies contact and joint-limit post-processing, and places the resulting robot animation in the viewer.

Expected result:

- the robot moves with the walking source;
- the terminal returns to normal informational output without a traceback; 
- the **Save** button on the **CSV Motion** row becomes available.

![Retargeted motion in viewer](images/viewer-retargeted.png)

Retargeting is computation-heavy. A long processing time is normal, especially on the first run. Do not click **Retarget** repeatedly while it is working.

If the robot remains still, make sure the time slider is moving and select **Play** if playback is paused. If **Save** remains disabled, the retarget did not produce an output; check the terminal for the first error rather than the last line of a long traceback.

## 5. Inspect the result

Before saving, use the bottom controls:

- **Play/Pause** starts or stops playback.
- **Time** scrubs to a specific moment.
- **Speed** slows, accelerates, or reverses playback.
- **Loop** repeats the animation.

You can use the transformations gizmos to move the robot away from the human.

The **Visibility** section can show the source mesh, source skeleton, joint axes, and positioning gizmos. 

![Visibility options in Viewer](images/viewer-visibility.png)

These controls change visualization only; they do not change the saved joint values. The **Reset** button restores visual offsets if a gizmo was moved accidentally.

For this first pass, look for a recognizable forward walk and continuous motion. Minor differences between human and robot poses are expected because their proportions, feet, and joint structures differ.

## 6. Save the CSV

On the **CSV Motion** row:

1. Select **Save**.
2. Browse to `assets/motions/test-export`.
3. Create the folder in the dialog if it does not exist.
4. Enter `my_first_retarget.csv`.
5. Confirm the save.

Expected result: the dialog closes and the file exists at the selected location. The `.csv` extension may be added automatically if you omitted it.

If saving fails, choose a directory where your user account has write permission, such as a folder under your home directory. Avoid overwriting the bundled reference CSV until you have finished comparing results.

## 7. Verify the saved file

Close the viewer normally. Then check the output.

Expected result: the file is not empty. The first line contains column names, and the next line contains numeric animation data.

The repository's reference result is:

```text
assets/motions/csv/Neutral_walk_forward_002__A057.csv
```

Both files should have a similar structure for the same Unitree G1 target. Do not treat a byte-for-byte difference as an automatic failure: output can vary with configuration and dependency changes. Visual continuity and valid columns are more useful first checks.

You can also close and re-launch the **BVH to CSV Converter** and use the CSV load button to load the .csv file directly onto the robot without retargeting from BVH animation. 

![Load CSV button in viewer](images/viewer-load-csv.png)

The robot will playback the animation from the .csv file.

![CSV animation loaded in viewer](images/viewer-csv-loaded.png)

## Optional: try another bundled robot

Once the G1 result works, reopen the viewer, select `unitree_h2` or `booster_t1`, load a different BVH animation and retarget it . Save each result under a distinct name.

Each target uses its own model, objectives, scaling, and post-processing configuration, so the joint columns and motion will differ. The bundled CSV reference folder is for Unitree G1; it is not a reference for H2 or T1.

## What you have confirmed

Completing this tutorial verifies that:

- Python dependencies and GPU execution work;
- large sample and robot assets are available;
- the viewer can load a SOMA BVH;
- the Unitree G1 pipeline can produce motion; 
- the result can be written as a robot-joint CSV.

That known-good baseline is important. When a later custom motion or robot fails, repeat this sample to distinguish an installation problem from a data or configuration problem.

## Where to go next

- To understand all viewer controls and use your own SOMA files, read **[Interactive retargeting](interactive-retargeting.md)**.
- To convert an entire directory, go to **[Batch retargeting](batch-retargeting.md)**.
- To add a robot, continue in the planned order with **[Robot configurator](robot-configurator.md)**. That tutorial uses Unitree H1 as a new-robot example; the bundled robots are Unitree G1, Unitree H2, and
Booster T1.
- If any expected result above was missing, use **[Troubleshooting](troubleshooting.md)** before tuning settings.

---

[← Previous: Installation](installation.md) · [Documentation home](../README.md) · [Next: Robot configurator →](robot-configurator.md)