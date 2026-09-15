# Installation

This page prepares a clean SOMA Retargeter checkout. The rest of the documentation uses
`uv`, but a conda-based installation is also presented as an option for installation. 
## Before you begin

You need:

- **Windows 10/11 or Linux** on x86-64;
- an **NVIDIA GPU** with Maxwell architecture or newer;
- an **NVIDIA driver version 545 or newer**, suitable for CUDA 12;
- **Git** and **Git LFS**;
- internet access while cloning, downloading large assets, and installing dependencies;
- enough free disk space for the repository, robot meshes, and Python environment.

Python 3.12 is required. You do not need to install it separately when `uv` is allowed to download Python; conda users create a Python 3.12 environment explicitly. A local CUDA Toolkit is not required.

## 1. Check the GPU driver

Open PowerShell on Windows or a terminal on Linux:

```bash
nvidia-smi
```

Expected result: a table that identifies your NVIDIA GPU and reports a driver version of at least 545.

If the command is missing or the driver is older, install a current NVIDIA driver, restart the computer, and run `nvidia-smi` again. Do not continue until the GPU appears correctly; Python installation cannot repair a missing GPU driver.

## 2. Install Git and Git LFS

Install Git from the official Git website or your operating system's package manager. Then install Git LFS and initialize it for your user account:

```bash
git lfs install
git lfs version
```

Expected result: the second command prints a Git LFS version.

On Ubuntu or Debian, the package installation is typically:

```bash
sudo apt-get update
sudo apt-get install git git-lfs
git lfs install
```

On Windows, Git LFS is normally included with Git for Windows. If `git lfs` is not recognized, rerun the Git for Windows installer and enable Git LFS.

## 3. Choose an installation method

Use one of these methods for the rest of the page.
### Option 1: uv *(used in this documentation)*

`uv` creates and manages the project environment without requiring you to activate it manually. Follow the official [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then open a new terminal and check:

```bash
uv --version
```

Expected result: a line beginning with `uv`.

If `uv` is not recognized immediately after installation, close and reopen the terminal so it reloads your `PATH`. If that does not help, use the installer's printed instructions to add the uv executable directory to `PATH`.

### Option 2: conda + pip

Use conda if your environment policy requires it or if you already manage Python toolchains through conda. Install Miniconda, Anaconda, or another conda distribution, then check:

```bash
conda --version
```

Expected result: a line beginning with `conda`.

Conda users will activate an environment before running project commands. Later pages use `uv run python`; with conda, use `python` in an activated `soma-retargeter` environment instead.

## 4. Get the repository

In a directory where you keep projects, run:

```bash
git clone https://github.com/NVIDIA/soma-retargeter.git
cd soma-retargeter
```

If you already have a checkout, open a terminal in its root instead.

## 5. Download the large assets

From the repository root:

```bash
git lfs pull
```

This replaces small Git LFS pointer files with the actual robot models, meshes, sample motions, and other large files.

Expected result: the command finishes without an error. It may print little or nothing after the download is complete.

To detect an incomplete asset download:

```bash
git lfs status
```

If `git lfs pull` fails, check the network connection and available disk space, then run it again. If authentication errors repeat, re-clone the repository from an HTTPS URL that your account can access, then run `git lfs pull` again.

## 6. Install the project

Use the method you chose in step 3.

### Option 1: uv

Run:

```bash
uv sync
```

`uv` reads the project lock file, obtains Python 3.12 when needed, creates a `.venv` directory inside the checkout, and installs the exact project dependencies.

Expected result: the command completes successfully and reports that packages were resolved and installed. The first run can take several minutes.

You do not have to activate `.venv`. Commands beginning with `uv run` use it automatically.

If the download times out, rerun `uv sync`; completed downloads are normally reused. If installation reports damaged or inconsistent cached packages, retry with:

```bash
uv cache clean
uv sync
```

### Option 2: conda + pip

Create and activate a Python 3.12 conda environment:

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
```

Install the project from the checkout:

```bash
python -m pip install -e .
```

The root `README.md` shows `pip install .`; this guide uses the editable form so runtime robot and SOMA assets are read directly from the checkout during documentation workflows.

Expected result: pip installs SOMA Retargeter and its dependencies into the active `soma-retargeter` conda environment. In each new terminal, run `conda activate soma-retargeter` before using `python`.

### Windows prerequisite for the viewer

If installation or viewer startup reports a missing Visual C++ runtime or an `imgui-bundle` DLL error, install the current [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), restart the terminal, and rerun the install command for your method: `uv sync` or `python -m pip install -e .` inside the activated conda environment.

### Linux prerequisite for file dialogs

The interactive viewer uses Tk file dialogs. On Ubuntu or Debian, install the Python 3.12 Tk package:

```bash
sudo apt-get install python3.12-tk
```

If your distribution uses a different package name, install its Tkinter package for Python 3.12. A missing `_tkinter` module affects the viewer's **Load** and **Save** dialogs; headless batch conversion does not need a display or Tk file dialog.

## 7. Verify the installation

First confirm the Python version:

With `uv`:

```bash
uv run python --version
```

With conda:

```bash
python --version
```

Expected result: `Python 3.12.x`.

Then confirm that the package imports and discovers all bundled robots:

With `uv`:

```bash
uv run python -c "from soma_retargeter.robotics.robot_registry import list_available_targets; print(', '.join(list_available_targets()))"
```

With conda:

```bash
python -c "from soma_retargeter.robotics.robot_registry import list_available_targets; print(', '.join(list_available_targets()))"
```

Expected result:

```text
agibot-a3t3, agibot_x2ultra, booster_t1, unitree_g1, unitree_h2
```

The order is alphabetical. If import fails, make sure you are still in the repository root, then rerun `uv sync` for uv or `python -m pip install -e .` in the activated conda environment. If one or more robot names are missing, rerun `git lfs pull`, then retry the verification.

## Optional: verify that a window can open

Start the converter:

With `uv`:

```bash
uv run python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl
```

With conda:

```bash
python app/bvh_to_csv_converter.py --config assets/default_bvh_to_csv_converter_config.json --viewer gl
```

Expected result: after initial GPU setup, a window titled **BVH to CSV Converter** opens with a robot and control panels. The first launch can pause while GPU kernels initialize.

Close the window normally after the check. If it is blank or immediately exits:

1. read the terminal for the first error;
2. recheck `nvidia-smi`;
3. apply the Windows runtime or Linux Tk note above when relevant; and
4. see [Troubleshooting](troubleshooting.md) if the error remains.

## Where to go next

Continue to **[First retarget](first-retarget.md)**. You will load a bundled walking motion, retarget it to Unitree G1, inspect the result, and save a new CSV. Keep this installation page available in case the first viewer launch exposes a driver, runtime, or Tk issue.

---

[Documentation home](../README.md) · [Next: First retarget →](first-retarget.md)
