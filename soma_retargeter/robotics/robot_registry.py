# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry for discovering and resolving robot retargeting targets from manifest files."""
import os
from dataclasses import dataclass

import soma_retargeter.io.utils as utils

_BUILTIN_ROOT = utils.get_assets_dir() / 'robotics'

@dataclass
class RobotTarget:
    name: str
    manifest_dir: str
    desc: dict
    retarget_configs: dict


class RobotRegistry:
    """
    Discovers robot targets from ``manifest.json`` files so that new robots
    can be added without any code changes.

    Each robot directory contains a ``manifest.json`` with at minimum::

        {
            "name": "my_robot",
            "desc": { "urdf_path": "desc/robot.urdf" },
            "retarget_configs": {
                "soma": "configs/soma_to_robot_retargeter_config.json"
            }
        }

    A manifest may also declare multiple targets via a ``"targets"`` array.

    The registry scans a built-in directory on first access. Additional
    directories can be registered at runtime via :meth:`add_search_path`;
    targets found there **override** built-in targets that share the same
    name, so users can iterate on a config without modifying the repository.
    """

    def __init__(self, builtin_root: str):
        """
        Args:
            builtin_root: Absolute path to the directory tree containing bundled robot manifests.
        """
        self._builtin_root = builtin_root
        self._targets: dict[str, RobotTarget] = {}
        self._initialized = False
        self._extra_paths: list[str] = []

    def add_search_path(self, path: str):
        """
        Register an additional directory tree to scan for manifests.

        Robots discovered here override built-in robots that share the same
        name, allowing users to iterate on configs without modifying the
        repository.
        """
        abs_path = os.path.abspath(path)
        self._extra_paths.append(abs_path)
        if self._initialized:
            self._scan_directory(abs_path)

    def get(self, name: str) -> RobotTarget:
        """Look up a robot target by name. Raises ``ValueError`` if unknown."""
        self._ensure_loaded()
        target = self._targets.get(name)
        if target is None:
            available = ", ".join(sorted(self._targets.keys()))
            raise ValueError(f"Unknown target type: [{name}]. Available: {available}")
        return target

    def list_names(self) -> list[str]:
        """Return a sorted list of all registered target names."""
        self._ensure_loaded()
        return sorted(self._targets.keys())

    def reload(self):
        """Clear all cached targets and re-scan the built-in root plus any extra paths."""
        self._targets = {}
        self._initialized = False
        self._ensure_loaded()

    def _ensure_loaded(self):
        """Scan the built-in root and any extra paths on first access."""
        if not self._initialized:
            self._scan_directory(self._builtin_root)
            self._initialized = True
            for path in self._extra_paths:
                self._scan_directory(path)

    def _scan_directory(self, root: str):
        """Recursively walk ``root`` and load every ``manifest.json`` found.

        Args:
            root: Directory to scan. Silently skips if the path does not exist.
        """
        if not os.path.isdir(root):
            return
        for dirpath, _, filenames in os.walk(root):
            if "manifest.json" in filenames:
                self._load_manifest(os.path.join(dirpath, "manifest.json"))

    def _load_manifest(self, path: str):
        """Parse a ``manifest.json`` file and register all valid robot targets it declares.

        Args:
            path: Absolute path to the manifest file.
        """
        data = utils.load_json(path)
        manifest_dir = os.path.dirname(os.path.abspath(path))
        entries = data.get("targets", [data])
        for entry in entries:
            if "name" not in entry:
                print(f"[WARNING] Malformed manifest {path}: entry is missing 'name', skipping")
                continue
            name = entry["name"]
            if "desc" not in entry:
                print(f"[WARNING] Robot '{name}' in {path} has no 'desc', skipping")
                continue
            self._targets[name] = RobotTarget(
                name=name,
                manifest_dir=manifest_dir,
                desc=entry["desc"],
                retarget_configs=entry.get("retarget_configs", {}),
            )

registry = RobotRegistry(_BUILTIN_ROOT)


def register_robots_path(path: str):
    """Register an additional directory tree to scan for robot manifest.json files."""
    registry.add_search_path(path)


def list_available_targets() -> list[str]:
    """Return a sorted list of all registered robot target names."""
    return registry.list_names()


def list_available_soma_targets() -> list[str]:
    """Return a sorted list of robots that have a soma retargeter config configured.

    Use this instead of ``list_available_targets()`` anywhere a soma retargeting
    pipeline is required (e.g. the BVH converter, IK optimizer). Robots registered
    via the Robot Configurator but not yet fully configured are excluded.
    """
    return [name for name in registry.list_names()
            if registry.get(name).retarget_configs.get("soma")]


def get_robot_asset_root(target: str) -> str:
    """Return the manifest directory for a robot target."""
    return registry.get(target).manifest_dir
