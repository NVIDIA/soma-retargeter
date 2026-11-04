# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from pathlib import Path
from typing import Union, Dict

# Assumes this file is in soma_retargeter/utils
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def get_package_root() -> Path:
    """Return the filesystem path to the package root."""
    return _PACKAGE_ROOT


def get_configs_dir() -> Path:
    """Return the configs directory path."""
    return get_package_root() / 'configs'


def get_config_file(*relative_parts: str) -> Path:
    """Return a path to a specific file under configs/."""
    return get_configs_dir().joinpath(*relative_parts)


def load_json(path: Union[str, Path]) -> Dict:
    """
    Load a JSON file from the specified path.
    Args:
        path (Union[str, Path]): The file path to the JSON file. Can be a string or Path object.
    Returns:
        Dict: The parsed JSON content as a dictionary.
    Raises:
        FileNotFoundError: If the JSON file does not exist at the specified path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR]: JSON file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
