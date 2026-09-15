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


def get_assets_dir() -> Path:
    """Return the assets directory path."""
    return get_package_root() / 'assets'


def get_asset_file(*relative_parts: str) -> Path:
    """Return a path to a specific file under assets/."""
    return get_assets_dir().joinpath(*relative_parts)


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


def save_json(path: Union[str, Path], data: Dict, indent: int = 4) -> None:
    """Write *data* to a JSON file at *path*, creating parent directories if needed.

    Args:
        path: Destination file path (str or Path).
        data: JSON-serialisable dict to write.
        indent: Number of spaces for indentation (default 4).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)
