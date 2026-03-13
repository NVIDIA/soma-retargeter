# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class BaseRenderer:
    """Base class for renderers that tracks registered viewer objects."""

    def __init__(self):
        self.registered_ids = set()

    def _register_unique_id(self, id):
        """Track an object id for later cleanup."""
        self.registered_ids.add(id)

    def _clear(self, renderer_dict):
        """Destroy and remove all registered objects from the viewer dict."""
        for id in self.registered_ids:
            destroy_func = getattr(renderer_dict[id], "destroy", None)
            if callable(destroy_func):
                destroy_func()

            del renderer_dict[id]

        self.registered_ids = set()
