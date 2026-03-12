# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Hooks for Warp kernel compilation events.

Provides a registry of callbacks that are invoked just before a Warp module is
compiled. This fires only when the binary is absent (i.e. first launch, cache
miss, or cache disabled) — never on cache hits.

Usage::

    import dav

    def on_compile(module_name, kernel_names, device):
        print(f"Compiling {len(kernel_names)} kernel(s) in '{module_name}' for {device}")

    dav.register_pre_compile_hook(on_compile)
"""

__all__ = ["register_pre_compile_hook"]

from collections.abc import Callable
from logging import getLogger

import warp as wp

logger = getLogger(__name__)

# Callbacks: (module_name: str, kernel_names: list[str], device) -> None
_pre_compile_callbacks: list[Callable] = []


def register_pre_compile_hook(callback: Callable) -> None:
    """Register a callback invoked just before a Warp module is compiled.

    The callback is called once per ``(module, device)`` pair when the compiled
    binary does not yet exist in the cache.  It is *not* called on cache hits.

    Args:
        callback: Callable with signature
            ``(module_name: str, kernel_names: list[str], device) -> None``.
            ``module_name`` is the Warp module name (e.g. ``"dav.operators.probe"``).
            ``kernel_names`` is the list of kernel keys in that module.
            ``device`` is the Warp device the module is being compiled for.
    """
    _pre_compile_callbacks.append(callback)


def _install() -> None:
    """Monkey-patch ``Module.compile`` to inject pre-compile notifications.

    Called once at ``dav.core`` import time.  Subsequent calls are no-ops.
    """
    if hasattr(wp, "Module"):
        w_module = wp.Module
    elif hasattr(wp, "context") and hasattr(wp.context, "Module"):
        w_module = wp.context.Module
    else:
        # Unexpected Warp version; fail gracefully without hooks.
        logger.warning("Unable to find Warp Module class; pre-compile hooks will be disabled.")
        return

    if hasattr(w_module, "_compile"):
        w_module_compile = w_module._compile
        compile_attr = "_compile"
    elif hasattr(w_module, "compile"):
        w_module_compile = w_module.compile
        compile_attr = "compile"
    else:
        # Unexpected Warp version; fail gracefully without hooks.
        logger.warning("Unable to find Warp compile method; pre-compile hooks will be disabled.")
        return

    if getattr(w_module_compile, "_dav_hooked", False):
        return

    _orig_compile = w_module_compile

    def _patched_compile(self, device=None, output_dir=None, output_name=None, output_arch=None, use_ptx=None):
        if _pre_compile_callbacks:
            kernel_names = list(self.kernels.keys())
            for cb in _pre_compile_callbacks:
                cb(self.name, kernel_names, device)
        return _orig_compile(self, device, output_dir, output_name, output_arch, use_ptx)

    _patched_compile._dav_hooked = True
    setattr(w_module, compile_attr, _patched_compile)
