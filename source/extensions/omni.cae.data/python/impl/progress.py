# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = [
    "ProgressContext",
    "progress_context",
    "interrupt",
]

import functools
from logging import getLogger

import carb.events
import omni.kit.app

logger = getLogger(__name__)


STATUS_BAR_PROGRESS_EVENT = "omni.kit.window.status_bar@progress"
STATUS_BAR_ACTIVITY_EVENT = "omni.kit.window.status_bar@activity"


class StackItem:
    def __init__(self, name: str, shift: float, scale: float):
        self.name = name
        self.shift = shift
        self.scale = scale


class Stack:
    def __init__(self):
        self.stack = []
        self._interrupt_flag = False
        self._last_progress_value = 0

    def interrupt(self):
        self._interrupt_flag = True

    def test_interrupt(self):
        if self._interrupt_flag:
            self.clear_interrupt()
            raise InterruptedError("User interrupted")

    def clear_interrupt(self):
        self._interrupt_flag = False

    def push(self, item: StackItem):
        self.stack.append(item)

    def pop(self):
        self.stack.pop()
        if not self.stack:
            # clear progress
            omni.kit.app.queue_event(STATUS_BAR_ACTIVITY_EVENT, {"text": ""})
            omni.kit.app.queue_event(STATUS_BAR_PROGRESS_EVENT, {"progress": -1.0})
            self._last_progress_value = 0
            self.clear_interrupt()

    def notify(self, progress: float):
        progress_value = self._get_progress(progress)

        # # uncomment this for debugging
        # if progress_value != 0.0 and progress_value < self._last_progress_value:
        #     logger.error("Progress went backwards!")
        #     raise RuntimeError("Progress went backwards!")

        self._last_progress_value = progress_value
        omni.kit.app.queue_event(STATUS_BAR_PROGRESS_EVENT, {"progress": progress_value})
        omni.kit.app.queue_event(STATUS_BAR_ACTIVITY_EVENT, {"text": self._get_activity()})

    def _get_activity(self) -> str:
        # iterate over the stack in reverse order and find the
        # last non-empty name
        for item in reversed(self.stack):
            if item.name:
                return item.name
        return ""

    def _get_progress(self, progress: float) -> float:
        # compute the progress based on the stack
        for item in reversed(self.stack):
            progress = progress * item.scale + item.shift
            assert progress >= 0.0 and progress <= 1.0
        return progress


stack = Stack()


class ProgressContext:
    def __init__(self, name: str = "", shift: float = 0.0, scale: float = 1.0):
        assert shift >= 0.0 and shift <= 1.0, "shift must be between 0.0 and 1.0"
        assert scale > 0.0 and scale <= 1.0, "scale must be between 0.0 and 1.0"
        self._name = name
        self._shift = shift
        self._scale = scale

    def __enter__(self):
        global stack
        stack.test_interrupt()
        self._indent = len(stack.stack)
        stack.push(StackItem(self._name, self._shift, self._scale))
        stack.notify(0.0)
        # logger.warning(f" -- {stack._get_progress(0):0.2f} {' ' * self._indent} start: {self._name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # logger.warning(f" -- {stack._get_progress(1.0):0.2f} {' ' * self._indent} end: {self._name}")
        stack.notify(1.0)
        stack.pop()
        return False  # propagate exceptions

    def notify(self, progress: float):
        global stack
        assert progress >= 0.0 and progress <= 1.0, "progress must be between 0.0 and 1.0"

        stack.test_interrupt()
        stack.notify(progress)


def progress_context(name: str):
    """
    Decorator for progress_context.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with ProgressContext(name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def interrupt():
    global stack
    stack.interrupt()
