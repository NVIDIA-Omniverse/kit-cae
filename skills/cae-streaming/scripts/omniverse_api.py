#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Tiny request/signal framework over the WebRTC data channel.
#
# A small request/response + signal framework over the WebRTC data channel,
# generalised for any cae-streaming app. The two public decorators are:
#
#   @api.request   — bidirectional RPC: client sends `<name>_request`, kit replies `<name>_response`.
#                   The decorated function (sync or async) returns the response payload.
#   @api.signal    — kit-initiated push: dispatch arbitrary events as `<name>_signal`.
#
# Wire shape (full spec in `references/streaming-protocol.md`):
#   browser → kit:  { event_type: "<name>_request",  payload: { ...args, id } }
#   kit → browser:  { event_type: "<name>_response", payload: { response, id } }
#   kit → browser:  { event_type: "<name>_signal",   payload: { ... } }      (push only)
#
# Why a custom wrapper instead of `LivestreamMessaging.observe[_and_dispatch]`?
#   • observe() does NOT unpack payload as kwargs (kwargs come back empty).
#   • observe_and_dispatch() does NOT await coroutines (returns a coroutine
#     object that the dispatcher fails to JSON-serialize).
# Going through carb.events + omni.kit.async_engine.run_coroutine fixes both.

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable

import carb
import carb.events
import omni.kit.app
import omni.kit.livestream.messaging as messaging
from omni.kit.async_engine import run_coroutine

LOG_INFO = carb.log_info
LOG_ERROR = carb.log_error


def exclusive(func: Callable) -> Callable:
    """Decorator for `@api.request` async handlers: only one execution runs at
    a time; concurrent calls are dropped (return `None`). Useful for handlers
    that mutate the stage and can't safely overlap.
    """
    lock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if lock.locked():
            return None
        async with lock:
            return await func(*args, **kwargs)

    return wrapper


class OmniverseAPI:
    """Register request/response and signal handlers for the WebRTC data channel.

    Example:

        api = OmniverseAPI()

        @api.request
        async def load_scene(scene_id: str = "", **_):
            await import_to_stage(...)
            return {"ok": True, "scene_id": scene_id}

        @api.signal
        def notify(level: str = "info", message: str = ""):
            return {"level": level, "message": message}
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, Any] = {}
        self._signal_subscriptions: dict[str, Any] = {}

    def cleanup(self) -> None:
        for sub in self._subscriptions.values():
            sub.unsubscribe()
        self._subscriptions.clear()
        for sub in self._signal_subscriptions.values():
            sub.unsubscribe()
        self._signal_subscriptions.clear()

    # ------------------------------------------------------------------ request
    def request(self, func: Callable | None = None, *, name: str | None = None):
        """Register `<name>_request` → `<name>_response` round-trip.

        Use as bare decorator:  @api.request
        Or with a custom name:  @api.request(name="my_op")
        """
        if func is None:  # @api.request(name=...)
            return lambda f: self.request(f, name=name)
        return self._register_request(func, name)

    def _register_request(self, func: Callable, name: str | None) -> Callable:
        op_name = name or func.__name__
        request_name = f"{op_name}_request"
        response_name = f"{op_name}_response"
        request_evt = carb.events.type_from_string(request_name)

        # Subscribe via the legacy message bus stream — that's what the
        # WebRTC plugin still uses to deliver inbound messages (we see
        # "Processing custom kit message" in the kit log on every request).
        stream = omni.kit.app.get_app().get_message_bus_event_stream()

        def on_event(e: carb.events.IEvent) -> None:
            try:
                args = dict(e.payload.get_dict())
            except Exception as err:
                LOG_ERROR(f"[ovapi] failed to extract payload for {request_name}: {err}")
                return
            request_id = args.pop("id", -1)
            args.pop("sender_id", None)
            LOG_INFO(f"[ovapi] {request_name} id={request_id} args={args}")

            async def run() -> None:
                try:
                    if inspect.iscoroutinefunction(func):
                        response = await func(**args)
                    else:
                        response = func(**args)
                except Exception as ex:
                    import traceback
                    LOG_ERROR(f"[ovapi] {op_name} failed: {traceback.format_exc()}")
                    response = {"ok": False, "error": str(ex)}

                if request_id == -1:
                    return
                # Dispatch the response. omni.kit.livestream.messaging-1.3.0
                # subscribes via carb.eventdispatcher (Events 2.0). Use the
                # NEW queue_event API so the subscriber fires reliably; this
                # is also what the deprecation warning told us to do.
                LOG_INFO(f"[ovapi] dispatching {response_name} id={request_id} ok={response.get('ok') if isinstance(response, dict) else '?'}")
                omni.kit.app.queue_event(response_name, {"response": response, "id": request_id})

            run_coroutine(run())

        sub = stream.create_subscription_to_pop_by_type(request_evt, on_event, name=request_name)
        self._subscriptions[request_name] = sub
        messaging.register_event_type_to_send(response_name)
        LOG_INFO(f"[ovapi] registered request handler: {request_name}")
        return func

    # ------------------------------------------------------------------- signal
    def signal(self, func: Callable | None = None, *, name: str | None = None):
        """Register `<name>_signal` as a push-only event the handler can emit
        by being called normally; or call `api.dispatch_signal(name, payload)`
        for an explicit push. Less common than `@api.request`."""
        if func is None:
            return lambda f: self.signal(f, name=name)
        op_name = name or func.__name__
        signal_name = f"{op_name}_signal"
        messaging.register_event_type_to_send(signal_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            payload = func(*args, **kwargs) or {}
            self.dispatch_signal(op_name, payload)
            return payload

        return wrapper

    def dispatch_signal(self, op_name: str, payload: dict) -> None:
        signal_name = f"{op_name}_signal"
        # Same Events-2.0 path as `_register_request`'s response dispatch.
        omni.kit.app.queue_event(signal_name, payload)
