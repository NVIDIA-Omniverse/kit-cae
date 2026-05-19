# Kit-Side Messaging API

Reference for `omni.kit.livestream.messaging` — the extension that bridges the
WebRTC data channel and Kit's `carb.eventdispatcher`. The bundled
`scripts/omniverse_api.py` wraps it into a `@api.request` / `@api.signal`
framework that's easier to use day-to-day; this file documents the
underlying primitives in case you want to write to them directly or build
your own framework on top.

## Class surface

`omni.kit.livestream.messaging.LivestreamMessaging` is the singleton. The
public methods you care about:

| Method | Purpose |
|---|---|
| `observe(event_name)` | Decorator that turns a function into a one-way receiver for `event_name`. |
| `observe_and_dispatch(event_name, response_name="")` | Decorator that turns a function into a request handler. Return value is auto-dispatched as `response_name`. |
| `register_event_type_to_send(event_name)` | Manually register an outgoing event type — only needed if you push without using the decorator. |
| `unregister_observe_type(event_name)` | Stop listening; useful if `serve.py` manages multiple handlers dynamically. |

The module also exposes function-level shims: `observe`, `observe_and_dispatch`,
`register_event_type_to_send`. They forward to the singleton.

## `observe_and_dispatch` semantics — SYNC HANDLERS ONLY

```python
from omni.kit.livestream.messaging import LivestreamMessaging

messaging = LivestreamMessaging()

@messaging.observe_and_dispatch("ping_request", response_name="ping_response")
def ping(event, text: str = "", **_):  # def, NOT async def
    return {"ok": True, "echo": text}
```

Key points:

- **Always pass `response_name=`.** The default is `f"{event_name}:response"`
  (with a colon). Our wire convention uses `_response` (underscore), so an
  explicit value is mandatory.
- **DO NOT use this decorator for `async def` handlers.** Verified live: the
  decorator captures the function's immediate return value. For an async
  function that's a coroutine, which the dispatcher then tries to
  JSON-serialize → `TypeError: Object of type coroutine is not JSON
  serializable` plus `RuntimeWarning: coroutine '<name>' was never awaited`.
  Use the async pattern below instead.
- **Return a dict.** Kit serializes the dict and includes the original `id`.
- **Errors stay inside the dict.** Don't raise — return `{"ok": False, "error": "..."}`.
  Raising surfaces a Python traceback and never reaches the browser.

## Async handler pattern (for any handler that `await`s)

Most CAE handlers need to await `import_to_stage`, `execute_command`, or
`wait_for_update`. Use sync `observe` (receive) + `asyncio.ensure_future`
(schedule) + manual `dispatch_event` (response):

```python
import asyncio
import carb.eventdispatcher
from omni.kit.livestream.messaging import LivestreamMessaging

messaging = LivestreamMessaging()
event_dispatcher = carb.eventdispatcher.get_eventdispatcher()

# Required: register the response event so the messaging extension
# forwards dispatched events of that name out the data channel.
messaging.register_event_type_to_send("load_scene_response")

@messaging.observe("load_scene_request")
def _receive(event, **payload):
    request_id = payload.get("id", -1)
    asyncio.ensure_future(_async_handler(request_id, payload))

async def _async_handler(request_id, payload):
    try:
        # await the async work — import_to_stage, execute_command, …
        result = {"ok": True, "...": "..."}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    event_dispatcher.dispatch_event(
        "load_scene_response",
        payload={"id": request_id, "response": result},
    )
```

The browser library matches the response to the request by the `id` field,
so always echo the original `id`.

## `observe` semantics (one-way)

For one-way receivers (no reply expected at all):

```python
@messaging.observe("client_focus_signal")
def on_client_focus(event, focused: bool = False, **_):
    print(f"Client focus changed: {focused}")
```

No response is dispatched. Use this for events the browser broadcasts but
doesn't await.

## Pushing signals from Kit

For Kit-initiated push events, use `carb.eventdispatcher` directly after
calling `register_event_type_to_send` once at startup:

```python
import carb.eventdispatcher

messaging.register_event_type_to_send("notification_signal")

ed = carb.eventdispatcher.get_eventdispatcher()
ed.dispatch_event("notification_signal", payload={
    "level": "info",
    "message": "Scene loaded",
})
```

The library forwards any dispatched event whose name was registered for
sending.

## Lifecycle

The extension's own `on_startup` acquires a sender id and subscribes to the
receive bus when the `.kit` lists it under `[dependencies]`. No explicit
init call is needed from your script. `on_shutdown` cleans up at teardown.

## Verifying load

If you suspect the extension didn't load, grep the build cache:

```bash
ls _build/linux-x86_64/release/extscache/ | grep livestream.messaging
```

…and add an explicit dependency line if missing:

```toml
[dependencies]
"omni.kit.livestream.messaging" = {}
```

## See also

- `scripts/omniverse_api.py` — the bundled `@api.request` / `@api.signal`
  framework (wraps everything above into a one-decorator API). Read this if
  you want to see how the patterns in this doc compose, or if you want to
  fork it for your own framework.
- `scripts/serve.py` — the runnable example handler set built on
  `omniverse_api.py`.
- `references/streaming-protocol.md` — wire format the messages must follow.
- API docs JSON (vendored): `experimental/kit-usd-agents/.../api_docs/omni.kit.livestream.messaging-1.2.1.api_docs.json`.
