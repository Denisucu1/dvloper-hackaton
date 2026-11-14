"""Hard deadline guard: the bot always answers before the arena's clock.

`guarded()` runs the real work in a worker thread and returns a fallback if it overruns. Only the main
thread ever writes to stdout, so a slow worker can never corrupt the protocol or answer twice.
"""
from __future__ import annotations

import threading
from typing import Callable, TypeVar, cast

from .log import log, log_exc

T = TypeVar("T")


def guarded(fn: Callable[[], T], seconds: float, on_timeout: Callable[[], T], label: str = "") -> tuple[T, bool]:
    """Run fn with a hard budget. Returns (value, timed_out). Never raises."""
    box: list[T] = []

    def target() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # noqa: BLE001 - a handler bug must not kill the container
            log_exc(f"handler {label} failed:", exc)

    worker = threading.Thread(target=target, name=f"work-{label}", daemon=True)
    worker.start()
    worker.join(max(0.2, float(seconds)))
    if box:
        return box[0], False
    log(f"DEADLINE {label}: no answer in {seconds:.1f}s, sending fallback")
    try:
        return on_timeout(), True
    except BaseException as exc:  # noqa: BLE001 - the fallback itself must not kill us
        log_exc(f"fallback {label} failed:", exc)
        return cast(T, None), True
