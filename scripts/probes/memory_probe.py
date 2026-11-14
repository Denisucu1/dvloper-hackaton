"""Probe: atomic memory writes under concurrent writers.

Rule 5.14: every container of the team shares ``/memory`` and may write at the same time. The only
acceptable failure mode is "last writer wins"; a reader must never see a half-written JSON document
and no ``.tmp`` file may survive.

The probe uses a private temporary directory (never the real ``/memory``): six threads write the same
JSON artifact 8 times each while three threads continuously parse it. It then checks that every read
parsed, the final document is valid, and the directory contains no temporary leftovers. Score is the
fraction of those invariants that held (six boolean checks, 0..1); latency columns are the individual
atomic write times.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time

from harness import memory

from scripts.probes.base import Probe, make_result, register

WRITERS = 6
WRITES_PER_WRITER = 8
READERS = 3
BLOB = "x" * 512


@register
class MemoryProbe(Probe):
    name = "memory.atomic"
    component = "memory"

    def run(self) -> dict:
        root = tempfile.mkdtemp(prefix="adn-probe-memory-")
        target = os.path.join(root, "shared-state.json")
        expected = WRITERS * WRITES_PER_WRITER
        write_latencies: list[float] = []
        write_errors: list[str] = []
        partial_reads: list[str] = []
        read_count = 0
        stop = threading.Event()
        lock = threading.Lock()

        def writer(number: int) -> None:
            for sequence in range(WRITES_PER_WRITER):
                started = time.perf_counter()
                try:
                    memory.atomic_write_json(
                        target,
                        {"writer": number, "sequence": sequence, "blob": BLOB},
                    )
                except Exception as exc:  # noqa: BLE001 - the probe reports, never raises
                    with lock:
                        write_errors.append(f"writer {number}/{sequence}: {type(exc).__name__}: {exc}")
                finally:
                    with lock:
                        write_latencies.append((time.perf_counter() - started) * 1000.0)
                # Give the readers a scheduling window between writes, so the concurrency check is
                # not a race against a very fast filesystem. Sleep is outside the measured latency.
                time.sleep(0.0005)

        def reader() -> None:
            nonlocal read_count
            while not stop.is_set():
                try:
                    with open(target, encoding="utf-8") as handle:
                        payload = json.load(handle)
                    if not isinstance(payload, dict) or "sequence" not in payload:
                        with lock:
                            partial_reads.append("parsed but incomplete")
                    with lock:
                        read_count += 1
                except FileNotFoundError:
                    pass
                except (OSError, ValueError) as exc:
                    with lock:
                        partial_reads.append(f"{type(exc).__name__}: {exc}")
                # Yield instead of spinning at 100% of a core.
                stop.wait(0.001)

        threads = [threading.Thread(target=writer, args=(number,), name=f"probe-w{number}")
                   for number in range(WRITERS)]
        threads += [threading.Thread(target=reader, name=f"probe-r{number}")
                    for number in range(READERS)]
        try:
            for thread in threads[WRITERS:]:
                thread.start()
            for thread in threads[:WRITERS]:
                thread.start()
            for thread in threads[:WRITERS]:
                thread.join(timeout=10.0)
            stop.set()
            for thread in threads[WRITERS:]:
                thread.join(timeout=5.0)

            final = memory.read_json(target, default=None)
            leftovers = [name for name in os.listdir(root) if name.endswith(".tmp")]
            final_ok = (
                isinstance(final, dict)
                and isinstance(final.get("writer"), int)
                and isinstance(final.get("sequence"), int)
                and final.get("blob") == BLOB
            )
            checks = [
                ("every writer completed", len(write_latencies) == expected and not write_errors),
                ("no write raised", not write_errors),
                ("no reader saw partial JSON", not partial_reads),
                ("readers observed the file while writers ran", read_count > 0),
                ("final document parses completely", final_ok),
                ("no temporary files survive", not leftovers),
            ]
            passed = sum(1 for _label, ok in checks if ok)
            details = {
                "writers": WRITERS,
                "writes_per_writer": WRITES_PER_WRITER,
                "expected_writes": expected,
                "completed_writes": len(write_latencies),
                "readers": READERS,
                "concurrent_reads": read_count,
                "write_errors": write_errors[:5],
                "partial_reads": partial_reads[:5],
                "tmp_leftovers": leftovers,
                "final_valid": final_ok,
                "checks": [label for label, _ok in checks],
                "failures": [label for label, ok in checks if not ok],
            }
            notes = (
                f"{len(write_latencies)} atomic writes, {read_count} concurrent reads, "
                f"{len(partial_reads)} partial JSON"
            )
            return make_result(
                score=passed / len(checks),
                latencies=write_latencies,
                n=expected,
                notes=notes,
                details=details,
            )
        finally:
            stop.set()
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            shutil.rmtree(root, ignore_errors=True)
