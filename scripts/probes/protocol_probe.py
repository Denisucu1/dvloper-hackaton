"""Probe: the end-to-end protocol smoke test as a subprocess.

``scripts/smoke.py`` drives ``main.py`` exactly like the arena does (prep -> start -> turns -> end ->
a second game -> a null-legal turn -> shutdown), all offline. This probe runs it as a child process
with a hard 10 s timeout and reports pass/fail plus the number of steps it completed. It is marked
``slow`` so ``report.py --fast`` can skip it while iterating on the component probes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from scripts.probes.base import ROOT, Probe, make_result, register

TIMEOUT_S = 10.0


@register
class ProtocolProbe(Probe):
    name = "protocol.smoke"
    component = "protocol"
    slow = True

    def run(self) -> dict:
        script = os.path.join(ROOT, "scripts", "smoke.py")
        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                [sys.executable, script],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
        except OSError as exc:
            timed_out = False
            returncode = None
            stdout = ""
            stderr = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        steps = sum(1 for line in stdout.splitlines() if line.strip().startswith("ok:"))
        passed = (not timed_out) and returncode == 0 and "SMOKE PASS" in stdout
        details = {
            "timeout_s": TIMEOUT_S,
            "returncode": returncode,
            "timed_out": timed_out,
            "steps": steps,
            "script": script,
            "stdout_tail": stdout[-800:],
            "stderr_tail": stderr[-800:],
        }
        if passed:
            notes = f"SMOKE PASS: {steps} protocol steps in {elapsed_ms:.0f} ms"
        elif timed_out:
            notes = f"smoke timed out after {TIMEOUT_S:.0f} s"
        else:
            tail = (stderr.strip() or stdout.strip() or "no output").splitlines()[-1][:120]
            notes = f"smoke failed (rc={returncode}): {tail}"
        return make_result(
            score=1.0 if passed else 0.0,
            latencies=[elapsed_ms],
            n=max(1, steps),
            notes=notes,
            details=details,
        )
