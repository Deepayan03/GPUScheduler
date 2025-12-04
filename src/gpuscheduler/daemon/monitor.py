"""
monitor.py 

Provides a robust GPU monitor implementation (NVIDIA via nvidia-smi, Apple via powermetrics).
Timestamps, per-GPU parsing, thread-safe last-snapshot storage, and a callback-capable Monitor class.
"""

from __future__ import annotations

import threading
import subprocess
import time
import platform
import sys
import os
import shutil
from typing import Callable, Dict, Optional, Any, List


def runCmd(cmd: list[str], timeout: float = 1.5) -> Optional[str]:
    """
    Run a command and return stdout text, or None on failure.
    Centralizes subprocess.run usage and exceptions.

    If the environment variable GPUSCHED_DEBUG=1 is set, this will print the
    command, return code, stdout and stderr to stderr for debugging.
    """
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if int(os.environ.get("GPUSCHED_DEBUG", "0")) == 1:
            # Print debug info to stderr so it doesn't mix with normal stdout
            debugMsg = ["[runCmd-debug] CMD: " + " ".join(cmd), f"RET: {proc.returncode}", "STDOUT:", proc.stdout, "STDERR:", proc.stderr]
            print("\n".join(debugMsg), file=sys.stderr)
        if proc.returncode == 0:
            return proc.stdout
        return None
    except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
        if int(os.environ.get("GPUSCHED_DEBUG", "0")) == 1:
            print(f"[runCmd-debug] EXC: {e}", file=sys.stderr)
        return None


def nvidiaStatsAll() -> Optional[Dict[str, Any]]:
    """
    Parse nvidia-smi CSV output into a dict with a 'gpus' list.
    Each GPU dict has: index, gpuUtilPercent, gpuMemUsedMb, gpuMemTotalMb, gpuMemUtilPercent
    Returns None if nvidia-smi is not available or parsing fails.
    """
    out = runCmd(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=1.5,
    )

    if not out:
        return None

    gpus: List[Dict[str, Any]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # expected: index, mem_used, mem_total, util_gpu, util_mem
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
            memUsedMb = float(parts[1])
            memTotalMb = float(parts[2])
            utilGpuPct = float(parts[3])
            utilMemPct = float(parts[4])
        except ValueError:
            continue

        gpus.append(
            {
                "index": idx,
                "gpuUtilPercent": utilGpuPct,
                "gpuMemUsedMb": memUsedMb,
                "gpuMemTotalMb": memTotalMb,
                "gpuMemUtilPercent": utilMemPct,
            }
        )

    if not gpus:
        return None

    return {"backend": "nvidia-smi", "gpus": gpus, "raw": out}


def powermetricsStats() -> Optional[Dict[str, Any]]:
    """
    Run powermetrics once and extract GPU active residency on macOS ARM.
    Returns None if not on macOS ARM or if command fails.

    This function locates the powermetrics binary using shutil.which() so it
    will work when PATH changes under sudo. If the binary is not found it
    still attempts to run 'powermetrics' (legacy behavior).
    """
    if sys.platform != "darwin" or "arm" not in platform.machine().lower():
        return None

    # Locate powermetrics binary explicitly (helps when PATH differs under sudo)
    pmPath = shutil.which("powermetrics")
    if pmPath is None:
        # common locations for powermetrics on macOS
        for candidate in ("/usr/bin/powermetrics", "/usr/sbin/powermetrics", "/bin/powermetrics"):
            if os.path.exists(candidate):
                pmPath = candidate
                break
    if pmPath is None:
        pmPath = "powermetrics"

    # If running as root, call powermetrics directly; otherwise prefix with sudo
    try:
        isRoot = (getattr(os, "geteuid", lambda: 1)() == 0)
    except Exception:
        isRoot = False

    if isRoot:
        cmd = [pmPath, "--samplers", "gpu_power", "-n", "1"]
    else:
        # Use full path with sudo if available to help sudo find the binary
        cmd = ["sudo", pmPath, "--samplers", "gpu_power", "-n", "1"]

    out = runCmd(cmd, timeout=15.0)
    if not out:
        return None

    utilPct: Optional[float] = None
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith("gpu hw active residency:"):
            try:
                afterColon = line.split(":", 1)[1].strip()
                percentStr = afterColon.split("%", 1)[0].strip()
                utilPct = float(percentStr)
            except Exception:
                utilPct = None
            break

    return {"backend": "powermetrics", "gpuUtilPercent": utilPct, "raw": out}


def getGpuStatsSnapshot() -> Dict[str, Any]:
    """
    Unified snapshot:
      - backend: "nvidia-smi" | "powermetrics" | "none"
      - timestamp: epoch float
      - (nvidia) gpus: [ {...}, ... ]
      - (powermetrics) gpuUtilPercent: float | None
      - raw: raw output
    """
    ts = time.time()
    n = nvidiaStatsAll()
    if n is not None:
        n["timestamp"] = ts
        return n

    p = powermetricsStats()
    if p is not None:
        p["timestamp"] = ts
        return p

    return {"backend": "none", "timestamp": ts, "raw": ""}


class Monitor:
    """
    Background monitor that periodically polls GPU stats, stores last snapshot,
    and optionally calls a user callback with every snapshot.
    """

    def __init__(self, pollInterval: float = 2.0, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.pollInterval = float(pollInterval)
        self.callback = callback

        self._lastLock = threading.Lock()
        self._lastSnapshot: Optional[Dict[str, Any]] = None

        self._stopEvent = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _loop(self):
        while not self._stopEvent.is_set():
            snap = getGpuStatsSnapshot()

            with self._lastLock:
                self._lastSnapshot = snap

            # deliver snapshot to callback if present; swallow exceptions so monitor stays alive
            try:
                if self.callback:
                    self.callback(snap)
            except Exception:
                pass

            # responsive sleep in small increments so stop() can interrupt quickly
            slept = 0.0
            while slept < self.pollInterval and not self._stopEvent.is_set():
                time.sleep(0.2)
                slept += 0.2

    def start(self):
        """Start monitor thread (no-op if already running)."""
        if self._thread and self._thread.is_alive():
            return
        self._stopEvent.clear()
        self._thread = threading.Thread(target=self._loop, name="gpusched-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        """Stop monitor thread and join with timeout."""
        self._stopEvent.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def getLastStats(self) -> Optional[Dict[str, Any]]:
        """
        Thread-safe getter that returns a shallow copy of the last snapshot,
        or None if no snapshot is available yet.
        """
        with self._lastLock:
            return None if self._lastSnapshot is None else dict(self._lastSnapshot)


if __name__ == "__main__":
    def demoCallback(snap: Dict[str, Any]):
        backend = snap.get("backend")
        if backend == "nvidia-smi":
            gpus = snap.get("gpus", [])
            for g in gpus:
                print(f"[demo] GPU{g['index']} util={g['gpuUtilPercent']}% mem={g['gpuMemUsedMb']}/{g['gpuMemTotalMb']} MB")
        elif backend == "powermetrics":
            print(f"[demo] powermetrics util={snap.get('gpuUtilPercent')}")
        else:
            print("[demo] no gpu backend")

    mon = Monitor(pollInterval=3.0, callback=demoCallback)
    print("Starting monitor (Ctrl+C to stop). Note: powermetrics may require sudo.)")
    mon.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        mon.stop()
        print("Stopped.")
