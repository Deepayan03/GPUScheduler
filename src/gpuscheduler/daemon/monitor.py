"""
monitor.py 

Provides a robust GPU monitor implementation (NVIDIA via nvidia-smi, Apple via powermetrics).
Timestamps, per-GPU parsing, thread-safe last-snapshot storage, and a callback-capable Monitor class.

Now includes:
- Utilization delta detection
- Intelligent scheduler wake triggering
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


# ----------------------------------------------------
# Command Runner
# ----------------------------------------------------

def runCmd(cmd: list[str], timeout: float = 1.5) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if int(os.environ.get("GPUSCHED_DEBUG", "0")) == 1:
            debugMsg = [
                "[runCmd-debug] CMD: " + " ".join(cmd),
                f"RET: {proc.returncode}",
                "STDOUT:",
                proc.stdout,
                "STDERR:",
                proc.stderr,
            ]
            print("\n".join(debugMsg), file=sys.stderr)

        if proc.returncode == 0:
            return proc.stdout

        return None

    except (subprocess.SubprocessError, FileNotFoundError, PermissionError) as e:
        if int(os.environ.get("GPUSCHED_DEBUG", "0")) == 1:
            print(f"[runCmd-debug] EXC: {e}", file=sys.stderr)
        return None


# ----------------------------------------------------
# NVIDIA Stats
# ----------------------------------------------------

def nvidiaStatsAll() -> Optional[Dict[str, Any]]:
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


# ----------------------------------------------------
# macOS powermetrics Stats
# ----------------------------------------------------

def powermetricsStats() -> Optional[Dict[str, Any]]:
    if sys.platform != "darwin" or "arm" not in platform.machine().lower():
        return None

    pmPath = shutil.which("powermetrics")
    if pmPath is None:
        for candidate in ("/usr/bin/powermetrics", "/usr/sbin/powermetrics", "/bin/powermetrics"):
            if os.path.exists(candidate):
                pmPath = candidate
                break
    if pmPath is None:
        pmPath = "powermetrics"

    try:
        isRoot = (getattr(os, "geteuid", lambda: 1)() == 0)
    except Exception:
        isRoot = False

    if isRoot:
        cmd = [pmPath, "--samplers", "gpu_power", "-n", "1"]
    else:
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


# ----------------------------------------------------
# Snapshot
# ----------------------------------------------------

def getGpuStatsSnapshot() -> Dict[str, Any]:
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


# ----------------------------------------------------
# Monitor Class (Delta-Aware)
# ----------------------------------------------------

class Monitor:

    def __init__(
        self,
        pollInterval: float = 2.0,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        utilDeltaThreshold: float = 10.0,
    ):
        self.pollInterval = float(pollInterval)
        self.callback = callback
        self.utilDeltaThreshold = utilDeltaThreshold

        self._lastLock = threading.Lock()
        self._lastSnapshot: Optional[Dict[str, Any]] = None

        self._previousUtil: Optional[float] = None

        self._stopEvent = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----------------------------------------------------
    # Util Extraction
    # ----------------------------------------------------

    def _extractUtil(self, snap: Dict[str, Any]) -> float:
        if not snap:
            return 0.0

        if snap.get("backend") == "nvidia-smi":
            gpus = snap.get("gpus", [])
            if not gpus:
                return 0.0
            return max(g.get("gpuUtilPercent", 0.0) for g in gpus)

        if snap.get("backend") == "powermetrics":
            return snap.get("gpuUtilPercent", 0.0) or 0.0

        return 0.0

    # ----------------------------------------------------
    # Background Loop
    # ----------------------------------------------------

    def _loop(self):
        while not self._stopEvent.is_set():
            snap = getGpuStatsSnapshot()

            with self._lastLock:
                self._lastSnapshot = snap

            if self.callback:
                currentUtil = self._extractUtil(snap)

                shouldNotify = False

                if self._previousUtil is None:
                    shouldNotify = True
                elif abs(currentUtil - self._previousUtil) >= self.utilDeltaThreshold:
                    shouldNotify = True

                if shouldNotify:
                    self._previousUtil = currentUtil
                    try:
                        self.callback(snap)
                    except Exception:
                        pass

            slept = 0.0
            while slept < self.pollInterval and not self._stopEvent.is_set():
                time.sleep(0.2)
                slept += 0.2

    # ----------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stopEvent.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="gpusched-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stopEvent.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def getLastStats(self) -> Optional[Dict[str, Any]]:
        with self._lastLock:
            return None if self._lastSnapshot is None else dict(self._lastSnapshot)
        