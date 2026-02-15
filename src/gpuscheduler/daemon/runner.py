"""
runner.py

Advanced subprocess runner for GPU Scheduler.

Features:
- GPU binding via CUDA_VISIBLE_DEVICES
- Process group isolation
- Pause / Resume support
- Cooperative preemption support
- Watchdog helpers
- Per-job log file management
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import shlex
from typing import Optional, Dict

from gpuscheduler.daemon.job import Job


_processTable: Dict[int, subprocess.Popen] = {}
_jobByPid: Dict[int, Job] = {}

DEFAULT_LOG_DIR = "/tmp/gpusched"


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------

def _ensureLogDir(logDir: str) -> None:
    os.makedirs(logDir, exist_ok=True)


def _getLogPath(jobId: str, logDir: str) -> str:
    return os.path.join(logDir, f"{jobId}.log")


def _getProcessGroupPid(pid: int) -> int:
    if os.name == "posix":
        try:
            return os.getpgid(pid)
        except Exception:
            return pid
    return pid


# ----------------------------------------------------
# Core Execution
# ----------------------------------------------------

def startJob(job: Job, gpuIndex: int, logDir: str = DEFAULT_LOG_DIR) -> int:
    """
    Start job bound to specific GPU.
    """

    _ensureLogDir(logDir)
    logPath = _getLogPath(job.id, logDir)
    logFile = open(logPath, "ab", buffering=0)

    env = os.environ.copy()

    if gpuIndex is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpuIndex)

    # Properly split command into arguments
    cmd = shlex.split(job.command)

    popenArgs = {
        "stdout": logFile,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }

    if os.name == "posix":
        popenArgs["preexec_fn"] = os.setsid
    else:
        popenArgs["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0x00000200,
        )

    # IMPORTANT: shell=False (default)
    proc = subprocess.Popen(cmd, **popenArgs)

    pid = proc.pid

    _processTable[pid] = proc
    _jobByPid[pid] = job

    return pid


# ----------------------------------------------------
# Monitoring
# ----------------------------------------------------

def pollJob(pid: int) -> Optional[int]:
    proc = _processTable.get(pid)
    if not proc:
        return None
    return proc.poll()


def isAlive(pid: int) -> bool:
    return pollJob(pid) is None


# ----------------------------------------------------
# Signals & Control
# ----------------------------------------------------

def sendSignal(pid: int, sig: int) -> bool:
    pgid = _getProcessGroupPid(pid)
    try:
        if os.name == "posix":
            os.killpg(pgid, sig)
        else:
            os.kill(pgid, sig)
        return True
    except Exception:
        return False


def pauseJob(pid: int) -> bool:
    if os.name != "posix":
        return False
    return sendSignal(pid, signal.SIGSTOP)


def resumeJob(pid: int) -> bool:
    if os.name != "posix":
        return False
    return sendSignal(pid, signal.SIGCONT)


def sendPreemptSignal(pid: int) -> bool:
    if os.name != "posix":
        return False
    return sendSignal(pid, signal.SIGUSR1)


# ----------------------------------------------------
# Termination
# ----------------------------------------------------

def terminateJob(pid: int, timeout: float = 5.0) -> Optional[int]:
    if pid not in _processTable:
        return None

    sendSignal(pid, signal.SIGTERM)

    waited = 0.0
    while waited < timeout:
        code = pollJob(pid)
        if code is not None:
            _cleanup(pid)
            return code
        time.sleep(0.25)
        waited += 0.25

    sendSignal(pid, signal.SIGKILL)

    waited = 0.0
    while waited < 2.0:
        code = pollJob(pid)
        if code is not None:
            _cleanup(pid)
            return code
        time.sleep(0.25)
        waited += 0.25

    return None


def _cleanup(pid: int) -> None:
    _processTable.pop(pid, None)
    _jobByPid.pop(pid, None)


# ----------------------------------------------------
# Watchdog
# ----------------------------------------------------

def checkRuntimeExceeded(pid: int) -> bool:
    job = _jobByPid.get(pid)
    if not job:
        return False
    return job.hasExceededRuntime()


# ----------------------------------------------------
# Log Utilities
# ----------------------------------------------------

def readJobLogTail(jobId: str, logDir: str = DEFAULT_LOG_DIR, maxBytes: int = 4096) -> bytes:
    path = _getLogPath(jobId, logDir)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            toRead = min(size, maxBytes)
            f.seek(size - toRead, os.SEEK_SET)
            return f.read(toRead)
    except Exception:
        return b""