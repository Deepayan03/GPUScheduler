"""
job.py

Advanced Job model used by the scheduler.

Supports:
- Multi-GPU requirements
- Memory constraints
- Preemption flags
- Runtime limits
- Dynamic GPU assignment
"""

from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    # Core identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command: str = ""
    priority: int = 10

    # Resource requirements
    requiredGpus: int = 1
    requiredMemMb: Optional[int] = None
    exclusive: bool = True

    # Runtime controls
    preemptible: bool = True
    maxRuntimeSeconds: Optional[int] = None

    # ACPR controls
    trustPolicy: Dict[str, Any] = field(default_factory=dict)
    checkpointPath: Optional[str] = None

    # Dynamic assignment (set by scheduler)
    assignedGpu: Optional[int] = None
    assignedGpus: List[int] = field(default_factory=list)

    # Lifecycle state
    status: JobStatus = JobStatus.QUEUED
    createdAt: float = field(default_factory=time.time)
    startedAt: Optional[float] = None
    finishedAt: Optional[float] = None
    pid: Optional[int] = None

    # Optional user metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    # ACPR proof ledger fields
    proofStatus: str = "disabled"
    proofChain: List[Dict[str, Any]] = field(default_factory=list)
    lastAttestation: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------
    # Serialization
    # ----------------------------------------------------

    def toDict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @staticmethod
    def fromDict(d: Dict[str, Any]) -> "Job":
        return Job(
            id=d.get("id", str(uuid.uuid4())),
            command=d.get("command", ""),
            priority=int(d.get("priority", 10)),
            requiredGpus=int(d.get("requiredGpus", 1)),
            requiredMemMb=d.get("requiredMemMb"),
            exclusive=bool(d.get("exclusive", True)),
            preemptible=bool(d.get("preemptible", True)),
            maxRuntimeSeconds=d.get("maxRuntimeSeconds"),
            trustPolicy=d.get("trustPolicy", {}) or {},
            checkpointPath=d.get("checkpointPath"),
            assignedGpu=d.get("assignedGpu"),
            assignedGpus=[
                int(g)
                for g in (
                    d.get("assignedGpus")
                    if d.get("assignedGpus") is not None
                    else (
                        [d.get("assignedGpu")]
                        if d.get("assignedGpu") is not None
                        else []
                    )
                )
                if g is not None
            ],
            status=JobStatus(d.get("status", "queued")),
            createdAt=float(d.get("createdAt", time.time())),
            startedAt=d.get("startedAt"),
            finishedAt=d.get("finishedAt"),
            pid=d.get("pid"),
            meta=d.get("meta", {}),
            proofStatus=d.get("proofStatus", "disabled"),
            proofChain=d.get("proofChain", []) or [],
            lastAttestation=d.get("lastAttestation"),
        )

    def toJson(self) -> str:
        return json.dumps(self.toDict())

    @staticmethod
    def fromJson(s: str) -> "Job":
        return Job.fromDict(json.loads(s))

    # ----------------------------------------------------
    # State transitions
    # ----------------------------------------------------

    def markStarted(self, pid: int, gpuIndex: int) -> None:
        self.pid = int(pid)
        self.status = JobStatus.RUNNING
        self.startedAt = time.time()
        self.assignedGpu = gpuIndex
        self.assignedGpus = [gpuIndex]

    def markFinished(self, success: bool = True) -> None:
        self.finishedAt = time.time()
        self.pid = None
        self.assignedGpu = None
        self.assignedGpus = []
        self.status = JobStatus.FINISHED if success else JobStatus.FAILED

    def markCancelled(self) -> None:
        self.finishedAt = time.time()
        self.pid = None
        self.assignedGpu = None
        self.assignedGpus = []
        self.status = JobStatus.CANCELLED

    # ----------------------------------------------------
    # Runtime helpers
    # ----------------------------------------------------

    def hasExceededRuntime(self) -> bool:
        if self.maxRuntimeSeconds is None or self.startedAt is None:
            consumed = float(self.meta.get("runTimeConsumedSeconds", 0.0))
            return consumed > float(self.maxRuntimeSeconds or 0.0)

        consumed = float(self.meta.get("runTimeConsumedSeconds", 0.0))
        consumed += max(0.0, time.time() - self.startedAt)
        return consumed > float(self.maxRuntimeSeconds)
