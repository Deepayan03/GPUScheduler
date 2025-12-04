# job.py

# Job model used by the scheduler.

# Provides:
# - JobStatus (Enum)
# - Job dataclass (id, command, priority, status, timestamps, pid)
# - toDict() / fromDict() helpers for persistence
# - toJson() / fromJson() convenience methods

from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional,Dict, Any


class JobStatus(Enum):
#  Enum representing lifecycle state of a Job.
# Use strings as values so serialization to JSON is straightforward.   
    QUEUED="queued"
    RUNNING="running"
    PAUSED="paused"
    FINISHED="finished"
    FAILED="failed"
    CANCELLED="cancelled"


class Job:
    # Job model for scheduler.
    # Fields (camelCase):
    #   - id: unique string id (uuid4)
    #   - command: shell command (str)
    #   - priority: int (lower -> higher priority)
    #   - status: JobStatus
    #   - createdAt: epoch float
    #   - startedAt: Optional[float]
    #   - finishedAt: Optional[float]
    #   - pid: Optional[int]  # PID of running process (set by runner)
    #   - meta: dict for optional user metadata (e.g., owner, workspace)

    id:str = field(default_factory=lambda:str(uuid.uuid4()))
    command:str = ""
    priority:int = 10
    status:JobStatus = JobStatus.QUEUED
    createdAt: float = field(default_factory=time.time)
    startedAt: Optional[float] = None
    finishedAt: Optional[float] = None
    pid:Optional[int] = None
    meta: Dict[str,Any] = field(default_factory=dict)


# -------helpers--------

    def toDict(self) -> Dict[str,Any]:
        # Convert to plain dict (JSON-serializable).
        # Enums are converted to their value strings.
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, JobStatus) else str(self.status)
        return d
    
    @staticmethod
    def fromDict(d:Dict[str,Any]) ->"Job":
        #  Create a Job from a dict produced by toDict().
        # Expects keys matching the Job dataclass fields.
        js=dict(d) # shallow copying
        #Convert status string back to job status
        statusVal = js.get("status", JobStatus.QUEUED.value)
        try:
            js["status"] = JobStatus(statusVal)
        except ValueError:
            js["status"] = JobStatus.QUEUED

        return Job(
            id = js.get("id",str(uuid.uuid4())),
            command = js.get("command",""),
            priority = int(js.get("priority" , 10)),
            status=js["status"],
            createdAt=float(js.get("createdAt", time.time())),
            startedAt=(float(js["startedAt"]) if js.get("startedAt") is not None else None),
            finishedAt=(float(js["finishedAt"]) if js.get("finishedAt") is not None else None),
            pid=(int(js["pid"]) if js.get("pid") is not None else None),
            meta=js.get("meta", {}),
        )
    def toJson(self) -> str:
        # return compact json string for storage
        return json.dumps(self.toDict(), separators=(",", ":"), ensure_ascii=False)
    
    @staticmethod
    def fromJson(s:"str") ->"Job":
        d= json.loads(s)
        return Job.fromDict(d)
    

    def markStarted(self, pid: int) -> None:
        """
        Mark job as running and set PID and startedAt timestamp.
        """
        self.pid = int(pid)
        self.status = JobStatus.RUNNING
        self.startedAt = time.time()

    def markFinished(self, success: bool = True) -> None:
        """
        Mark job finished (FINISHED or FAILED) and set finishedAt.
        """
        self.finishedAt = time.time()
        self.pid = None
        self.status = JobStatus.FINISHED if success else JobStatus.FAILED

    def markCancelled(self) -> None:
        """
        Mark job cancelled by user.
        """
        self.finishedAt = time.time()
        self.pid = None
        self.status = JobStatus.CANCELLED

