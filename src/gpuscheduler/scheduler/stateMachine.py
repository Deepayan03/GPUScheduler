"""
stateMachine.py

Strict job lifecycle controller.

Enforces valid state transitions.
Prevents illegal status mutations.
"""

from __future__ import annotations

import time
from typing import Set

from gpuscheduler.daemon.job import Job, JobStatus


class JobStateMachine:

    # Allowed transitions map
    _allowedTransitions = {
        JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED},
        JobStatus.RUNNING: {
            JobStatus.PAUSED,
            JobStatus.FINISHED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        },
        JobStatus.PAUSED: {
            JobStatus.RUNNING,
            JobStatus.CANCELLED,
        },
        JobStatus.FINISHED: set(),
        JobStatus.FAILED: set(),
        JobStatus.CANCELLED: set(),
    }

    # ----------------------------------------------------
    # Transition Logic
    # ----------------------------------------------------

    @classmethod
    def transition(cls, job: Job, newStatus: JobStatus) -> None:
        current = job.status

        if newStatus not in cls._allowedTransitions.get(current, set()):
            raise ValueError(
                f"Illegal transition from {current.value} to {newStatus.value}"
            )

        # Apply transition
        job.status = newStatus

        # Apply timestamp logic
        now = time.time()

        if newStatus == JobStatus.RUNNING:
            job.startedAt = now

        if newStatus in {JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELLED}:
            job.finishedAt = now
            job.pid = None
            job.assignedGpu = None

    # ----------------------------------------------------
    # Convenience Wrappers
    # ----------------------------------------------------

    @classmethod
    def start(cls, job: Job) -> None:
        cls.transition(job, JobStatus.RUNNING)

    @classmethod
    def pause(cls, job: Job) -> None:
        cls.transition(job, JobStatus.PAUSED)

    @classmethod
    def resume(cls, job: Job) -> None:
        cls.transition(job, JobStatus.RUNNING)

    @classmethod
    def finish(cls, job: Job, success: bool = True) -> None:
        cls.transition(job, JobStatus.FINISHED if success else JobStatus.FAILED)

    @classmethod
    def cancel(cls, job: Job) -> None:
        cls.transition(job, JobStatus.CANCELLED)