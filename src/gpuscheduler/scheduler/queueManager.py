"""
queueManager.py

Advanced multi-GPU aware scheduling queue.

Responsibilities:
- Maintain priority heap
- Support requiredGpus
- Handle exclusive vs shared jobs
- Apply aging
- Allocate GPUs
- NEVER mutate job lifecycle state
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from gpuscheduler.daemon.job import Job, JobStatus


class QueueManager:
    def __init__(self, agingFactor: float = 0.01):
        self._heap: List[Tuple[float, float, str]] = []
        self._jobMap: Dict[str, Job] = {}

        # gpuIndex -> list of running jobs
        self._runningByGpu: Dict[int, List[Job]] = {}

        self._agingFactor = agingFactor

    # ----------------------------------------------------
    # Job Management
    # ----------------------------------------------------

    def addJob(self, job: Job) -> None:
        self._jobMap[job.id] = job
        heapq.heappush(
            self._heap,
            (
                job.priority,
                job.createdAt,
                job.id,
            ),
        )

    def removeJob(self, jobId: str) -> None:
        self._jobMap.pop(jobId, None)
        self._heap = [item for item in self._heap if item[2] != jobId]
        heapq.heapify(self._heap)

    # ----------------------------------------------------
    # Aging
    # ----------------------------------------------------

    def _computeEffectivePriority(self, job: Job) -> float:
        waited = time.time() - job.createdAt
        return job.priority - (waited * self._agingFactor)

    def _rebuildHeapWithAging(self) -> None:
        newHeap = []
        for _, _, jobId in self._heap:
            job = self._jobMap.get(jobId)
            if job and job.status == JobStatus.QUEUED:
                eff = self._computeEffectivePriority(job)
                heapq.heappush(newHeap, (eff, job.createdAt, job.id))
        self._heap = newHeap

    # ----------------------------------------------------
    # GPU Allocation Logic
    # ----------------------------------------------------

    def _getFreeGpus(self, allGpuIndices: List[int]) -> List[int]:
        free = []

        for gpu in allGpuIndices:
            runningJobs = self._runningByGpu.get(gpu, [])

            if not runningJobs:
                free.append(gpu)
            else:
                # Shared allowed only if all running jobs are non-exclusive
                if all(not job.exclusive for job in runningJobs):
                    free.append(gpu)

        return free

    def findAndAssignJob(
        self,
        allGpuIndices: List[int],
    ) -> Optional[Tuple[Job, List[int]]]:
        """
        Allocate GPUs for highest-priority QUEUED job.

        Returns:
            (job, allocatedGpuList)
        or
            None
        """

        if not self._heap:
            return None

        self._rebuildHeapWithAging()

        freeGpus = self._getFreeGpus(allGpuIndices)

        if not freeGpus:
            return None

        tempHeap = list(self._heap)
        heapq.heapify(tempHeap)

        while tempHeap:
            _, _, jobId = heapq.heappop(tempHeap)
            job = self._jobMap.get(jobId)

            if not job:
                continue

            if job.status != JobStatus.QUEUED:
                continue

            if job.requiredGpus <= len(freeGpus):
                allocated = freeGpus[: job.requiredGpus]

                # Track allocation ONLY (no status mutation)
                for gpu in allocated:
                    self._runningByGpu.setdefault(gpu, []).append(job)

                # Remove from heap
                self._heap = [
                    item for item in self._heap if item[2] != job.id
                ]
                heapq.heapify(self._heap)

                return job, allocated

        return None

    def releaseJob(self, job: Job) -> None:
        """
        Free GPUs used by job.
        """
        for gpu, jobs in list(self._runningByGpu.items()):
            if job in jobs:
                jobs.remove(job)
            if not jobs:
                self._runningByGpu.pop(gpu, None)

    # ----------------------------------------------------
    # State Inspection
    # ----------------------------------------------------

    def getRunningJobs(self) -> Dict[int, List[Job]]:
        return {gpu: list(jobs) for gpu, jobs in self._runningByGpu.items()}

    def getQueueSize(self) -> int:
        return len(self._heap)

    def hasPendingJobs(self) -> bool:
        return any(
            job.status == JobStatus.QUEUED
            for job in self._jobMap.values()
        )