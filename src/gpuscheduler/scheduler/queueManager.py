"""
queueManager.py

Thread-safe QueueManager for GPU Scheduler.

Responsibilities:
- Maintain a schedulable queue (queued + paused jobs)
- Track running jobs per GPU
- Support preempt/requeue cycles
- Provide dynamic effective-priority ordering (aging + fairness penalty)
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Dict, List, Optional, Tuple

from gpuscheduler.daemon.job import Job, JobStatus


class QueueManager:
    def __init__(self, agingFactor: float = 0.0):
        self._heap: List[Tuple[int, float, str]] = []
        self._jobMap: Dict[str, Job] = {}
        self._runningByGpu: Dict[int, List[Job]] = {}
        self.defaultAgingFactor = max(0.0, float(agingFactor))
        self._lock = threading.RLock()

    # ----------------------------------------------------
    # Job Submission and cancellation
    # ----------------------------------------------------

    def addJob(self, job: Job) -> None:
        with self._lock:
            self._jobMap[job.id] = job
            heapq.heappush(
                self._heap,
                (job.priority, job.createdAt, job.id),
            )

    def removeJob(self, jobId: str) -> None:
        with self._lock:
            self._heap = [item for item in self._heap if item[2] != jobId]
            self._jobMap.pop(jobId, None)

    # ----------------------------------------------------
    # Scheduling
    # ----------------------------------------------------

    def findAndAssignJob(
        self,
        allGpuIndices: List[int],
        agingFactor: Optional[float] = None,
    ) -> Optional[Tuple[Job, List[int]]]:
        with self._lock:
            if not self._heap:
                return None

            freeGpus = self._getFreeGpus(allGpuIndices)
            if not freeGpus:
                return None

            queuedJobs = self.getQueuedJobs(
                agingFactor=agingFactor,
                includePaused=True,
            )
            for job in queuedJobs:
                if job.requiredGpus > len(freeGpus):
                    continue

                allocated = list(freeGpus[: job.requiredGpus])
                self._assignJobNoLock(job, allocated)
                return job, allocated

            return None

    def assignJobToGpus(
        self,
        jobId: str,
        allocatedGpus: List[int],
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobMap.get(jobId)
            if job is None:
                return None
            if job.status not in {JobStatus.QUEUED, JobStatus.PAUSED}:
                return None

            self._assignJobNoLock(job, allocatedGpus)
            return job

    # ----------------------------------------------------
    # Release / Requeue
    # ----------------------------------------------------

    def releaseJob(self, job: Job) -> None:
        with self._lock:
            for gpu, jobs in list(self._runningByGpu.items()):
                if job in jobs:
                    jobs.remove(job)
                if not jobs:
                    self._runningByGpu.pop(gpu, None)

    def requeueJob(
        self,
        job: Job,
        refreshTimestamp: bool = True,
        targetStatus: JobStatus = JobStatus.QUEUED,
        preservePid: bool = False,
    ) -> None:
        with self._lock:
            if targetStatus not in {JobStatus.QUEUED, JobStatus.PAUSED}:
                raise ValueError("targetStatus must be queued or paused")

            self._jobMap[job.id] = job
            job.status = targetStatus
            if not preservePid:
                job.pid = None
            job.assignedGpu = None
            job.assignedGpus = []

            if refreshTimestamp:
                job.createdAt = time.time()

            heapq.heappush(
                self._heap,
                (job.priority, job.createdAt, job.id),
            )

    # ----------------------------------------------------
    # Introspection
    # ----------------------------------------------------

    def getRunningJobs(self) -> List[Job]:
        with self._lock:
            seen: Dict[str, Job] = {}
            for jobs in self._runningByGpu.values():
                for job in jobs:
                    seen[job.id] = job
            return list(seen.values())

    def getRunningJobsOnGpu(self, gpuIndex: int) -> List[Job]:
        with self._lock:
            return list(self._runningByGpu.get(gpuIndex, []))

    def getRunningCountByUser(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for job in self.getRunningJobs():
                user = self._extractJobUser(job)
                counts[user] = counts.get(user, 0) + 1
            return counts

    def peekHighestPriorityQueued(
        self,
        agingFactor: Optional[float] = None,
    ) -> Optional[Job]:
        jobs = self.getQueuedJobs(
            agingFactor=agingFactor,
            includePaused=True,
        )
        if not jobs:
            return None
        return jobs[0]

    def getQueuedJobs(
        self,
        agingFactor: Optional[float] = None,
        now: Optional[float] = None,
        includePaused: bool = True,
        fairnessPenaltyByUser: Optional[Dict[str, float]] = None,
    ) -> List[Job]:
        with self._lock:
            nowTs = float(now if now is not None else time.time())
            effectiveAging = (
                self.defaultAgingFactor
                if agingFactor is None
                else max(0.0, float(agingFactor))
            )

            rows: List[Tuple[float, float, str, Job]] = []
            seenIds = set()
            for _, _, jobId in self._heap:
                if jobId in seenIds:
                    continue
                seenIds.add(jobId)

                job = self._jobMap.get(jobId)
                if job is None:
                    continue

                if job.status == JobStatus.QUEUED:
                    pass
                elif includePaused and job.status == JobStatus.PAUSED:
                    pass
                else:
                    continue

                effectivePriority = self.getEffectivePriority(
                    job,
                    agingFactor=effectiveAging,
                    now=nowTs,
                    fairnessPenaltyByUser=fairnessPenaltyByUser,
                )
                rows.append(
                    (
                        effectivePriority,
                        float(job.createdAt),
                        job.id,
                        job,
                    )
                )

            rows.sort(key=lambda row: (row[0], row[1], row[2]))
            return [row[3] for row in rows]

    def getEffectivePriority(
        self,
        job: Job,
        agingFactor: Optional[float] = None,
        now: Optional[float] = None,
        fairnessPenaltyByUser: Optional[Dict[str, float]] = None,
    ) -> float:
        nowTs = float(now if now is not None else time.time())
        effectiveAging = (
            self.defaultAgingFactor
            if agingFactor is None
            else max(0.0, float(agingFactor))
        )

        waitSeconds = max(0.0, nowTs - float(job.createdAt))
        effectivePriority = float(job.priority) - (effectiveAging * waitSeconds)

        # Slight bias to resume paused work when possible.
        if job.status == JobStatus.PAUSED:
            effectivePriority -= 0.05

        if fairnessPenaltyByUser:
            user = self._extractJobUser(job)
            effectivePriority += float(fairnessPenaltyByUser.get(user, 0.0))

        return effectivePriority

    def getFreeGpus(self, allGpuIndices: List[int]) -> List[int]:
        with self._lock:
            return self._getFreeGpus(allGpuIndices)

    # ----------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------

    def _assignJobNoLock(
        self,
        job: Job,
        allocatedGpus: List[int],
    ) -> None:
        for gpu in allocatedGpus:
            self._runningByGpu.setdefault(gpu, []).append(job)

        self._heap = [item for item in self._heap if item[2] != job.id]
        heapq.heapify(self._heap)

    def _getFreeGpus(self, allGpuIndices: List[int]) -> List[int]:
        free = []
        for gpu in allGpuIndices:
            runningJobs = self._runningByGpu.get(gpu, [])
            if not runningJobs:
                free.append(gpu)
                continue

            # Shared allowed only if all running jobs are non-exclusive.
            if all(not job.exclusive for job in runningJobs):
                free.append(gpu)
        return free

    def _extractJobUser(self, job: Job) -> str:
        user = job.meta.get("user")
        if isinstance(user, str) and user.strip():
            return user.strip()
        return "default"
