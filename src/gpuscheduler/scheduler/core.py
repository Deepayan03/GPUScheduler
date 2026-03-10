"""
core.py

Event-driven SchedulerCore.

Features:
- Condition-variable based scheduling loop
- Immediate wake on job submission
- Immediate wake on job completion
- Hybrid preemption
- Clean lifecycle handling
"""

from __future__ import annotations

import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from gpuscheduler.daemon import runner
from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.scheduler.queueManager import QueueManager
from gpuscheduler.scheduler.policy import SchedulingPolicy
from gpuscheduler.scheduler.stateMachine import JobStateMachine
from gpuscheduler.daemon.monitor import Monitor
from gpuscheduler.security.attestor import MockAttestor, isAttestationCompliant
from gpuscheduler.security.proof import ProofLedger, hashFile


class SchedulerCore:

    def __init__(
        self,
        gpuIndices: Optional[List[int]] = None,
        agingFactor: float = 0.002,
        maxConcurrentPerUser: Optional[int] = 2,
        fairSharePriorityPenalty: float = 0.75,
        placementMode: str = "fragmentation_aware",
        victimWeights: Optional[Dict[str, float]] = None,
    ):
        self.gpuIndices = gpuIndices or [0]
        self.agingFactor = max(0.0, float(agingFactor))
        self.maxConcurrentPerUser = (
            None
            if maxConcurrentPerUser is None
            else max(0, int(maxConcurrentPerUser))
        )
        self.fairSharePriorityPenalty = max(0.0, float(fairSharePriorityPenalty))
        self.placementMode = placementMode
        self.victimWeights = {
            "priorityGap": 1.5,
            "memory": 1.0,
            "gpuCount": 1.0,
            "runtime": 0.25,
            "preemptionHistory": 2.0,
            "userShareBias": 0.75,
        }
        if victimWeights:
            for key, value in victimWeights.items():
                self.victimWeights[key] = float(value)

        self.queueManager = QueueManager(agingFactor=self.agingFactor)
        self.policy = SchedulingPolicy()

        self.monitor = Monitor(
            pollInterval=2.0,
            callback=self._onMonitorUpdate,
            utilDeltaThreshold=10.0,
        )

        self.attestor = MockAttestor()
        self.proofLedger = ProofLedger()

        self._condition = threading.Condition()
        self._stop = False
        self._terminalJobsById: Dict[str, Job] = {}
        self._terminalOrder: List[str] = []
        self._terminalLock = threading.RLock()

    def _onMonitorUpdate(self, snapshot):
        # Wake scheduler when GPU stats change
        with self._condition:
            self._condition.notify()

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def submitJob(self, job: Job) -> None:
        with self._condition:
            if not isinstance(job.meta, dict):
                job.meta = {}
            if not job.meta.get("user"):
                job.meta["user"] = "default"
            job.meta.setdefault("preemptionCount", 0)
            job.meta.setdefault("runTimeConsumedSeconds", 0.0)
            self.queueManager.addJob(job)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify()

    def getTerminalJobs(self) -> List[Job]:
        with self._terminalLock:
            return [
                self._cloneJob(self._terminalJobsById[jobId])
                for jobId in self._terminalOrder
                if jobId in self._terminalJobsById
            ]

    # ----------------------------------------------------
    # Core Loop
    # ----------------------------------------------------

    def run(self):
        print("Starting SchedulerCore...")
        self.monitor.start()

        while True:

            with self._condition:
                if self._stop:
                    break

            # Phase 1: Check for finished jobs
            finishedSomething = self._handleCompletions()

            # Phase 2: Try preemption
            preempted = self._handlePreemption()

            if preempted:
                continue  # restart loop cleanly

            # Phase 3: Try scheduling new jobs
            scheduled = self._handleScheduling()

            # Wait for next event if nothing happened
            if not (finishedSomething or scheduled):
                with self._condition:
                    self._condition.wait(timeout=2.0)

        self.monitor.stop()
        print("Scheduler stopped.")

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify()

    def _getJobUser(self, job: Job) -> str:
        user = job.meta.get("user")
        if isinstance(user, str) and user.strip():
            return user.strip()
        return "default"

    def _getRunningUserCounts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for job in self.queueManager.getRunningJobs():
            user = self._getJobUser(job)
            counts[user] = counts.get(user, 0) + 1
        return counts

    def _buildUserFairnessPenalty(self) -> Dict[str, float]:
        if self.fairSharePriorityPenalty <= 0.0:
            return {}
        runningCounts = self._getRunningUserCounts()
        penalties: Dict[str, float] = {}
        for user, count in runningCounts.items():
            if count <= 1:
                continue
            penalties[user] = (count - 1) * self.fairSharePriorityPenalty
        return penalties

    def _canRunUnderFairShare(
        self,
        job: Job,
        runningCounts: Dict[str, int],
    ) -> bool:
        if self.maxConcurrentPerUser in {None, 0}:
            return True
        user = self._getJobUser(job)
        return runningCounts.get(user, 0) < int(self.maxConcurrentPerUser)

    def _runtimeSeconds(self, job: Job) -> float:
        consumed = float(job.meta.get("runTimeConsumedSeconds", 0.0))
        if job.status == JobStatus.RUNNING and job.startedAt is not None:
            consumed += max(0.0, time.time() - job.startedAt)
        return consumed

    def _bumpRuntimeConsumed(self, job: Job) -> None:
        if job.startedAt is None:
            return
        elapsed = max(0.0, time.time() - job.startedAt)
        job.meta["runTimeConsumedSeconds"] = (
            float(job.meta.get("runTimeConsumedSeconds", 0.0)) + elapsed
        )
        # Resume should track a fresh active span.
        job.startedAt = None

    def _victimPreemptionCount(self, job: Job) -> int:
        try:
            return int(job.meta.get("preemptionCount", 0))
        except Exception:
            return 0

    def _victimScore(
        self,
        candidate: Job,
        victim: Job,
        snapshot,
        runningCountsByUser: Optional[Dict[str, int]] = None,
    ) -> float:
        weights = self.victimWeights
        priorityGap = max(0.0, float(victim.priority - candidate.priority))
        victimMemMb = float(victim.requiredMemMb or 0.0)
        victimGpuCount = float(max(1, victim.requiredGpus))
        runtimeMinutes = self._runtimeSeconds(victim) / 60.0
        preemptionHistory = float(self._victimPreemptionCount(victim))

        # For memory pressure preemption, larger-memory victims are preferred.
        if candidate.requiredMemMb is not None and candidate.requiredMemMb > 0:
            memoryTerm = -victimMemMb
        else:
            memoryTerm = victimMemMb

        userBias = 0.0
        if runningCountsByUser:
            victimUser = self._getJobUser(victim)
            userBias = float(max(0, runningCountsByUser.get(victimUser, 0) - 1))

        score = 0.0
        score += -weights["priorityGap"] * priorityGap
        score += weights["memory"] * memoryTerm
        score += weights["gpuCount"] * victimGpuCount
        score += weights["runtime"] * runtimeMinutes
        score += weights["preemptionHistory"] * preemptionHistory
        score += -weights["userShareBias"] * userBias
        return score

    # ----------------------------------------------------
    # Completion Handling
    # ----------------------------------------------------

    def _handleCompletions(self) -> bool:
        somethingChanged = False

        for job in self.queueManager.getRunningJobs():
            if job.pid is None:
                continue

            if job.maxRuntimeSeconds is not None and job.hasExceededRuntime():
                gpuIndex = self._getPrimaryAssignedGpu(job)
                runner.terminateJob(job.pid)
                job.meta["runTimeConsumedSeconds"] = self._runtimeSeconds(job)
                JobStateMachine.finish(job, success=False)
                job.meta["failureReason"] = "max_runtime_exceeded"
                self.queueManager.releaseJob(job)
                self._appendProofEvent(
                    job,
                    eventType="timeout",
                    gpuIndex=gpuIndex,
                    attestation=job.lastAttestation,
                    extra={"failureReason": "max_runtime_exceeded"},
                )
                self._recordTerminalJob(job)
                somethingChanged = True

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Job {job.id} failed: exceeded max runtime "
                    f"({job.maxRuntimeSeconds}s)"
                )

                self._wake()
                continue

            exitCode = runner.pollJob(job.pid)
            if exitCode is not None:
                gpuIndex = self._getPrimaryAssignedGpu(job)
                job.meta["runTimeConsumedSeconds"] = self._runtimeSeconds(job)
                JobStateMachine.finish(job)
                self.queueManager.releaseJob(job)
                self._appendProofEvent(
                    job,
                    eventType="finish",
                    gpuIndex=gpuIndex,
                    attestation=job.lastAttestation,
                    extra={"exitCode": exitCode},
                )
                self._recordTerminalJob(job)
                somethingChanged = True

                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Job {job.id} finished with exit code {exitCode}"
                )

                self._wake()

        return somethingChanged

    # ----------------------------------------------------
    # Preemption
    # ----------------------------------------------------

    def _handlePreemption(self) -> bool:
        snapshot = self.monitor.getLastStats()
        if snapshot is None:
            return False

        queuedJobs = self.queueManager.getQueuedJobs(
            agingFactor=self.agingFactor,
            includePaused=True,
            fairnessPenaltyByUser=self._buildUserFairnessPenalty(),
        )
        if not queuedJobs:
            return False

        runningCountsByUser = self._getRunningUserCounts()
        candidate = None
        for queuedJob in queuedJobs:
            if self._canRunUnderFairShare(queuedJob, runningCountsByUser):
                candidate = queuedJob
                break
        if candidate is None:
            return False

        # If the highest-priority queued job can already be scheduled, avoid preemption.
        if self._findPlacementForJob(
            candidate,
            snapshot,
            enforcePolicy=True,
        ) is not None:
            return False

        memoryVictims = self._selectMemoryPreemptionVictims(
            candidate,
            snapshot,
            runningCountsByUser=runningCountsByUser,
        )
        if memoryVictims is not None:
            targetGpus, victims = memoryVictims
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Memory preemption on GPUs {targetGpus} "
                f"for job {candidate.id}: "
                f"{[job.id for job in victims]}"
            )
            self._preemptJobs(
                victims=victims,
                candidateJobId=candidate.id,
                reason="memory",
                targetGpus=targetGpus,
            )
            return True

        # Smart victim selection for priority preemption.
        candidateVictims: Dict[str, Job] = {}
        for gpuIndex in self.gpuIndices:
            currentUtil = self._getGpuUtil(snapshot, gpuIndex)
            for runningJob in self.queueManager.getRunningJobsOnGpu(gpuIndex):
                if not runningJob.preemptible:
                    continue
                if candidate.priority >= runningJob.priority:
                    continue
                if not self.policy.shouldPreempt(
                    gpuIndex,
                    currentUtil,
                    runningJob.priority,
                    candidate.priority,
                ):
                    continue
                candidateVictims[runningJob.id] = runningJob

        if candidateVictims:
            rankedVictims = sorted(
                candidateVictims.values(),
                key=lambda victim: (
                    self._victimScore(
                        candidate,
                        victim,
                        snapshot,
                        runningCountsByUser=runningCountsByUser,
                    ),
                    victim.createdAt,
                    victim.id,
                ),
            )
            victim = rankedVictims[0]
            targetGpus = self._getJobAssignedGpus(victim)
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Scored preemption victim {victim.id} "
                f"for higher priority job {candidate.id}"
            )
            self._preemptJobs(
                victims=[victim],
                candidateJobId=candidate.id,
                reason="priority",
                targetGpus=targetGpus,
            )
            return True

        return False

    # ----------------------------------------------------
    # Scheduling
    # ----------------------------------------------------

    def _handleScheduling(self) -> bool:
        snapshot = self.monitor.getLastStats()
        fairnessPenalty = self._buildUserFairnessPenalty()
        queuedJobs = self.queueManager.getQueuedJobs(
            agingFactor=self.agingFactor,
            includePaused=True,
            fairnessPenaltyByUser=fairnessPenalty,
        )
        if not queuedJobs:
            return False

        runningCountsByUser = self._getRunningUserCounts()

        for candidate in queuedJobs:
            if not self._canRunUnderFairShare(candidate, runningCountsByUser):
                continue

            allocatedGpus = self._findPlacementForJob(
                candidate,
                snapshot,
                enforcePolicy=False,
            )
            if not allocatedGpus:
                continue

            if not self._canPlaceJobOnGpus(
                candidate,
                allocatedGpus,
                snapshot,
                enforcePolicy=True,
            ):
                continue

            acprEnabled = self._isAcpEnabled(candidate)
            attestation = None
            allow = True
            if acprEnabled:
                if candidate.proofChain and not self.proofLedger.verifyJobChain(candidate):
                    candidate.proofStatus = "invalid"
                    allow = False
                else:
                    attestation = self.attestor.attest(allocatedGpus[0])
                    if not isAttestationCompliant(attestation, candidate.trustPolicy):
                        candidate.proofStatus = "attestation_denied"
                        allow = False

            if not allow:
                continue

            job = self.queueManager.assignJobToGpus(
                candidate.id,
                allocatedGpus,
            )
            if job is None:
                continue

            resumedThisCycle = False
            if job.status == JobStatus.PAUSED and job.pid is not None:
                if runner.resumeJob(job.pid):
                    JobStateMachine.resume(job)
                    job.startedAt = time.time()
                    job.assignedGpus = list(allocatedGpus)
                    job.assignedGpu = allocatedGpus[0] if allocatedGpus else None
                    job.meta["resumeMode"] = "sigcont"
                    resumedThisCycle = True
                else:
                    # Paused process no longer exists; fallback to restart path.
                    self.queueManager.releaseJob(job)
                    job.pid = None
                    job.meta["resumeMode"] = "restart"
                    self.queueManager.requeueJob(
                        job,
                        refreshTimestamp=False,
                        targetStatus=JobStatus.QUEUED,
                        preservePid=False,
                    )
                    continue
            else:
                pid = runner.startJob(
                    job,
                    gpuIndices=allocatedGpus,
                )
                JobStateMachine.start(job)
                job.pid = pid
                job.assignedGpus = list(allocatedGpus)
                job.assignedGpu = allocatedGpus[0] if allocatedGpus else None
                if not isinstance(job.meta, dict):
                    job.meta = {}
                job.meta.setdefault("runTimeConsumedSeconds", 0.0)
                job.meta["resumeMode"] = "start"

            job.meta.pop("pausedAssignedGpus", None)

            if acprEnabled:
                job.lastAttestation = attestation
                eventType = (
                    "resume"
                    if resumedThisCycle or job.meta.get("resumeFromCheckpoint")
                    else "start"
                )

                checkpointHash = None
                resumePath = job.meta.get("resumeFromCheckpoint")
                if isinstance(resumePath, str):
                    checkpointHash = hashFile(resumePath)

                self._appendProofEvent(
                    job,
                    eventType=eventType,
                    gpuIndex=allocatedGpus[0],
                    attestation=attestation,
                    checkpointHash=checkpointHash,
                )

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"{'Resumed' if resumedThisCycle else 'Started'} job {job.id} "
                f"on GPU {allocatedGpus}"
            )

            return True

        return False

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------

    def _getGpuUtil(self, snapshot, gpuIndex: int) -> float:
        if snapshot.get("backend") == "nvidia-smi":
            for g in snapshot.get("gpus", []):
                if g.get("index") == gpuIndex:
                    return g.get("gpuUtilPercent", 0.0)

        if snapshot.get("backend") == "powermetrics":
            return snapshot.get("gpuUtilPercent", 0.0)

        return 0.0

    def _getGpuMemUtil(self, snapshot, gpuIndex: int) -> Optional[float]:
        if snapshot.get("backend") == "nvidia-smi":
            for g in snapshot.get("gpus", []):
                if g.get("index") == gpuIndex:
                    memUtil = g.get("gpuMemUtilPercent")
                    if memUtil is None:
                        return None
                    return float(memUtil)
        return None

    def _getGpuFreeMemMb(self, snapshot, gpuIndex: int) -> Optional[float]:
        if snapshot.get("backend") == "nvidia-smi":
            for g in snapshot.get("gpus", []):
                if g.get("index") != gpuIndex:
                    continue

                memUsedMb = g.get("gpuMemUsedMb")
                memTotalMb = g.get("gpuMemTotalMb")
                if memUsedMb is None or memTotalMb is None:
                    return None

                try:
                    freeMemMb = float(memTotalMb) - float(memUsedMb)
                except (TypeError, ValueError):
                    return None

                return max(0.0, freeMemMb)

        return None

    def _placementScore(
        self,
        job: Job,
        gpuSet: List[int],
        snapshot,
    ) -> Tuple[float, float, float, Tuple[int, ...]]:
        utilCost = 0.0
        for gpuIndex in gpuSet:
            utilCost += self._getGpuUtil(snapshot, gpuIndex) if snapshot else 0.0
        utilCost = utilCost / max(1, len(gpuSet))

        leftovers: List[float] = []
        if snapshot and snapshot.get("backend") == "nvidia-smi":
            required = float(job.requiredMemMb or 0.0)
            for gpuIndex in gpuSet:
                freeMem = self._getGpuFreeMemMb(snapshot, gpuIndex)
                if freeMem is None:
                    continue
                leftovers.append(max(0.0, float(freeMem) - required))

        if leftovers:
            totalLeftover = sum(leftovers)
            spread = max(leftovers) - min(leftovers)
        else:
            totalLeftover = 0.0
            spread = 0.0

        keyTail = tuple(sorted(gpuSet))
        if self.placementMode == "best_fit":
            return (totalLeftover, utilCost, spread, keyTail)
        if self.placementMode == "lowest_util":
            return (utilCost, totalLeftover, spread, keyTail)
        # fragmentation-aware default.
        return (spread, totalLeftover, utilCost, keyTail)

    def _getPausedPreferredGpus(self, job: Job) -> List[int]:
        raw = job.meta.get("pausedAssignedGpus")
        if not isinstance(raw, list):
            return []
        result: List[int] = []
        for g in raw:
            try:
                gi = int(g)
            except Exception:
                continue
            if gi not in result:
                result.append(gi)
        return result

    def _canPlaceJobOnGpus(
        self,
        job: Job,
        allocatedGpus: List[int],
        snapshot,
        enforcePolicy: bool,
    ) -> bool:
        if job.requiredMemMb is not None and snapshot is None:
            if not (job.status == JobStatus.PAUSED and job.pid is not None):
                return False

        if snapshot:
            for gpuIndex in allocatedGpus:
                util = self._getGpuUtil(snapshot, gpuIndex)
                memUtil = self._getGpuMemUtil(snapshot, gpuIndex)

                if enforcePolicy and not self.policy.canScheduleOnGpu(
                    gpuIndex,
                    util,
                    currentMemUtil=memUtil,
                ):
                    return False

                if job.requiredMemMb is not None:
                    freeMemMb = self._getGpuFreeMemMb(snapshot, gpuIndex)
                    if freeMemMb is None or freeMemMb < job.requiredMemMb:
                        return False

        return True

    def _findPlacementForJob(
        self,
        job: Job,
        snapshot,
        enforcePolicy: bool,
    ) -> Optional[List[int]]:
        freeGpus = self.queueManager.getFreeGpus(self.gpuIndices)
        if job.requiredGpus > len(freeGpus):
            return None

        # True pause/resume: resume paused jobs on the same GPU set.
        if job.status == JobStatus.PAUSED and job.pid is not None:
            preferred = self._getPausedPreferredGpus(job)
            if preferred and len(preferred) == int(job.requiredGpus):
                if all(g in freeGpus for g in preferred):
                    if self._canPlaceJobOnGpus(
                        job,
                        preferred,
                        snapshot,
                        enforcePolicy=enforcePolicy,
                    ):
                        return list(preferred)
            return None

        orderedGpus = sorted(freeGpus)
        gpuSets = [
            list(combo)
            for combo in combinations(
                orderedGpus,
                int(job.requiredGpus),
            )
        ]
        if not gpuSets:
            return None

        feasible: List[Tuple[Tuple[float, float, float, Tuple[int, ...]], List[int]]] = []
        for gpuSet in gpuSets:
            if self._canPlaceJobOnGpus(
                job,
                gpuSet,
                snapshot,
                enforcePolicy=enforcePolicy,
            ):
                feasible.append(
                    (
                        self._placementScore(job, gpuSet, snapshot),
                        gpuSet,
                    )
                )

        if not feasible:
            return None
        feasible.sort(key=lambda item: item[0])
        return feasible[0][1]

    def _estimateReclaimMemMb(
        self,
        runningJob: Job,
        fallbackMemMb: float,
    ) -> float:
        if runningJob.requiredMemMb is not None:
            return float(runningJob.requiredMemMb)
        return float(fallbackMemMb)

    def _getJobAssignedGpus(self, job: Job) -> List[int]:
        if job.assignedGpus:
            return list(job.assignedGpus)
        if job.assignedGpu is not None:
            return [job.assignedGpu]
        return []

    def _isGpuFreeByQueueRule(self, runningJobs: List[Job]) -> bool:
        if not runningJobs:
            return True
        return all(not job.exclusive for job in runningJobs)

    def _gpuStateAfterVictims(
        self,
        gpuIndex: int,
        candidate: Job,
        runningJobs: List[Job],
        baseFreeMemMb: float,
        victimsById: Dict[str, Job],
    ) -> Tuple[bool, bool]:
        runningAfter = [
            job
            for job in runningJobs
            if job.id not in victimsById
        ]
        gpuFree = self._isGpuFreeByQueueRule(runningAfter)

        requiredMemMb = float(candidate.requiredMemMb or 0.0)
        reclaimedMemMb = 0.0
        for victim in victimsById.values():
            if gpuIndex not in self._getJobAssignedGpus(victim):
                continue
            reclaimedMemMb += self._estimateReclaimMemMb(
                victim,
                fallbackMemMb=requiredMemMb,
            )

        memOk = (baseFreeMemMb + reclaimedMemMb) >= requiredMemMb
        return gpuFree, memOk

    def _selectMemoryPreemptionVictims(
        self,
        candidate: Job,
        snapshot,
        runningCountsByUser: Optional[Dict[str, int]] = None,
    ) -> Optional[Tuple[List[int], List[Job]]]:
        if candidate.requiredMemMb is None:
            return None

        requiredGpuCount = int(candidate.requiredGpus)
        if requiredGpuCount <= 0:
            return None
        if requiredGpuCount > len(self.gpuIndices):
            return None

        requiredMemMb = float(candidate.requiredMemMb)
        if requiredMemMb <= 0:
            return None

        candidateGpuCombos = [
            list(combo)
            for combo in combinations(self.gpuIndices, requiredGpuCount)
        ]
        if not candidateGpuCombos:
            return None

        bestChoice: Optional[Tuple[List[int], List[Job]]] = None
        bestScore: Optional[Tuple[int, int, float]] = None

        for targetGpus in candidateGpuCombos:
            baseFreeByGpu: Dict[int, float] = {}
            runningByGpu: Dict[int, List[Job]] = {}
            preemptibleByGpu: Dict[int, List[Job]] = {}

            comboFeasible = True
            for gpuIndex in targetGpus:
                freeMemMb = self._getGpuFreeMemMb(snapshot, gpuIndex)
                if freeMemMb is None:
                    comboFeasible = False
                    break
                baseFreeByGpu[gpuIndex] = float(freeMemMb)

                runningJobs = self.queueManager.getRunningJobsOnGpu(gpuIndex)
                runningByGpu[gpuIndex] = runningJobs

                preemptibleByGpu[gpuIndex] = [
                    job
                    for job in runningJobs
                    if job.preemptible and candidate.priority < job.priority
                ]
            if not comboFeasible:
                continue

            victimsById: Dict[str, Job] = {}
            maxIterations = (
                sum(len(jobs) for jobs in preemptibleByGpu.values()) + 1
            )

            for _ in range(maxIterations):
                failingGpu: Optional[int] = None
                for gpuIndex in targetGpus:
                    gpuFree, memOk = self._gpuStateAfterVictims(
                        gpuIndex=gpuIndex,
                        candidate=candidate,
                        runningJobs=runningByGpu[gpuIndex],
                        baseFreeMemMb=baseFreeByGpu[gpuIndex],
                        victimsById=victimsById,
                    )
                    if not (gpuFree and memOk):
                        failingGpu = gpuIndex
                        break

                if failingGpu is None:
                    break

                options = [
                    job
                    for job in preemptibleByGpu[failingGpu]
                    if job.id not in victimsById
                ]
                if not options:
                    comboFeasible = False
                    break

                options.sort(
                    key=lambda job: (
                        self._victimScore(
                            candidate,
                            job,
                            snapshot,
                            runningCountsByUser=runningCountsByUser,
                        ),
                        job.createdAt,
                        job.id,
                    )
                )
                chosen = options[0]
                victimsById[chosen.id] = chosen

            if not comboFeasible:
                continue

            allSatisfied = True
            totalReclaimedMem = 0.0
            for gpuIndex in targetGpus:
                gpuFree, memOk = self._gpuStateAfterVictims(
                    gpuIndex=gpuIndex,
                    candidate=candidate,
                    runningJobs=runningByGpu[gpuIndex],
                    baseFreeMemMb=baseFreeByGpu[gpuIndex],
                    victimsById=victimsById,
                )
                if not (gpuFree and memOk):
                    allSatisfied = False
                    break

                for victim in victimsById.values():
                    if gpuIndex in self._getJobAssignedGpus(victim):
                        totalReclaimedMem += self._estimateReclaimMemMb(
                            victim,
                            fallbackMemMb=requiredMemMb,
                        )

            if not allSatisfied:
                continue

            victims = list(victimsById.values())
            if not victims:
                continue

            score = (
                sum(
                    self._victimScore(
                        candidate,
                        victim,
                        snapshot,
                        runningCountsByUser=runningCountsByUser,
                    )
                    for victim in victims
                ),
                len(victims),
                -totalReclaimedMem,
            )
            if bestScore is None or score < bestScore:
                bestScore = score
                bestChoice = (targetGpus, victims)

        return bestChoice

    def _preemptJobs(
        self,
        victims: List[Job],
        candidateJobId: str,
        reason: str,
        targetGpus: Optional[List[int]] = None,
    ) -> None:
        seenJobIds = set()
        for runningJob in victims:
            if runningJob.id in seenJobIds:
                continue
            seenJobIds.add(runningJob.id)

            checkpointHash = self._requestCheckpoint(runningJob)
            assignedBeforePause = self._getJobAssignedGpus(runningJob)
            self._bumpRuntimeConsumed(runningJob)

            pausedSuccessfully = False
            if runningJob.pid is not None:
                pausedSuccessfully = runner.pauseJob(runningJob.pid)
                if not pausedSuccessfully:
                    runner.terminateJob(runningJob.pid)

            JobStateMachine.pause(runningJob)
            self.queueManager.releaseJob(runningJob)

            runningJob.meta["preemptionCount"] = (
                int(runningJob.meta.get("preemptionCount", 0)) + 1
            )
            runningJob.meta["resumeMode"] = (
                "sigcont" if pausedSuccessfully else "restart"
            )
            runningJob.meta["pausedAssignedGpus"] = list(assignedBeforePause)

            if pausedSuccessfully:
                self.queueManager.requeueJob(
                    runningJob,
                    refreshTimestamp=False,
                    targetStatus=JobStatus.PAUSED,
                    preservePid=True,
                )
            else:
                runningJob.meta.pop("pausedAssignedGpus", None)
                self.queueManager.requeueJob(
                    runningJob,
                    refreshTimestamp=True,
                    targetStatus=JobStatus.QUEUED,
                    preservePid=False,
                )

            self._appendProofEvent(
                runningJob,
                eventType="preempt",
                gpuIndex=(assignedBeforePause[0] if assignedBeforePause else None),
                attestation=runningJob.lastAttestation,
                checkpointHash=checkpointHash,
                extra={
                    "preemptReason": reason,
                    "preemptedForJobId": candidateJobId,
                    "targetGpus": targetGpus or [],
                    "resumeFrom": runningJob.meta.get("resumeFromCheckpoint"),
                    "preemptMode": runningJob.meta.get("resumeMode"),
                },
            )

    def _isAcpEnabled(self, job: Job) -> bool:
        if job.meta.get("acprEnabled") is True:
            return True
        if job.trustPolicy:
            return True
        if job.checkpointPath:
            return True
        if job.proofChain:
            return True
        return False

    def _appendProofEvent(
        self,
        job: Job,
        eventType: str,
        gpuIndex: Optional[int],
        attestation: Optional[Dict[str, Any]] = None,
        checkpointHash: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._isAcpEnabled(job):
            return
        try:
            self.proofLedger.appendEvent(
                job=job,
                eventType=eventType,
                gpuIndex=gpuIndex,
                attestation=attestation,
                checkpointHash=checkpointHash,
                extra=extra,
            )
        except Exception as e:
            print(f"Proof append failed for job {job.id}: {e}")

    def _requestCheckpoint(self, job: Job) -> Optional[str]:
        if job.pid is None:
            return None

        runner.sendPreemptSignal(job.pid)

        checkpointPath = job.checkpointPath
        if not checkpointPath:
            return None

        path = Path(checkpointPath)
        previousMtime = path.stat().st_mtime if path.exists() else None

        timeoutSeconds = float(
            job.meta.get("checkpointTimeoutSeconds", 15.0)
        )
        startWait = time.time()
        deadline = startWait + timeoutSeconds

        while time.time() < deadline:
            if path.exists() and path.is_file():
                mtime = path.stat().st_mtime
                if previousMtime is None or mtime > previousMtime:
                    checkpointHash = hashFile(str(path))
                    if checkpointHash:
                        job.meta["resumeFromCheckpoint"] = str(path)
                        return checkpointHash

            # If a previous checkpoint already exists, avoid stalling preemption.
            if previousMtime is not None and (time.time() - startWait) >= 1.0:
                break
            time.sleep(0.25)

        if path.exists() and path.is_file():
            checkpointHash = hashFile(str(path))
            if checkpointHash:
                job.meta["resumeFromCheckpoint"] = str(path)
                return checkpointHash

        return None

    def _cloneJob(self, job: Job) -> Job:
        return Job.fromDict(job.toDict())

    def _recordTerminalJob(self, job: Job) -> None:
        with self._terminalLock:
            self._terminalJobsById[job.id] = self._cloneJob(job)
            if job.id not in self._terminalOrder:
                self._terminalOrder.append(job.id)

    def _getPrimaryAssignedGpu(self, job: Job) -> Optional[int]:
        if job.assignedGpu is not None:
            return job.assignedGpu
        if job.assignedGpus:
            return job.assignedGpus[0]
        return None

    # ----------------------------------------------------
    # Job cancellations
    # ----------------------------------------------------

    def cancelJob(self, jobId: str) -> bool:
        """
        Cancel a job by ID.
        Returns True if job was found and cancelled.
        """

        with self._condition:

            # 1️⃣ Check queued jobs
            for job in self.queueManager.getQueuedJobs():
                if job.id == jobId:
                    if job.pid is not None:
                        runner.terminateJob(job.pid)
                    JobStateMachine.cancel(job)
                    self.queueManager.removeJob(jobId)
                    self._appendProofEvent(
                        job,
                        eventType="cancel",
                        gpuIndex=self._getPrimaryAssignedGpu(job),
                        attestation=job.lastAttestation,
                        extra={"state": "queued"},
                    )
                    self._recordTerminalJob(job)
                    print(f"Cancelled queued job {jobId}")
                    self._condition.notify()
                    return True

            # 2️⃣ Check running jobs
            for job in self.queueManager.getRunningJobs():
                if job.id == jobId:
                    gpuIndex = self._getPrimaryAssignedGpu(job)
                    if job.pid:
                        runner.terminateJob(job.pid)

                    JobStateMachine.cancel(job)
                    self.queueManager.releaseJob(job)
                    self._appendProofEvent(
                        job,
                        eventType="cancel",
                        gpuIndex=gpuIndex,
                        attestation=job.lastAttestation,
                        extra={"state": "running"},
                    )
                    self._recordTerminalJob(job)

                    print(f"Cancelled running job {jobId}")
                    self._condition.notify()
                    return True

        return False
