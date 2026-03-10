"""
Tests for newly implemented scheduler features:
- Smart victim scoring
- Aging-based priority
- Fair-share user caps
- Advanced placement strategy
- True pause/resume preemption
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gpuscheduler.daemon.job import Job, JobStatus
from gpuscheduler.scheduler.core import SchedulerCore
from gpuscheduler.scheduler.queueManager import QueueManager
from gpuscheduler.scheduler.stateMachine import JobStateMachine


def _register_running_job(
    core: SchedulerCore,
    job: Job,
    gpu_indices: list[int],
    pid: int,
    started_seconds_ago: float = 1.0,
) -> None:
    core.queueManager.addJob(job)
    assigned = core.queueManager.assignJobToGpus(job.id, gpu_indices)
    assert assigned is not None
    JobStateMachine.start(job)
    job.pid = pid
    job.assignedGpus = list(gpu_indices)
    job.assignedGpu = gpu_indices[0] if gpu_indices else None
    job.startedAt = time.time() - started_seconds_ago


class AgingPriorityTests(unittest.TestCase):
    def test_aging_promotes_old_jobs(self):
        qm = QueueManager(agingFactor=0.05)
        older = Job(command="older", priority=10)
        newer = Job(command="newer", priority=1)
        older.createdAt = time.time() - 300.0
        newer.createdAt = time.time()
        qm.addJob(older)
        qm.addJob(newer)

        ordered = qm.getQueuedJobs(agingFactor=0.05)
        self.assertEqual("older", ordered[0].command)
        self.assertEqual("newer", ordered[1].command)

    def test_fairness_penalty_affects_effective_priority(self):
        qm = QueueManager(agingFactor=0.0)
        a = Job(command="a", priority=5, meta={"user": "u1"})
        b = Job(command="b", priority=5, meta={"user": "u2"})
        a.createdAt = b.createdAt = time.time()
        qm.addJob(a)
        qm.addJob(b)

        ordered = qm.getQueuedJobs(
            fairnessPenaltyByUser={"u1": 2.0},
            includePaused=True,
        )
        self.assertEqual("b", ordered[0].command)
        self.assertEqual("a", ordered[1].command)

    def test_include_paused_flag_filters_queue(self):
        qm = QueueManager(agingFactor=0.0)
        queued = Job(command="queued", priority=5)
        paused = Job(command="paused", priority=1)
        paused.status = JobStatus.PAUSED
        qm.addJob(queued)
        qm.addJob(paused)

        only_queued = qm.getQueuedJobs(includePaused=False)
        self.assertEqual(["queued"], [j.command for j in only_queued])


class FairShareTests(unittest.TestCase):
    def test_fair_share_cap_skips_user_over_limit(self):
        core = SchedulerCore(
            gpuIndices=[0, 1],
            maxConcurrentPerUser=1,
            placementMode="best_fit",
        )

        # Existing running job for user alice on GPU 0.
        running = Job(command="run", priority=5, meta={"user": "alice"})
        _register_running_job(core, running, [0], pid=1001)

        queued_alice = Job(command="alice-next", priority=1, meta={"user": "alice"})
        queued_bob = Job(command="bob-next", priority=2, meta={"user": "bob"})
        core.submitJob(queued_alice)
        core.submitJob(queued_bob)

        with patch("gpuscheduler.scheduler.core.runner.startJob", return_value=2002):
            with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
                scheduled = core._handleScheduling()

        self.assertTrue(scheduled)
        running_commands = {job.command for job in core.queueManager.getRunningJobs()}
        self.assertIn("run", running_commands)
        self.assertIn("bob-next", running_commands)
        self.assertNotIn("alice-next", running_commands)

    def test_preemption_skips_candidate_blocked_by_fair_share(self):
        core = SchedulerCore(gpuIndices=[0], maxConcurrentPerUser=1)

        # Alice already occupies the only running slot for user alice.
        alice_running = Job(command="alice-running", priority=9, meta={"user": "alice"})
        _register_running_job(core, alice_running, [0], pid=4001, started_seconds_ago=20)

        # Victim currently running on the single GPU.
        victim = Job(command="victim", priority=8, meta={"user": "charlie"})
        core.queueManager.addJob(victim)
        assigned = core.queueManager.assignJobToGpus(victim.id, [0])
        self.assertIsNotNone(assigned)
        JobStateMachine.start(victim)
        victim.pid = 4002
        victim.assignedGpu = 0
        victim.assignedGpus = [0]
        victim.startedAt = time.time() - 10

        blocked = Job(command="alice-top", priority=1, meta={"user": "alice"})
        allowed = Job(command="bob-next", priority=2, meta={"user": "bob"})
        core.submitJob(blocked)
        core.submitJob(allowed)

        capture = {}

        def _capture(victims, candidateJobId, reason, targetGpus=None):
            capture["candidate"] = candidateJobId

        with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
            with patch.object(core, "_preemptJobs", side_effect=_capture):
                preempted = core._handlePreemption()

        self.assertTrue(preempted)
        self.assertEqual(allowed.id, capture["candidate"])


class PlacementStrategyTests(unittest.TestCase):
    def test_fragmentation_aware_prefers_balanced_leftovers(self):
        core = SchedulerCore(gpuIndices=[0, 1, 2], placementMode="fragmentation_aware")
        job = Job(command="fit", requiredGpus=2, requiredMemMb=500)
        snapshot = {
            "backend": "nvidia-smi",
            "gpus": [
                {"index": 0, "gpuUtilPercent": 10, "gpuMemUsedMb": 500, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 25},
                {"index": 1, "gpuUtilPercent": 10, "gpuMemUsedMb": 400, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 20},
                {"index": 2, "gpuUtilPercent": 10, "gpuMemUsedMb": 1000, "gpuMemTotalMb": 4000, "gpuMemUtilPercent": 25},
            ],
        }
        chosen = core._findPlacementForJob(job, snapshot, enforcePolicy=False)
        self.assertEqual([0, 1], chosen)

    def test_lowest_util_prefers_lower_utilization_combo(self):
        core = SchedulerCore(gpuIndices=[0, 1, 2], placementMode="lowest_util")
        job = Job(command="util", requiredGpus=2, requiredMemMb=None)
        snapshot = {
            "backend": "nvidia-smi",
            "gpus": [
                {"index": 0, "gpuUtilPercent": 5, "gpuMemUsedMb": 200, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 10},
                {"index": 1, "gpuUtilPercent": 80, "gpuMemUsedMb": 200, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 10},
                {"index": 2, "gpuUtilPercent": 1, "gpuMemUsedMb": 200, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 10},
            ],
        }
        chosen = core._findPlacementForJob(job, snapshot, enforcePolicy=False)
        self.assertEqual([0, 2], chosen)

    def test_best_fit_prefers_tight_memory_fit(self):
        core = SchedulerCore(gpuIndices=[0, 1, 2], placementMode="best_fit")
        job = Job(command="best", requiredGpus=2, requiredMemMb=900)
        snapshot = {
            "backend": "nvidia-smi",
            "gpus": [
                {"index": 0, "gpuUtilPercent": 20, "gpuMemUsedMb": 200, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 10},  # free 1800 -> left 900
                {"index": 1, "gpuUtilPercent": 20, "gpuMemUsedMb": 300, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 15},  # free 1700 -> left 800
                {"index": 2, "gpuUtilPercent": 20, "gpuMemUsedMb": 1050, "gpuMemTotalMb": 2000, "gpuMemUtilPercent": 52}, # free 950 -> left 50
            ],
        }
        chosen = core._findPlacementForJob(job, snapshot, enforcePolicy=False)
        # Tightest leftover total is [1,2] => 800 + 50.
        self.assertEqual([1, 2], chosen)

    def test_paused_job_requires_same_gpu_set(self):
        core = SchedulerCore(gpuIndices=[0, 1], placementMode="best_fit")
        paused = Job(command="paused", priority=2, requiredGpus=1, meta={"user": "u"})
        paused.status = JobStatus.PAUSED
        paused.pid = 7777
        paused.meta["pausedAssignedGpus"] = [1]
        core.queueManager.addJob(paused)

        # Occupy GPU 1 so resume cannot use preferred GPU.
        busy = Job(command="busy", priority=9)
        _register_running_job(core, busy, [1], pid=7001)

        no_place = core._findPlacementForJob(paused, {"backend": "none"}, enforcePolicy=False)
        self.assertIsNone(no_place)

        core.queueManager.releaseJob(busy)
        place = core._findPlacementForJob(paused, {"backend": "none"}, enforcePolicy=False)
        self.assertEqual([1], place)


class VictimScoringTests(unittest.TestCase):
    def test_priority_preemption_uses_scored_victim(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")

        # Incoming high-priority job.
        incoming = Job(command="incoming", priority=1, meta={"user": "team-a"})
        core.submitJob(incoming)

        # Two possible victims with equal priority but different runtime/preemption history.
        victim_a = Job(command="victim-a", priority=9, meta={"user": "team-b", "preemptionCount": 4, "runTimeConsumedSeconds": 900})
        victim_b = Job(command="victim-b", priority=9, meta={"user": "team-b", "preemptionCount": 0, "runTimeConsumedSeconds": 10})
        _register_running_job(core, victim_a, [0], pid=3001, started_seconds_ago=100)

        # Make victim_b a shared occupant on same GPU for candidate enumeration.
        core.queueManager.addJob(victim_b)
        assigned = core.queueManager.assignJobToGpus(victim_b.id, [0])
        self.assertIsNotNone(assigned)
        JobStateMachine.start(victim_b)
        victim_b.pid = 3002
        victim_b.assignedGpu = 0
        victim_b.assignedGpus = [0]
        victim_b.startedAt = time.time() - 5

        chosen = {}

        def _capture_preempt(victims, candidateJobId, reason, targetGpus=None):
            chosen["victims"] = [j.id for j in victims]

        with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
            with patch.object(core, "_preemptJobs", side_effect=_capture_preempt):
                preempted = core._handlePreemption()

        self.assertTrue(preempted)
        self.assertEqual([victim_b.id], chosen["victims"])

    def test_memory_victim_scoring_prefers_higher_mem_reclaim(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")
        candidate = Job(command="incoming", priority=1, requiredMemMb=2000)

        small = Job(command="small", priority=9, requiredMemMb=1500, meta={"user": "u1"})
        large = Job(command="large", priority=9, requiredMemMb=6000, meta={"user": "u1"})
        _register_running_job(core, small, [0], pid=8001)
        core.queueManager.addJob(large)
        assigned = core.queueManager.assignJobToGpus(large.id, [0])
        self.assertIsNotNone(assigned)
        JobStateMachine.start(large)
        large.pid = 8002
        large.assignedGpu = 0
        large.assignedGpus = [0]
        large.startedAt = time.time() - 5
        # Make both shareable for this synthetic setup.
        small.exclusive = False
        large.exclusive = False

        snapshot = {
            "backend": "nvidia-smi",
            "gpus": [
                {
                    "index": 0,
                    "gpuUtilPercent": 20.0,
                    "gpuMemUsedMb": 9000.0,
                    "gpuMemTotalMb": 10000.0,
                    "gpuMemUtilPercent": 90.0,
                }
            ],
        }

        picked = core._selectMemoryPreemptionVictims(candidate, snapshot)
        self.assertIsNotNone(picked)
        _, victims = picked
        self.assertEqual([large.id], [v.id for v in victims])


class PauseResumeTests(unittest.TestCase):
    def test_preempt_uses_sigstop_and_requeue_paused(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")
        running = Job(command="long", priority=9, meta={"user": "alice"})
        _register_running_job(core, running, [0], pid=4444, started_seconds_ago=20)

        with patch("gpuscheduler.scheduler.core.runner.pauseJob", return_value=True) as pause_mock:
            with patch("gpuscheduler.scheduler.core.runner.terminateJob") as kill_mock:
                with patch.object(core, "_requestCheckpoint", return_value=None):
                    core._preemptJobs(
                        victims=[running],
                        candidateJobId="incoming",
                        reason="priority",
                        targetGpus=[0],
                    )

        pause_mock.assert_called_once_with(4444)
        kill_mock.assert_not_called()
        self.assertEqual(JobStatus.PAUSED, running.status)
        self.assertEqual(4444, running.pid)
        self.assertEqual("sigcont", running.meta["resumeMode"])
        queued = core.queueManager.getQueuedJobs(includePaused=True)
        self.assertEqual(1, len(queued))
        self.assertEqual(JobStatus.PAUSED, queued[0].status)
        self.assertEqual(0, len(core.queueManager.getRunningJobs()))
        self.assertGreater(float(running.meta.get("runTimeConsumedSeconds", 0.0)), 0.0)

    def test_scheduling_resumes_paused_process(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")
        paused = Job(command="paused", priority=5, meta={"user": "alice"})
        _register_running_job(core, paused, [0], pid=5555, started_seconds_ago=10)

        with patch("gpuscheduler.scheduler.core.runner.pauseJob", return_value=True):
            with patch.object(core, "_requestCheckpoint", return_value=None):
                core._preemptJobs(
                    victims=[paused],
                    candidateJobId="incoming",
                    reason="priority",
                    targetGpus=[0],
                )

        self.assertEqual(JobStatus.PAUSED, paused.status)

        with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
            with patch("gpuscheduler.scheduler.core.runner.resumeJob", return_value=True) as resume_mock:
                scheduled = core._handleScheduling()

        self.assertTrue(scheduled)
        resume_mock.assert_called_once_with(5555)
        self.assertEqual(JobStatus.RUNNING, paused.status)
        self.assertEqual(5555, paused.pid)
        running_ids = {j.id for j in core.queueManager.getRunningJobs()}
        self.assertIn(paused.id, running_ids)

    def test_pause_failure_falls_back_to_restart_queue(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")
        running = Job(command="restart-me", priority=9, meta={"user": "alice"})
        _register_running_job(core, running, [0], pid=6666, started_seconds_ago=3)

        with patch("gpuscheduler.scheduler.core.runner.pauseJob", return_value=False):
            with patch("gpuscheduler.scheduler.core.runner.terminateJob") as term_mock:
                with patch.object(core, "_requestCheckpoint", return_value=None):
                    core._preemptJobs(
                        victims=[running],
                        candidateJobId="incoming",
                        reason="priority",
                        targetGpus=[0],
                    )

        term_mock.assert_called_once_with(6666)
        self.assertEqual(JobStatus.QUEUED, running.status)
        self.assertIsNone(running.pid)
        self.assertEqual("restart", running.meta["resumeMode"])

    def test_resume_failure_requeues_then_restarts_next_cycle(self):
        core = SchedulerCore(gpuIndices=[0], placementMode="best_fit")
        paused = Job(command="paused", priority=2, meta={"user": "u"})
        _register_running_job(core, paused, [0], pid=7778, started_seconds_ago=5)

        with patch("gpuscheduler.scheduler.core.runner.pauseJob", return_value=True):
            with patch.object(core, "_requestCheckpoint", return_value=None):
                core._preemptJobs(
                    victims=[paused],
                    candidateJobId="incoming",
                    reason="priority",
                    targetGpus=[0],
                )
        self.assertEqual(JobStatus.PAUSED, paused.status)

        with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
            with patch("gpuscheduler.scheduler.core.runner.resumeJob", return_value=False):
                first = core._handleScheduling()
        self.assertFalse(first)
        self.assertEqual(JobStatus.QUEUED, paused.status)
        self.assertIsNone(paused.pid)

        with patch.object(core.monitor, "getLastStats", return_value={"backend": "none"}):
            with patch("gpuscheduler.scheduler.core.runner.startJob", return_value=9090):
                second = core._handleScheduling()
        self.assertTrue(second)
        self.assertEqual(JobStatus.RUNNING, paused.status)
        self.assertEqual(9090, paused.pid)


if __name__ == "__main__":
    unittest.main()
