import time
from gpuscheduler.daemon.job import Job
from gpuscheduler.scheduler.queueManager import QueueManager


def main():
    qm = QueueManager(agingFactor=0.05)

    # Simulate 2 GPUs
    gpus = [0, 1]

    print("Creating jobs...")

    job1 = Job(command="job1", priority=5, requiredGpus=1)
    job2 = Job(command="job2", priority=10, requiredGpus=1)
    job3 = Job(command="job3", priority=1, requiredGpus=2)  # needs both GPUs

    qm.addJob(job1)
    qm.addJob(job2)
    qm.addJob(job3)

    print("Queue size:", qm.getQueueSize())

    print("\nAttempt allocation 1:")
    result1 = qm.findAndAssignJob(gpus)
    print(result1)

    print("\nRunning state:", qm.getRunningJobs())

    print("\nAttempt allocation 2:")
    result2 = qm.findAndAssignJob(gpus)
    print(result2)

    print("\nRunning state:", qm.getRunningJobs())

    print("\nReleasing first job...")
    if result1:
        qm.releaseJob(result1[0])
    print("Running state after release:", qm.getRunningJobs())

    print("\nWaiting for aging to kick in...")
    time.sleep(3)

    print("\nAttempt allocation after aging:")
    result = qm.findAndAssignJob(gpus)
    print(result)


if __name__ == "__main__":
    main()