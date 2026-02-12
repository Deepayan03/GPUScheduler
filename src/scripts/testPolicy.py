import time
from gpuscheduler.scheduler.policy import SchedulerPolicy


def main():
    policy = SchedulerPolicy(
        staticUtilThreshold=60,
        historyWindow=3,
        spikeDelta=20,
        cooldownSeconds=3,
    )

    gpu = 0

    print("Testing normal utilization...")
    print(policy.canScheduleOnGpu(gpu, 30))
    print(policy.canScheduleOnGpu(gpu, 40))
    print(policy.canScheduleOnGpu(gpu, 35))

    print("\nTesting spike...")
    print(policy.canScheduleOnGpu(gpu, 20))
    print(policy.canScheduleOnGpu(gpu, 90))  # spike
    print(policy.canScheduleOnGpu(gpu, 25))  # should cooldown

    print("\nWaiting for cooldown...")
    time.sleep(4)

    print(policy.canScheduleOnGpu(gpu, 25))  # should allow again

    print("\nTesting static fallback...")
    print(policy.canScheduleOnGpu(gpu, 55))  # below threshold
    print(policy.canScheduleOnGpu(gpu, 75))  # above threshold

    print("\nTesting preemption decision...")
    print(policy.shouldPreempt(gpu, currentUtil=50, jobPriority=10, incomingPriority=1))
    print(policy.shouldPreempt(gpu, currentUtil=95, jobPriority=10, incomingPriority=1))


if __name__ == "__main__":
    main()