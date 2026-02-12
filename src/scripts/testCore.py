import time
from gpuscheduler.scheduler.core import SchedulerCore
from gpuscheduler.daemon.job import Job


def main():
    core = SchedulerCore(pollInterval=2.0)

    job1 = Job(command="sleep 20")
    job2 = Job(command="sleep 20")

    core.submitJob(job1)
    core.submitJob(job2)

    core.run()


if __name__ == "__main__":
    main()