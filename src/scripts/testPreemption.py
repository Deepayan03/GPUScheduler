import time
import threading
from gpuscheduler.scheduler.core import SchedulerCore
from gpuscheduler.daemon.job import Job


def main():
    core = SchedulerCore(pollInterval=1.0)

    schedulerThread = threading.Thread(target=core.run)
    schedulerThread.start()

    # Wait until monitor has produced snapshot
    time.sleep(5)

    # Submit low priority job
    job1 = Job(command="sleep 20", priority=10, exclusive=True)
    core.submitJob(job1)

    # Wait until job1 is definitely running
    time.sleep(5)

    # Now submit high priority job
    print("\nSubmitting high priority job\n")
    job2 = Job(command="sleep 5", priority=1, exclusive=True)
    core.submitJob(job2)

    # Let system run
    time.sleep(30)

    core.stop()
    schedulerThread.join()


if __name__ == "__main__":
    main()