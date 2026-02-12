import time
from gpuscheduler.daemon.job import Job
from gpuscheduler.daemon import runner


def main():
    print("Creating advanced job...")

    job = Job(
        command="python3 -c \"import time,signal; "
                "print('start'); "
                "time.sleep(10); "
                "print('end')\"",
        priority=5,
        requiredGpus=1,
        maxRuntimeSeconds=20,
    )

    print("Starting job on GPU 0...")
    pid = runner.startJob(job, gpuIndex=0, logDir="/tmp/gpusched_test")
    job.markStarted(pid, gpuIndex=0)

    print("PID:", pid)

    time.sleep(2)

    print("Pausing job...")
    runner.pauseJob(pid)

    time.sleep(2)

    print("Resuming job...")
    runner.resumeJob(pid)

    time.sleep(2)

    print("Sending cooperative preempt signal...")
    runner.sendPreemptSignal(pid)

    time.sleep(2)

    print("Checking if runtime exceeded:", runner.checkRuntimeExceeded(pid))

    print("Terminating job early...")
    exitCode = runner.terminateJob(pid)

    print("Exit code:", exitCode)

    print("\n=== LOG OUTPUT ===")
    log = runner.readJobLogTail(job.id, logDir="/tmp/gpusched_test")
    print(log.decode())


if __name__ == "__main__":
    main()