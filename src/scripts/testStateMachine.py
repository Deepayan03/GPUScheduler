from gpuscheduler.daemon.job import Job
from gpuscheduler.scheduler.stateMachine import JobStateMachine
from gpuscheduler.daemon.job import JobStatus


def main():
    job = Job(command="test")

    print("Initial:", job.status)

    JobStateMachine.start(job)
    print("After start:", job.status)

    JobStateMachine.pause(job)
    print("After pause:", job.status)

    JobStateMachine.resume(job)
    print("After resume:", job.status)

    JobStateMachine.finish(job, success=True)
    print("After finish:", job.status)

    print("\nTrying illegal transition...")
    try:
        JobStateMachine.start(job)
    except Exception as e:
        print("Caught:", e)


if __name__ == "__main__":
    main()