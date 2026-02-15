"""
submit.py

CLI job submission tool.
Writes job JSON into inbox folder.
"""

import argparse
import json
import uuid
from pathlib import Path
from gpuscheduler.daemon.job import Job

INBOX_DIR = Path("inbox")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--mem", type=int, default=None)

    args = parser.parse_args()

    job = Job(
        command=args.cmd,
        priority=args.priority,
        requiredGpus=args.gpus,
        requiredMemMb=args.mem,
    )

    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    filePath = INBOX_DIR / f"{job.id}.json"

    with open(filePath, "w") as f:
        json.dump(job.toDict(), f, indent=2)

    print(f"Job {job.id} submitted.")


if __name__ == "__main__":
    main()