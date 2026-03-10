"""
cancel.py

Backward-compatible wrapper for `gpusched cancel`.
"""

import argparse

from gpuscheduler.cli import main as cliMain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobId", dest="job_id", default=None)
    parser.add_argument("--job-id", dest="job_id_alt", default=None)
    args = parser.parse_args()

    jobId = args.job_id_alt or args.job_id
    if not jobId:
        parser.error("one of --jobId/--job-id is required")

    return cliMain(["cancel", "--job-id", jobId])


if __name__ == "__main__":
    raise SystemExit(main())
