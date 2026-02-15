"""
cancel.py

Submit a cancel request to the running scheduler daemon.
Writes a cancel control file into control directory.
"""

import argparse
import json
from pathlib import Path


CONTROL_DIR = Path("control")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobId", required=True)
    args = parser.parse_args()

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    filePath = CONTROL_DIR / f"cancel_{args.jobId}.json"

    with open(filePath, "w") as f:
        json.dump({"jobId": args.jobId}, f)

    print(f"Cancel request submitted for {args.jobId}")


if __name__ == "__main__":
    main()