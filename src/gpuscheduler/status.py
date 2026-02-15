"""
status.py

Reads scheduler state snapshot and prints job information.
"""

import json
from pathlib import Path


STATE_FILE = Path("state/snapshot.json")


def main():
    if not STATE_FILE.exists():
        print("No scheduler state found.")
        return

    with open(STATE_FILE, "r") as f:
        data = json.load(f)

    print("\n=== RUNNING JOBS ===")
    running = data.get("running", [])
    if not running:
        print("None")
    else:
        for job in running:
            print(
                f"{job['id']} | GPU {job.get('assignedGpu')} | "
                f"Priority {job['priority']} | PID {job.get('pid')}"
            )

    print("\n=== QUEUED JOBS ===")
    queued = data.get("queued", [])
    if not queued:
        print("None")
    else:
        for job in queued:
            print(
                f"{job['id']} | Priority {job['priority']}"
            )


if __name__ == "__main__":
    main()