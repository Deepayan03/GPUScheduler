"""
status.py

Backward-compatible wrapper for `gpusched status`.
"""

import argparse

from gpuscheduler.cli import main as cliMain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--include-terminal", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    argv = ["status"]
    if args.all:
        argv.append("--all")
    if args.db_path:
        argv.extend(["--db-path", args.db_path])
    if args.include_terminal:
        argv.append("--include-terminal")
    if args.json:
        argv.append("--json")

    return cliMain(argv)


if __name__ == "__main__":
    raise SystemExit(main())
