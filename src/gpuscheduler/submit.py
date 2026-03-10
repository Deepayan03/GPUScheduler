"""
submit.py

Backward-compatible wrapper for `gpusched submit`.
"""

import argparse

from gpuscheduler.cli import main as cliMain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--mem", type=int, default=None)
    parser.add_argument("--max-runtime", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--trust-policy-file", type=str, default=None)
    parser.add_argument("--meta-json", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--acpr", action="store_true")
    args = parser.parse_args()

    argv = [
        "submit",
        "--cmd",
        args.cmd,
        "--priority",
        str(args.priority),
        "--gpus",
        str(args.gpus),
    ]
    if args.mem is not None:
        argv.extend(["--mem", str(args.mem)])
    if args.max_runtime is not None:
        argv.extend(["--max-runtime", str(args.max_runtime)])
    if args.checkpoint_path:
        argv.extend(["--checkpoint-path", args.checkpoint_path])
    if args.trust_policy_file:
        argv.extend(["--trust-policy-file", args.trust_policy_file])
    if args.meta_json:
        argv.extend(["--meta-json", args.meta_json])
    if args.user:
        argv.extend(["--user", args.user])
    if args.acpr:
        argv.append("--acpr")

    return cliMain(argv)


if __name__ == "__main__":
    raise SystemExit(main())
