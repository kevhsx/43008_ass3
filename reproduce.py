"""One command for tests, main comparison, robustness sweep and result verification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/reproduction_all")
    parser.add_argument("--include-adaptation", action="store_true", help="Also run the 1000-episode changed-layout extension")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(out / ".mplcache"))
    commands = [
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "run_experiments.py", "--output", str(out / "main")],
        [sys.executable, "run_sweep.py", "--output", str(out / "sweep")],
        [sys.executable, "verify_results.py", "--results", str(out / "main"), "--sweep", str(out / "sweep")],
    ]
    if args.include_adaptation:
        commands.extend([
            [sys.executable, "run_adaptation.py", "--main-results", str(out / "main"), "--output", str(out / "adaptation")],
            [sys.executable, "verify_results.py", "--results", str(out / "adaptation"), "--adaptation", str(out / "adaptation"), "--base-results", str(out / "main")],
        ])
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
