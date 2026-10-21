#!/bin/zsh
cd "${0:A:h}/../.." || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo "First, run: python3 -m venv .venv"
  echo "Then run: .venv/bin/python -m pip install -r requirements-lock.txt"
  read "reply?Press Enter to close"
  exit 1
fi
.venv/bin/python demo.py "$@"
demo_exit_code=$?
if (( demo_exit_code != 0 )); then
  read "reply?Launch failed. Press Enter to close"
fi
exit $demo_exit_code
