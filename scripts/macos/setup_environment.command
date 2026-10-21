#!/bin/zsh
cd "${0:A:h}/../.." || exit 1
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv || exit 1
fi
.venv/bin/python -m pip install -r requirements-lock.txt
setup_exit_code=$?
if (( setup_exit_code == 0 )); then
  echo "Setup complete. Select .venv/bin/python in VS Code or open scripts/macos/start_demo.command."
fi
read "reply?Press Enter to close"
exit $setup_exit_code
