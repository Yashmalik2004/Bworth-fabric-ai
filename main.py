"""BWorth Fabric Intelligence — command-line entry point.

    python main.py validate             # audit the dataset
    python main.py train [train args]   # train ResNet-50 (+ final test evaluation)
    python main.py evaluate [args]      # re-evaluate a checkpoint on the test split
    python main.py predict IMAGE        # predict one image
    python main.py app                  # launch the Streamlit demo
    python main.py test                 # run the test suite
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "validate": "training.composition.validate_dataset",
    "train": "training.composition.train",
    "evaluate": "training.composition.evaluate",
    "predict": "inference.composition.predict",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in COMMANDS:
        return subprocess.call([sys.executable, "-m", COMMANDS[cmd], *rest], cwd=PROJECT_ROOT)
    if cmd == "app":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", *rest],
                               cwd=PROJECT_ROOT)
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests", *rest], cwd=PROJECT_ROOT)
    print(f"Unknown command: {cmd}\n{__doc__}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
