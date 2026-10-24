"""Train 3-image MNIST membership: python -m examples.members.member3."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.members.member import main

if __name__ == "__main__":
    main(3)
