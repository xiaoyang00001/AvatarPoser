import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from avatarposer_realtime.runtime_env import configure_runtime_environment

configure_runtime_environment()

from avatarposer_realtime.app import main


if __name__ == "__main__":
    main()
