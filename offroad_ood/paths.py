"""Central path resolution for the OffRoad-OOD benchmark.

Set the environment variable ``OFFROAD_OOD_ROOT`` to the repository root
(the directory holding ``configs/``, ``data/`` and ``results/``). If unset,
the current working directory is used, so running the scripts from the repo
root works out of the box.
"""
import os

ROOT = os.environ.get("OFFROAD_OOD_ROOT", os.getcwd())
