from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aliengo_competition.common.helpers import get_args
from aliengo_competition.robot_interface.sim import SimAliengoRobot
from scripts.play import load_env

DEFAULT_CAMERA_DEPTH_MAX_M = 10.0

def _clone_args(args, *, task: str, mode: str, headless: bool, checkpoint=-1):
    clone = SimpleNamespace(**vars(args))
    clone.task = task
    clone.mode = mode
    clone.headless = headless
    clone.num_envs = 1
    clone.checkpoint = checkpoint
    clone.resume = True
    return clone


def _resolve_run_dir() -> Path:
    runs_root = Path(__file__).resolve().parents[3] / "runs" / "gait-conditioned-agility"

    candidates = [path for path in runs_root.glob("*/*/*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def make_robot_interface(
    *,
    args=None,
    task: str = "aliengo_flat",
    mode: str = "sim",
    headless: bool = True,
    checkpoint=-1,
):
    if args is None:
        args = get_args()
    _clone_args(args, task=task, mode=mode, headless=headless, checkpoint=checkpoint)
    if checkpoint not in (-1, None):
        print("Explicit checkpoints are ignored for controller demo; using the latest exported JIT policy from the selected run.")

    run_dir = _resolve_run_dir()
    print(f"Loading controller low-level policy from: {run_dir}")
    env, policy = load_env(run_dir, headless=headless)
    return SimAliengoRobot(env=env, policy=policy)
