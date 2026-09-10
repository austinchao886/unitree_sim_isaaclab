#!/usr/bin/env python3
"""Export the resolved PhysX/IsaacLab contract for one G1 task.

This is a read-only compatibility diagnostic.  It deliberately does not start
DDS or apply any action, so an unqualified USD can be inspected safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--profile-id", required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import tasks  # noqa: F401  Registers the environments.
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _tensor(value: Any) -> Any:
    """Convert one Isaac tensor-like value to finite JSON data."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 0 and value.shape[0] == 1:
            value = value[0]
        result = value.tolist()
        flat = value.reshape(-1)
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("asset snapshot contains a non-finite tensor")
        return result
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _first_attr(obj: Any, names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        if hasattr(obj, name):
            return name, getattr(obj, name)
    return None, None


def _resolved_asset_path(env_cfg: Any) -> str | None:
    spawn = env_cfg.scene.robot.spawn
    for name in ("asset_path", "usd_path"):
        value = getattr(spawn, name, None)
        if value:
            return str(Path(value).resolve())
    return None


def _properties(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if obj is None:
        return {name: None for name in names}
    return {name: getattr(obj, name, None) for name in names}


def main() -> int:
    env = None
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.sim.reset()
        env.reset()
        robot = env.scene["robot"]
        data = robot.data

        fields: dict[str, Any] = {}
        aliases = {
            "joint_position_limits": ("joint_pos_limits", "joint_limits"),
            "joint_effort_limits": (
                "joint_effort_limits",
                "default_joint_effort_limits",
            ),
            "joint_velocity_limits": (
                "joint_vel_limits",
                "joint_velocity_limits",
                "default_joint_velocity_limits",
            ),
            "joint_stiffness": ("joint_stiffness", "default_joint_stiffness"),
            "joint_damping": ("joint_damping", "default_joint_damping"),
            "joint_armature": ("joint_armature", "default_joint_armature"),
            "joint_friction": ("joint_friction", "default_joint_friction"),
            "body_mass": ("body_mass", "default_mass"),
            "body_inertia": ("body_inertia", "default_inertia"),
        }
        resolved_names: dict[str, str | None] = {}
        for output_name, candidates in aliases.items():
            resolved_name, value = _first_attr(data, candidates)
            resolved_names[output_name] = resolved_name
            fields[output_name] = _tensor(value)

        payload = {
            "schema_version": 1,
            "profile_id": args_cli.profile_id,
            "task": args_cli.task,
            "asset_path": _resolved_asset_path(env_cfg),
            "physics_dt_s": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "joint_names": list(data.joint_names),
            "body_names": list(data.body_names),
            "default_root_state": _tensor(data.default_root_state),
            "default_joint_pos": _tensor(data.default_joint_pos),
            "default_joint_vel": _tensor(data.default_joint_vel),
            "resolved_data_attributes": resolved_names,
            "simulation_contract": {
                "robot_rigid_properties": _properties(
                    getattr(env_cfg.scene.robot.spawn, "rigid_props", None),
                    (
                        "disable_gravity",
                        "linear_damping",
                        "angular_damping",
                        "max_linear_velocity",
                        "max_angular_velocity",
                        "max_depenetration_velocity",
                    ),
                ),
                "articulation_root_properties": _properties(
                    getattr(env_cfg.scene.robot.spawn, "articulation_props", None),
                    (
                        "enabled_self_collisions",
                        "solver_position_iteration_count",
                        "solver_velocity_iteration_count",
                    ),
                ),
                "ground_material": _properties(
                    getattr(env_cfg.scene.ground.spawn, "physics_material", None),
                    (
                        "static_friction",
                        "dynamic_friction",
                        "restitution",
                        "friction_combine_mode",
                        "restitution_combine_mode",
                    ),
                ),
                "simulation_material": _properties(
                    getattr(env_cfg.sim, "physics_material", None),
                    (
                        "static_friction",
                        "dynamic_friction",
                        "restitution",
                        "friction_combine_mode",
                        "restitution_combine_mode",
                    ),
                ),
            },
            **fields,
        }
        # Refuse to publish a misleading partial snapshot.
        required = (
            "joint_position_limits",
            "joint_effort_limits",
            "joint_velocity_limits",
            "joint_damping",
            "joint_armature",
            "body_mass",
            "body_inertia",
        )
        missing = [name for name in required if payload[name] is None]
        if missing:
            raise RuntimeError(f"Isaac did not expose required asset fields: {missing}")
        if len(payload["joint_names"]) < 29:
            raise RuntimeError("loaded articulation has fewer than 29 joints")
        if not math.isclose(payload["physics_dt_s"], 0.005, abs_tol=1.0e-9):
            raise RuntimeError(
                f"task physics dt is not the 200 Hz LowCmd contract: {payload['physics_dt_s']}"
            )

        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args_cli.output.with_suffix(args_cli.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args_cli.output)
        print(
            f"wrote {args_cli.profile_id}: joints={len(payload['joint_names'])}, "
            f"bodies={len(payload['body_names'])}, output={args_cli.output}"
        )
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
