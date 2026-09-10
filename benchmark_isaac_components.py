#!/usr/bin/env python3
"""Benchmark Isaac physics, 200 Hz safety, and DDS bridge components."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--component",
    choices=(
        "physics_only",
        "qualified_step",
        "physics_safety",
        "physics_safety_async",
        "physics_safety_latched",
        "physics_safety_scripted",
        "dds_bridge",
    ),
    required=True,
)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Flat-G129-SONIC-Official")
parser.add_argument(
    "--asset-profile",
    default="sonic_official_g1",
    help="Compatibility profile under motion_pipeline/config/asset_profiles",
)
parser.add_argument("--disable-unused-contact-sensor", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "motion_pipeline"))

import tasks  # noqa: F401
from action_provider.action_provider_sonic_dds import SonicDDSActionProvider
from dds.dds_master import dds_manager
from dds.g1_robot_dds import G1RobotDDS
from motion_pipeline.asset_profile import load_asset_profile
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states


def summarize_ms(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

    def percentile(value: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * value))]

    return {
        "count": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def qualified_sonic_step(env, action: torch.Tensor) -> None:
    """Step the qualified zero-reward task without unused RL bookkeeping."""

    env.action_manager.process_action(action.to(env.device))
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1


def apply_body_mass_overrides(robot, profile, num_envs: int) -> dict[str, dict[str, float]]:
    """Apply the same deterministic mass/inertia contract as the live runner."""

    if not profile.body_mass_override_kg:
        return {}
    body_names = list(robot.data.body_names)
    body_index = {name: index for index, name in enumerate(body_names)}
    missing = sorted(set(profile.body_mass_override_kg) - set(body_index))
    if missing:
        raise ValueError(
            f"asset profile {profile.profile_id} body-mass overrides refer to "
            f"missing bodies: {missing}"
        )
    masses = robot.root_physx_view.get_masses()
    inertias = robot.root_physx_view.get_inertias()
    applied = {}
    for body_name, target_mass_kg in profile.body_mass_override_kg.items():
        index = body_index[body_name]
        original_mass = masses[:, index].clone()
        ratio = float(target_mass_kg) / original_mass
        masses[:, index] = float(target_mass_kg)
        inertias[:, index] *= ratio[:, None]
        applied[body_name] = {
            "authored_mass_kg": float(original_mass[0]),
            "runtime_mass_kg": float(target_mass_kg),
        }
    env_ids_cpu = torch.arange(num_envs, dtype=torch.int32, device="cpu")
    robot.root_physx_view.set_masses(masses, env_ids_cpu)
    robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
    return applied


@torch.jit.script
def scripted_safety_metrics(
    root: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        (
            root[2],
            torch.max(torch.abs(joint_velocity)),
            torch.isfinite(joint_position).all().to(root.dtype),
            torch.isfinite(joint_velocity).all().to(root.dtype),
            torch.isfinite(root).all().to(root.dtype),
        )
    )


class PipelinedReadback:
    """Copy safety scalars asynchronously and consume them one step later."""

    def __init__(self, size: int, device):
        self.host = [
            torch.empty(size, dtype=torch.float32, pin_memory=True)
            for _ in range(2)
        ]
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.ready = [False, False]
        self.stream = torch.cuda.Stream(device=device)
        self.index = 0

    def submit(self, values: torch.Tensor) -> list[float] | None:
        previous = 1 - self.index
        result = None
        if self.ready[previous]:
            self.events[previous].synchronize()
            result = self.host[previous].tolist()
        current_stream = torch.cuda.current_stream(values.device)
        self.stream.wait_stream(current_stream)
        with torch.cuda.stream(self.stream):
            self.host[self.index].copy_(values, non_blocking=True)
            self.events[self.index].record(self.stream)
        self.ready[self.index] = True
        self.index = previous
        return result


def main() -> None:
    if args.steps <= 0 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup-steps non-negative")
    output = args.output.resolve()
    exchange = (PROJECT_ROOT.parent / "motion_exchange").resolve()
    if not output.is_relative_to(exchange / "diagnostics/performance"):
        raise ValueError("benchmark output must be under motion_exchange/diagnostics/performance")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.sim.render_interval = 20
    if args.disable_unused_contact_sensor:
        cfg.scene.contact_forces = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.sim.reset()
    env.reset()
    robot = env.scene["robot"]
    profile = load_asset_profile(
        args.asset_profile,
        PROJECT_ROOT.parent / "motion_pipeline/config/asset_profiles",
    )
    profile.assert_task(args.task)
    profile.assert_articulation(list(robot.data.joint_names))
    body_mass_overrides = apply_body_mass_overrides(robot, profile, env.num_envs)
    args.asset_profile_contract = profile.runtime_contract()
    if args.component in {
        "qualified_step",
        "physics_safety",
        "physics_safety_async",
        "physics_safety_latched",
        "physics_safety_scripted",
        "dds_bridge",
    }:
        if env.termination_manager.active_terms or env.reward_manager.active_terms:
            raise RuntimeError("qualified step requires no reward or termination terms")
        if env.command_manager.active_terms:
            raise RuntimeError("qualified step requires no command terms")
        if "interval" in env.event_manager.available_modes:
            raise RuntimeError("qualified step requires no interval events")
        if env.recorder_manager.active_terms:
            raise RuntimeError("qualified step requires no recorder terms")

    provider = None
    g1_dds = None
    if args.component == "dds_bridge":
        g1_dds = G1RobotDDS(node_name="g1_latency_component_benchmark")
        if not dds_manager.register_object("g129", g1_dds):
            raise RuntimeError("DDS object g129 is already registered")
        dds_manager.set_publish_rate("g129", 200.0)
        dds_manager.start_publishing(["g129"])
        dds_manager.start_subscribing(["g129"])
        provider = SonicDDSActionProvider(env, args)

    action = torch.zeros_like(robot.data.default_joint_pos)
    physics_ms: list[float] = []
    safety_ms: list[float] = []
    bridge_ms: list[float] = []
    loop_ms: list[float] = []
    sync_checksum = 0.0
    safety_readback = (
        PipelinedReadback(5, env.device)
        if args.component == "physics_safety_async"
        else None
    )
    safety_latched_flags = torch.zeros(3, dtype=torch.bool, device=env.device)
    sim_step_s = float(cfg.sim.dt * cfg.decimation)
    measured_started = None
    total_steps = args.warmup_steps + args.steps

    with torch.inference_mode():
        for step in range(total_steps):
            if step == args.warmup_steps:
                measured_started = time.perf_counter()
            loop_started = time.perf_counter()
            if args.component == "dds_bridge":
                bridge_started = time.perf_counter()
                if step % 4 == 0:
                    get_robot_boy_joint_states(
                        env,
                        enable_dds=True,
                        joint_name_map=profile.contract_to_asset_joint,
                        joint_axis_signs=profile.joint_axis_sign,
                    )
                action = provider.get_action(env)
                if step >= args.warmup_steps:
                    bridge_ms.append((time.perf_counter() - bridge_started) * 1000.0)

            physics_started = time.perf_counter()
            if args.component in {
                "qualified_step",
                "physics_safety",
                "physics_safety_async",
                "physics_safety_latched",
                "physics_safety_scripted",
                "dds_bridge",
            }:
                qualified_sonic_step(env, action)
            else:
                env.step(action)
            if step >= args.warmup_steps:
                physics_ms.append((time.perf_counter() - physics_started) * 1000.0)

            if args.component in {
                "physics_safety",
                "physics_safety_async",
                "physics_safety_latched",
                "physics_safety_scripted",
                "dds_bridge",
            }:
                safety_started = time.perf_counter()
                root = robot.data.root_state_w[0]
                joint_velocity = robot.data.joint_vel[0]
                if args.component == "physics_safety_scripted":
                    packed = scripted_safety_metrics(
                        root,
                        robot.data.joint_pos,
                        joint_velocity,
                    )
                else:
                    finite_flags = torch.stack(
                        (
                            torch.isfinite(robot.data.joint_pos).all(),
                            torch.isfinite(robot.data.joint_vel).all(),
                            torch.isfinite(root).all(),
                        )
                    )
                    if args.component == "physics_safety_latched":
                        safety_latched_flags.logical_or_(~finite_flags)
                        reported_finite = ~safety_latched_flags
                    else:
                        reported_finite = finite_flags
                    packed = torch.stack(
                        (
                            root[2],
                            torch.max(torch.abs(joint_velocity)),
                            reported_finite[0].to(root.dtype),
                            reported_finite[1].to(root.dtype),
                            reported_finite[2].to(root.dtype),
                        )
                    )
                if args.component == "physics_safety_latched" and step % 4 != 0:
                    packed_values = None
                elif safety_readback is not None:
                    packed_values = safety_readback.submit(packed)
                else:
                    packed_values = packed.detach().cpu().tolist()
                if packed_values is not None:
                    sync_checksum += sum(packed_values)
                    if args.component == "physics_safety_latched":
                        safety_latched_flags.zero_()
                if step >= args.warmup_steps:
                    safety_ms.append((time.perf_counter() - safety_started) * 1000.0)
            if step >= args.warmup_steps:
                loop_ms.append((time.perf_counter() - loop_started) * 1000.0)

    wall_s = time.perf_counter() - measured_started
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "component": args.component,
        "task": args.task,
        "unused_contact_sensor_disabled": args.disable_unused_contact_sensor,
        "asset_profile": profile.profile_id,
        "asset_profile_contract": profile.runtime_contract(),
        "body_mass_overrides": body_mass_overrides,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "physics_dt_s": sim_step_s,
        "simulation_time_s": args.steps * sim_step_s,
        "wall_time_s": wall_s,
        "realtime_factor": args.steps * sim_step_s / wall_s,
        "phase_latency": {
            "physics_step": summarize_ms(physics_ms),
            "critical_safety": summarize_ms(safety_ms),
            "dds_bridge": summarize_ms(bridge_ms),
            "loop": summarize_ms(loop_ms),
        },
        "dds": g1_dds.performance_stats() if g1_dds is not None else None,
        "sync_checksum": sync_checksum,
        "image_version": os.getenv("ISAAC_IMAGE_VERSION", "unitree_sim_isaaclab:isaacsim4.5-gui"),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    if provider is not None:
        provider.cleanup()
    dds_manager.stop_all_communication()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
