#!/usr/bin/env python3
"""Run the isolated official SONIC G1 baseline with low-overhead diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Flat-G129-SONIC-Official")
parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 runs until Ctrl+C")
parser.add_argument(
    "--unsupported-duration",
    type=float,
    default=0.0,
    help="Stop successfully after N seconds of unsupported simulation time",
)
parser.add_argument("--stats-interval", type=float, default=1.0)
parser.add_argument("--sonic-command-timeout", type=float, default=0.25)
parser.add_argument("--release-delay", type=float, default=2.0, help="Seconds of live LowCmd before releasing bootstrap support")
parser.add_argument("--no-bootstrap-support", action="store_true", help="Disable the official gantry-equivalent root support")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import tasks  # noqa: F401  Registers the environment.
from action_provider.action_provider_sonic_dds import G1_MOTOR_JOINTS, SonicDDSActionProvider
from dds.dds_master import dds_manager
from dds.g1_robot_dds import G1RobotDDS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states


TASK_NAME = args_cli.task


def main() -> int:
    env = None
    provider = None
    running = True
    result = "STOPPED"
    samples: list[dict[str, float | str]] = []
    support_active = True

    def request_stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_utc = datetime.now(timezone.utc)
    output_dir = PROJECT_ROOT.parent / "motion_exchange/diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_slug = TASK_NAME.lower().replace("isaac-flat-", "").replace("-", "_")
    log_path = output_dir / f"sonic_{task_slug}_{started_utc.strftime('%Y%m%dT%H%M%SZ')}.json"

    try:
        env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
        env.sim.reset()
        env.reset()

        robot = env.scene["robot"]
        names = list(robot.data.joint_names)
        missing = [name for name in G1_MOTOR_JOINTS if name not in names]
        if missing:
            raise RuntimeError(
                f"SONIC body joint contract mismatch: missing={missing}, count={len(names)}"
            )
        extras = [name for name in names if name not in G1_MOTOR_JOINTS]
        print(
            f"[baseline] verified SONIC 29-DOF body contract; "
            f"articulation_count={len(names)}, extra_joints={extras}"
        )
        print(f"[baseline] native order={names}")
        print(f"[baseline] initial root height={float(robot.data.root_pos_w[0, 2]):.4f} m")

        g1_dds = G1RobotDDS(node_name="g1_sonic_official_baseline")
        if not dds_manager.register_object("g129", g1_dds):
            raise RuntimeError("DDS object g129 is already registered")
        dds_manager.set_publish_rate("g129", 200.0)
        dds_manager.start_publishing(["g129"])
        dds_manager.start_subscribing(["g129"])

        provider = SonicDDSActionProvider(env, args_cli)
        start = time.monotonic()
        next_stats = start
        unsafe_reason = None
        first_step = True
        support_active = not args_cli.no_bootstrap_support
        support_release_at = None
        support_release_step = None
        support_pose = robot.data.root_state_w[:, :7].clone()
        support_velocity = torch.zeros_like(robot.data.root_state_w[:, 7:13])
        step_count = 0
        sim_step_s = float(env_cfg.sim.dt * env_cfg.decimation)

        print("[baseline] GUI ready. Start SONIC and send ']' only; do NOT send 'T'.")
        print(f"[baseline] diagnostics will be written to {log_path}")

        with torch.inference_mode():
            while running and simulation_app.is_running():
                get_robot_boy_joint_states(env, enable_dds=True)
                action = provider.get_action(env)
                now = time.monotonic()
                if support_active and provider.has_fresh_command:
                    if support_release_at is None:
                        support_release_at = now + max(0.0, args_cli.release_delay)
                        print(
                            "[baseline] first live LowCmd received; "
                            f"releasing bootstrap support in {args_cli.release_delay:.1f}s"
                        )
                    elif now >= support_release_at:
                        support_active = False
                        support_release_step = step_count
                        print("[baseline] bootstrap support released; unsupported stability test started")
                if support_active:
                    robot.write_root_pose_to_sim(support_pose)
                    robot.write_root_velocity_to_sim(support_velocity)
                if first_step:
                    q_error = torch.abs(robot.data.default_joint_pos[0] - robot.data.joint_pos[0])
                    error_index = int(torch.argmax(q_error).item())
                    torque_index = int(torch.argmax(torch.abs(action[0])).item())
                    print(
                        "[baseline] first-step command: "
                        f"max|q_default-q|={float(q_error[error_index]):.6f}rad "
                        f"joint={names[error_index]}, "
                        f"max|tau|={float(torch.abs(action[0, torque_index])):.6f}Nm "
                        f"joint={names[torque_index]}"
                    )
                    first_step = False
                env.step(action)
                step_count += 1
                if support_active:
                    # Match the official MuJoCo gantry: do not let gravity move
                    # the floating base before the tracker has taken control.
                    robot.write_root_pose_to_sim(support_pose)
                    robot.write_root_velocity_to_sim(support_velocity)

                now = time.monotonic()
                root_height = float(robot.data.root_pos_w[0, 2])
                joint_vel = robot.data.joint_vel[0]
                max_index = int(torch.argmax(torch.abs(joint_vel)).item())
                max_dq = float(torch.abs(joint_vel[max_index]))
                finite = bool(
                    torch.isfinite(robot.data.joint_pos).all()
                    and torch.isfinite(robot.data.joint_vel).all()
                    and torch.isfinite(robot.data.root_state_w).all()
                )

                if now >= next_stats:
                    unsupported_sim_time = (
                        None
                        if support_release_step is None
                        else (step_count - support_release_step) * sim_step_s
                    )
                    sample = {
                        "elapsed_s": round(now - start, 3),
                        "simulation_time_s": round(step_count * sim_step_s, 3),
                        "unsupported_simulation_time_s": (
                            None if unsupported_sim_time is None else round(unsupported_sim_time, 3)
                        ),
                        "root_height_m": root_height,
                        "max_abs_joint_velocity_rad_s": max_dq,
                        "max_velocity_joint": names[max_index],
                        "bootstrap_support_active": support_active,
                    }
                    samples.append(sample)
                    print(
                        "[baseline] "
                        f"t={sample['elapsed_s']:.1f}s root_z={root_height:.3f}m "
                        f"max|dq|={max_dq:.3f}rad/s joint={names[max_index]}"
                    )
                    next_stats = now + max(0.1, args_cli.stats_interval)

                if not finite:
                    unsafe_reason = "non-finite robot state"
                elif root_height < 0.51:
                    unsafe_reason = f"fall threshold crossed: root_height={root_height:.4f}m"
                elif max_dq > 35.0:
                    unsafe_reason = (
                        f"SONIC velocity threshold crossed: joint={names[max_index]}, dq={max_dq:.4f}rad/s"
                    )

                if unsafe_reason is not None:
                    result = "UNSAFE"
                    print(f"[baseline] UNSAFE: {unsafe_reason}")
                    break
                if args_cli.duration > 0 and now - start >= args_cli.duration:
                    result = "SUPPORTED_STABLE" if support_active else "STABLE"
                    break
                if (
                    args_cli.unsupported_duration > 0
                    and support_release_step is not None
                    and (step_count - support_release_step) * sim_step_s
                    >= args_cli.unsupported_duration
                ):
                    result = "STABLE"
                    print(
                        "[baseline] completed "
                        f"{args_cli.unsupported_duration:.1f}s unsupported simulation-time hold"
                    )
                    break
        if result == "STOPPED" and running:
            result = "GUI_CLOSED"

    except Exception as exc:
        result = "FAILED"
        unsafe_reason = str(exc)
        print(f"[baseline] FAILED: {exc}")
        raise
    finally:
        report = {
            "schema_version": "1.0",
            "task": TASK_NAME,
            "started_at": started_utc.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "reason": unsafe_reason,
            "bootstrap_support_released": not support_active,
            "dds_domain": os.getenv("DDS_DOMAIN", "42"),
            "dds_interface": os.getenv("DDS_INTERFACE"),
            "lowstate_topic": os.getenv("SIM_LOWSTATE_TOPIC", "rt/socialnav_sim/g1/lowstate"),
            "lowcmd_topic": os.getenv("SIM_LOWCMD_TOPIC", "rt/socialnav_sim/g1/lowcmd"),
            "samples": samples,
        }
        log_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[baseline] result={result}; report={log_path}")
        if provider is not None:
            provider.cleanup()
        dds_manager.stop_all_communication()
        if env is not None:
            env.close()
        simulation_app.close()
    return 0 if result in {"STABLE", "SUPPORTED_STABLE", "STOPPED", "GUI_CLOSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
