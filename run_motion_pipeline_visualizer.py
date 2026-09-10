#!/usr/bin/env python3
"""Display authoritative G1 state in a non-authoritative Isaac WebRTC viewer."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


PROCESS_STARTED = time.monotonic()
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
MOTION_PIPELINE_ROOT = PROJECT_ROOT.parent / "motion_pipeline"
sys.path.insert(0, str(MOTION_PIPELINE_ROOT))

from motion_pipeline.asset_profile import load_asset_profile
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Flat-G129-SONIC-Official")
parser.add_argument("--asset-profile", default="sonic_official_g1")
parser.add_argument(
    "--visual-state-input",
    default="/motion_exchange/.runtime/g1_visual_state.bin",
)
parser.add_argument("--target-fps", type=float, default=20.0)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--stale-timeout", type=float, default=1.0)
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument(
    "--camera-eye", type=float, nargs=3, default=(2.8, -3.0, 1.75)
)
parser.add_argument(
    "--camera-lookat", type=float, nargs=3, default=(0.0, 0.0, 0.82)
)
parser.add_argument(
    "--visual-material-preset",
    choices=("source", "urdf_contrast"),
    default="urdf_contrast",
    help=(
        "urdf_contrast corrects the two existing imported shader constants "
        "without adding materials, bindings, or render features"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.target_fps <= 0.0:
    parser.error("--target-fps must be positive")
if args_cli.width <= 0 or args_cli.height <= 0:
    parser.error("viewer resolution must be positive")
if not Path(args_cli.visual_state_input).is_absolute():
    parser.error("--visual-state-input must be an absolute path")

profile = load_asset_profile(
    args_cli.asset_profile, MOTION_PIPELINE_ROOT / "config/asset_profiles"
)
profile.assert_task(args_cli.task)
if not profile.qualified:
    parser.error(f"asset profile {profile.profile_id} is not qualified")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Sdf, UsdShade

import tasks  # noqa: F401
from action_provider.action_provider_sonic_dds import G1_MOTOR_JOINTS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from motion_pipeline.visual_state import VisualStateReader


def summarize_ms(values) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
        return ordered[index]

    return {
        "count": len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def apply_imported_shader_contrast(stage) -> int:
    """Recover the source URDF's light/black contrast in-place.

    Isaac's URDF conversion collapsed the authored 0.70 light material to 1.0
    and the 0.05 black material to 0.498.  Updating the two existing shader
    inputs does not author material bindings, resync prims, add draw calls, or
    change the qualified robot asset on disk.
    """

    corrected = {
        "DefaultMaterial": Gf.Vec3f(0.70, 0.70, 0.70),
        "DefaultMaterial_0": Gf.Vec3f(0.05, 0.05, 0.05),
    }
    updates = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        material_name = prim.GetParent().GetName()
        color = corrected.get(material_name)
        if color is None:
            continue
        shader = UsdShade.Shader(prim)
        diffuse = shader.GetInput("diffuse_color_constant")
        if diffuse and diffuse.Set(color):
            updates += 1
    return updates


def main() -> int:
    if not args_cli.headless and int(args_cli.livestream) == 0:
        print(
            "[visualizer] warning: neither --headless livestream nor desktop GUI "
            "was explicitly selected"
        )

    running = True

    def request_stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    runtime_dir = PROJECT_ROOT.parent / "motion_exchange/.runtime"
    performance_dir = PROJECT_ROOT.parent / "motion_exchange/diagnostics/performance"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    performance_dir.mkdir(parents=True, exist_ok=True)
    status_path = runtime_dir / "visualizer_status.json"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = performance_dir / f"gui_visualizer_{run_id}.json"
    reader = VisualStateReader(args_cli.visual_state_input)
    env = None
    result = "STOPPED"
    render_ms = deque(maxlen=20000)
    fresh_snapshot_age_ms = deque(maxlen=20000)
    frames = 0
    unique_snapshots = 0
    stale_frames = 0
    streaming_frames = 0
    last_sequence = None
    latest_snapshot = None
    started_at = datetime.now(timezone.utc)
    loop_started_at = None
    next_status = 0.0

    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.scene.contact_forces = None
        env_cfg.sim.render_interval = 1
        print("[visualizer] creating read-only Isaac environment", flush=True)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.sim.reset()
        env.reset()
        if args_cli.visual_material_preset == "urdf_contrast":
            updated_shaders = apply_imported_shader_contrast(env.sim.stage)
            if updated_shaders != 2:
                raise RuntimeError(
                    "expected to correct two imported G1 shaders, updated "
                    f"{updated_shaders}"
                )
            print(
                "[visualizer] corrected two imported G1 shader constants "
                "without rebinding materials",
                flush=True,
            )
        robot = env.scene["robot"]
        names = robot.data.joint_names
        asset_index = {name: index for index, name in enumerate(names)}
        asset_indices = [
            asset_index[profile.contract_to_asset_joint[name]]
            for name in G1_MOTOR_JOINTS
        ]
        axis_signs = torch.tensor(
            [profile.joint_axis_sign[name] for name in G1_MOTOR_JOINTS],
            dtype=torch.float32,
            device=env.device,
        )
        q_asset = robot.data.default_joint_pos.clone()
        qd_asset = torch.zeros_like(q_asset)
        root_state = robot.data.root_state_w.clone()

        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("viewer experience did not create an active viewport")
        viewport.set_texture_resolution((int(args_cli.width), int(args_cli.height)))
        viewport.camera_path = Sdf.Path("/OmniverseKit_Persp")
        env.sim.set_camera_view(
            eye=list(args_cli.camera_eye),
            target=list(args_cli.camera_lookat),
            camera_prim_path="/OmniverseKit_Persp",
        )
        for _ in range(3):
            simulation_app.update()
        startup_s = time.monotonic() - PROCESS_STARTED
        print(
            f"[visualizer] READY startup={startup_s:.3f}s "
            f"resolution={args_cli.width}x{args_cli.height} "
            f"target={args_cli.target_fps:.1f}fps",
            flush=True,
        )
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "state": "WAITING_FOR_STATE",
                "authoritative": False,
                "control_topics_published": [],
                "startup_s": startup_s,
                "updated_epoch_s": time.time(),
            },
        )

        loop_started_at = time.monotonic()
        frame_period = 1.0 / args_cli.target_fps
        next_frame = loop_started_at
        while running and simulation_app.is_running():
            now = time.monotonic()
            if args_cli.duration > 0.0 and now - loop_started_at >= args_cli.duration:
                result = "COMPLETED"
                break
            snapshot = reader.read()
            if snapshot is not None:
                latest_snapshot = snapshot
                if snapshot.age_s <= args_cli.stale_timeout:
                    fresh_snapshot_age_ms.append(snapshot.age_s * 1000.0)
                if snapshot.sequence != last_sequence:
                    unique_snapshots += 1
                    last_sequence = snapshot.sequence
                    q_contract = torch.tensor(
                        snapshot.joint_positions,
                        dtype=torch.float32,
                        device=env.device,
                    )
                    qd_contract = torch.tensor(
                        snapshot.joint_velocities,
                        dtype=torch.float32,
                        device=env.device,
                    )
                    q_asset[0, asset_indices] = q_contract * axis_signs
                    qd_asset[0, asset_indices] = qd_contract * axis_signs
                    root_state[0, :7] = torch.tensor(
                        snapshot.root_pose_wxyz,
                        dtype=torch.float32,
                        device=env.device,
                    )
                    root_state[0, 7:13] = 0.0
                    robot.write_root_state_to_sim(root_state)
                    robot.write_joint_state_to_sim(q_asset, qd_asset)
                    env.sim.forward()
            if latest_snapshot is None or latest_snapshot.age_s > args_cli.stale_timeout:
                stale_frames += 1
            else:
                streaming_frames += 1

            render_started = time.perf_counter()
            env.sim.render()
            render_ms.append((time.perf_counter() - render_started) * 1000.0)
            frames += 1
            now = time.monotonic()
            if now >= next_status:
                elapsed = max(now - loop_started_at, 1.0e-9)
                state = (
                    "STREAMING"
                    if latest_snapshot is not None
                    and latest_snapshot.age_s <= args_cli.stale_timeout
                    else "WAITING_FOR_STATE"
                )
                atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "state": state,
                        "authoritative": False,
                        "control_topics_published": [],
                        "frames": frames,
                        "achieved_fps": frames / elapsed,
                        "unique_snapshots": unique_snapshots,
                        "streaming_frames": streaming_frames,
                        "snapshot_age_ms": (
                            None if latest_snapshot is None else latest_snapshot.age_s * 1000.0
                        ),
                        "render": summarize_ms(render_ms),
                        "fresh_snapshot_age": summarize_ms(fresh_snapshot_age_ms),
                        "updated_epoch_s": time.time(),
                    },
                )
                next_status = now + 1.0
            next_frame += frame_period
            delay = next_frame - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_frame = time.monotonic()
        else:
            result = "GUI_CLOSED"
    except Exception as exc:
        result = "FAILED"
        print(f"[visualizer] FAILED: {exc}", flush=True)
        raise
    finally:
        finished = time.monotonic()
        runtime_s = (
            None if loop_started_at is None else max(0.0, finished - loop_started_at)
        )
        report = {
            "schema_version": 1,
            "result": result,
            "authoritative": False,
            "purpose": "read_only_state_mirror",
            "control_topics_published": [],
            "task": args_cli.task,
            "asset_profile": profile.profile_id,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "startup_s": (
                None if loop_started_at is None else loop_started_at - PROCESS_STARTED
            ),
            "runtime_s": runtime_s,
            "target_fps": args_cli.target_fps,
            "achieved_fps": (
                None if not runtime_s or runtime_s <= 0.0 else frames / runtime_s
            ),
            "resolution": [args_cli.width, args_cli.height],
            "frames": frames,
            "unique_snapshots": unique_snapshots,
            "streaming_frames": streaming_frames,
            "stale_frames": stale_frames,
            "render": summarize_ms(render_ms),
            "fresh_snapshot_age": summarize_ms(fresh_snapshot_age_ms),
            "visual_state_input": args_cli.visual_state_input,
            "cuda_device": str(env.device) if env is not None else None,
        }
        try:
            atomic_json(report_path, report)
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "state": result,
                    "authoritative": False,
                    "control_topics_published": [],
                    "report_path": str(report_path),
                    "updated_epoch_s": time.time(),
                },
            )
            print(f"[visualizer] result={result}; report={report_path}", flush=True)
        finally:
            reader.close()
            if env is not None:
                env.close()
            simulation_app.close()
    return 0 if result in {"COMPLETED", "STOPPED", "GUI_CLOSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
