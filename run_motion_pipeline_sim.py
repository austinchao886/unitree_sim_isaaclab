#!/usr/bin/env python3
"""Run a qualified SONIC-compatible G1 asset as the Isaac execution endpoint."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


PROCESS_STARTED_MONOTONIC = time.monotonic()
PROCESS_STARTED_UTC = datetime.now(timezone.utc)


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
MOTION_PIPELINE_ROOT = PROJECT_ROOT.parent / "motion_pipeline"
sys.path.insert(0, str(MOTION_PIPELINE_ROOT))

from motion_pipeline.asset_profile import load_asset_profile
from motion_pipeline.bootstrap_support import (
    bootstrap_controller_is_quiet,
    bootstrap_root_is_stable,
    elastic_support_scale,
    resolve_elastic_target_height,
)
from motion_pipeline.runtime_lifecycle import RuntimeState


ASSET_PROFILE_DIR = MOTION_PIPELINE_ROOT / "config/asset_profiles"

# joint_pos.csv uses SONIC/IsaacLab order while DDS uses Unitree motor order.
# Each entry is the CSV column selected for the corresponding Unitree motor.
G1_UNITREE_FROM_ISAACLAB = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]


def resolve_reference_path(value: str) -> Path:
    """Resolve only pipeline-owned reference paths visible inside Isaac."""
    raw = Path(value)
    if raw.is_absolute() and raw.parts[:2] == ("/", "motion_exchange"):
        raw = PROJECT_ROOT.parent / "motion_exchange" / Path(*raw.parts[2:])
    resolved = raw.resolve()
    allowed_roots = (
        (PROJECT_ROOT.parent / "motion_exchange").resolve(),
        (
            PROJECT_ROOT.parent
            / "motion_pipeline/vendor/GR00T-WholeBodyControl/gear_sonic_deploy/reference/example"
        ).resolve(),
    )
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"reference path is outside allowed roots: {resolved}")
    return resolved


def load_reference_joint_positions(path: Path) -> list[list[float]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) != 29:
            raise ValueError(f"reference joint_pos must have 29 columns: {path}")
        rows = []
        for frame, row in enumerate(reader):
            if len(row) != 29:
                raise ValueError(
                    f"reference joint_pos frame {frame} has {len(row)} columns"
                )
            isaaclab = [float(value) for value in row]
            if not all(math.isfinite(value) for value in isaaclab):
                raise ValueError(f"non-finite reference joint_pos at frame {frame}")
            rows.append([isaaclab[index] for index in G1_UNITREE_FROM_ISAACLAB])
    if not rows:
        raise ValueError(f"reference joint_pos is empty: {path}")
    return rows


def summarize_ms(values) -> dict[str, float | int | None]:
    """Summarize bounded latency samples without adding runtime dependencies."""

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


class AsyncTraceWriter:
    """Bounded non-blocking JSONL writer; safety status uses a separate path."""

    def __init__(self, path: Path, max_pending: int = 1024):
        self.path = path
        self.queue = queue.Queue(maxsize=max_pending)
        self.thread = None
        self.error: str | None = None
        self.written = 0
        self.dropped = 0
        self.close_duration_s = None

    def __enter__(self):
        def worker() -> None:
            try:
                with self.path.open("w") as handle:
                    while True:
                        row = self.queue.get()
                        if row is None:
                            break
                        handle.write(row)
                        self.written += 1
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception as exc:
                self.error = str(exc)

        self.thread = threading.Thread(
            target=worker,
            name="isaac-trace-writer",
            daemon=True,
        )
        self.thread.start()
        return self

    def write(self, row: str) -> None:
        if self.error is not None:
            raise RuntimeError(f"trace writer failed: {self.error}")
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def __exit__(self, exc_type, _exc, _traceback):
        close_started = time.monotonic()
        try:
            self.queue.put(None, timeout=5.0)
        except queue.Full:
            self.error = self.error or "trace queue did not drain within 5 seconds"
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            if self.thread.is_alive():
                self.error = self.error or "trace writer did not stop within 10 seconds"
        self.close_duration_s = time.monotonic() - close_started
        if exc_type is None and self.error is not None:
            raise RuntimeError(self.error)
        return False

    def stats(self) -> dict[str, int | str | None]:
        return {
            "written_samples": self.written,
            "dropped_samples": self.dropped,
            "error": self.error,
            "close_duration_s": self.close_duration_s,
        }

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Flat-G129-SONIC-Official")
parser.add_argument(
    "--asset-profile",
    default="sonic_official_g1",
    help="Named compatibility profile under motion_pipeline/config/asset_profiles",
)
parser.add_argument(
    "--fixed-root-lowcmd-diagnostic",
    action="store_true",
    help=(
        "Accept bounded LowCmd probes without an approved artifact. This mode "
        "requires fixed root support and never advertises an executable READY endpoint."
    ),
)
parser.add_argument(
    "--body-properties-output",
    default=None,
    help=(
        "Optional JSON diagnostic path for resolved rigid-body mass, COM and inertia. "
        "The path must be inside motion_exchange."
    ),
)
parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds; 0 runs until Ctrl+C")
parser.add_argument(
    "--unsupported-duration",
    type=float,
    default=0.0,
    help="Stop successfully after N seconds of unsupported simulation time",
)
parser.add_argument("--stats-interval", type=float, default=1.0)
parser.add_argument("--trace-hz", type=float, default=50.0)
parser.add_argument(
    "--visual-state-output",
    default=os.getenv("SIM_VISUAL_STATE_PATH"),
    help=(
        "Optional fixed-size mmap written at LowState cadence for a separate "
        "read-only visualizer; it is never used for control"
    ),
)
parser.add_argument(
    "--render-interval",
    type=int,
    default=20,
    help="Physics steps per GUI render (20 = 10 Hz rendering at 200 Hz physics)",
)
parser.add_argument(
    "--no-realtime-limit",
    action="store_true",
    help="Run physics as fast as possible instead of pacing the 200 Hz servo to wall time",
)
parser.add_argument(
    "--sonic-command-timeout",
    type=float,
    default=0.5,
    help="LowCmd timeout while settling or executing an approved motion",
)
parser.add_argument(
    "--standing-command-timeout",
    type=float,
    default=1.0,
    help="LowCmd timeout while the persistent controller is idle-standing",
)
parser.add_argument(
    "--bootstrap-support",
    choices=("elastic", "fixed", "none"),
    default="elastic",
    help=(
        "Bootstrap support model. 'elastic' matches the official SONIC MuJoCo "
        "pelvis band; 'fixed' is diagnostic-only."
    ),
)
parser.add_argument(
    "--release-delay",
    type=float,
    default=0.25,
    help="Minimum seconds of live LowCmd before support may be released",
)
parser.add_argument(
    "--release-stable-duration",
    type=float,
    default=1.0,
    help=(
        "Supported quiet-window diagnostic duration; free-base release uses a "
        "bounded warm-up envelope and strict quiet gating after support fade"
    ),
)
parser.add_argument(
    "--release-max-tracking-error",
    type=float,
    default=0.5,
    help=(
        "Deprecated diagnostic value. LowCmd q_target error is not reference "
        "tracking error and is not used to release bootstrap support."
    ),
)
parser.add_argument(
    "--release-max-root-height-error",
    type=float,
    default=0.05,
    help="Maximum supported root-height error before support release (m)",
)
parser.add_argument(
    "--release-max-root-tilt",
    type=float,
    default=0.15,
    help="Maximum supported root tilt before support release (rad)",
)
parser.add_argument(
    "--release-max-root-vertical-velocity",
    type=float,
    default=0.15,
    help="Maximum supported vertical root speed before support release (m/s)",
)
parser.add_argument(
    "--release-max-joint-velocity",
    type=float,
    default=1.0,
    help="Maximum joint speed throughout the rolling release window (rad/s)",
)
parser.add_argument(
    "--release-max-target-rate",
    type=float,
    default=8.0,
    help="Maximum SONIC q-target change rate during the release window (rad/s)",
)
parser.add_argument(
    "--release-max-torque-ratio",
    type=float,
    default=0.95,
    help="Maximum requested effort/official actuator limit during release",
)
parser.add_argument(
    "--command-blend-duration",
    type=float,
    default=0.1,
    help=(
        "Simulation seconds to blend the previous effort into SONIC CONTROL; "
        "keep this short so policy action history and physical response stay "
        "synchronized"
    ),
)
parser.add_argument(
    "--command-blend-trigger-delta",
    type=float,
    default=0.01,
    help=(
        "Arm the LowCmd handoff on the first live command, then begin the "
        "effort blend when a policy target changes by this many radians"
    ),
)
parser.add_argument(
    "--support-fade-min-handoff-progress",
    type=float,
    default=1.0,
    help=(
        "Minimum completed effort-handoff fraction before support fade may "
        "start; the default waits until SONIC owns the full command"
    ),
)
parser.add_argument(
    "--bootstrap-target-slew-rate",
    type=float,
    default=0.0,
    help=(
        "Simulation-only q_des rate limit during bootstrap (rad/s); zero "
        "disables target shaping and preserves the raw SONIC target"
    ),
)
parser.add_argument(
    "--bootstrap-target-catchup-error",
    type=float,
    default=0.05,
    help=(
        "Maximum raw-to-shaped q_des difference before support fade may start"
    ),
)
parser.add_argument(
    "--bootstrap-damping-multiplier",
    type=float,
    default=1.0,
    help=(
        "Temporary LowCmd kd multiplier during supported handoff/fade; restored "
        "to the authoritative SONIC value after unsupported quiet gating"
    ),
)
parser.add_argument(
    "--bootstrap-waist-pitch-hold-gain-multiplier",
    type=float,
    default=4.0,
    help=(
        "Simulation-only pre-CONTROL waist_pitch neutral-hold stiffness "
        "multiplier; damping is scaled by its square root"
    ),
)
parser.add_argument(
    "--support-attitude-fade-duration",
    type=float,
    default=0.1,
    help=(
        "Simulation seconds to remove pelvis XY/attitude support before "
        "unloading the vertical band"
    ),
)
parser.add_argument(
    "--support-fade-duration",
    type=float,
    default=0.5,
    help="Simulation seconds to fade elastic pelvis support from full to zero",
)
parser.add_argument(
    "--fade-max-joint-velocity",
    type=float,
    default=4.5,
    help="Abort support fade if any joint exceeds this speed (rad/s)",
)
parser.add_argument(
    "--fade-velocity-violation-duration",
    type=float,
    default=0.02,
    help=(
        "Continuous time above --fade-max-joint-velocity required to abort; "
        "a 2x threshold excursion still aborts immediately"
    ),
)
parser.add_argument(
    "--unsupported-stable-duration",
    type=float,
    default=0.5,
    help="Continuous quiet time required after support reaches zero",
)
parser.add_argument(
    "--max-settle-tracking-error",
    type=float,
    default=2.5,
    help=(
        "Hard LowCmd q_target-to-actual error threshold before support release "
        "(rad); retained flag name for CLI compatibility"
    ),
)
parser.add_argument(
    "--max-settle-duration",
    type=float,
    default=30.0,
    help="Abort if the supported controller never reaches the release gate",
)
parser.add_argument(
    "--max-runtime-tracking-error",
    type=float,
    default=1.0,
    help="Abort playback above this actual-to-reference joint error (rad)",
)
parser.add_argument(
    "--max-runtime-command-position-error",
    type=float,
    default=3.6,
    help=(
        "Abort unsupported execution above this LowCmd q_target-to-actual error "
        "(rad).  Calibrated above the 3.234 rad official MuJoCo baseline peak; "
        "this is not reference tracking error."
    ),
)
parser.add_argument(
    "--max-settle-joint-velocity",
    type=float,
    default=0.0,
    help=(
        "Optional absolute pre-release velocity abort threshold. Zero uses each "
        "official actuator's velocity limit."
    ),
)
parser.add_argument(
    "--elastic-target-height",
    type=float,
    default=None,
    help=(
        "Explicit elastic-band pelvis target height in metres. By default the "
        "runner holds the spawn height while idle, then uses reference frame zero."
    ),
)
parser.add_argument(
    "--no-bootstrap-support",
    action="store_true",
    help="Deprecated alias for --bootstrap-support none",
)
parser.add_argument(
    "--exit-after-command",
    action="store_true",
    help="Exit successfully after a live SONIC session ends; useful for one approved motion",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record approved reference playback from a fixed viewport camera",
)
parser.add_argument(
    "--video-output",
    default=None,
    help="MP4 filename under motion_exchange/executions (default: run-specific name)",
)
parser.add_argument(
    "--replay-trace",
    default=None,
    help="Offline-render a completed execution JSONL instead of starting DDS control",
)
parser.add_argument(
    "--replay-motion-id",
    default=None,
    help="Motion artifact whose timing.json receives the non-blocking video result",
)
parser.add_argument(
    "--replay-cold-bootstrap",
    action="store_true",
    help="Record offline rendering as excluded cold shader/bootstrap time",
)
parser.add_argument(
    "--replay-max-frames",
    type=int,
    default=0,
    help="Limit offline replay output frames for diagnostics; zero renders all frames",
)
parser.add_argument("--video-fps", type=float, default=20.0)
parser.add_argument(
    "--video-quality-profile",
    choices=("performance", "offline-high"),
    default="performance",
    help=(
        "performance preserves the qualified live capture path; offline-high "
        "uses a 1440p DLSS Quality render and is restricted to trace replay"
    ),
)
parser.add_argument(
    "--video-width",
    type=int,
    default=None,
    help="Override video width (profile default: 1280 or 2560)",
)
parser.add_argument(
    "--video-height",
    type=int,
    default=None,
    help="Override video height (profile default: 720 or 1440)",
)
parser.add_argument(
    "--video-camera-eye",
    type=float,
    nargs=3,
    default=(2.8, -3.0, 1.75),
    metavar=("X", "Y", "Z"),
    help="Fixed world-frame camera position; default is a right-front full-body view",
)
parser.add_argument(
    "--video-camera-lookat",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.82),
    metavar=("X", "Y", "Z"),
    help="Fixed world-frame camera target",
)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video_width is None:
    args_cli.video_width = (
        2560 if args_cli.video_quality_profile == "offline-high" else 1280
    )
if args_cli.video_height is None:
    args_cli.video_height = (
        1440 if args_cli.video_quality_profile == "offline-high" else 720
    )

if args_cli.visual_state_output:
    visual_state_output = Path(args_cli.visual_state_output)
    if not visual_state_output.is_absolute():
        parser.error("--visual-state-output must be an absolute path")
    os.environ["SIM_VISUAL_STATE_PATH"] = str(visual_state_output)

if args_cli.video:
    if not args_cli.enable_cameras:
        parser.error("--video requires --enable_cameras")
    if args_cli.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if args_cli.video_width <= 0 or args_cli.video_height <= 0:
        parser.error("--video-width and --video-height must be positive")
if args_cli.replay_trace and not args_cli.video:
    parser.error("--replay-trace requires --video")
if args_cli.replay_trace and not args_cli.replay_motion_id:
    parser.error("--replay-trace requires --replay-motion-id")
if args_cli.replay_motion_id and not args_cli.replay_trace:
    parser.error("--replay-motion-id requires --replay-trace")
if args_cli.video_quality_profile == "offline-high" and not args_cli.replay_trace:
    parser.error("--video-quality-profile offline-high requires --replay-trace")
if not math.isfinite(args_cli.sonic_command_timeout) or args_cli.sonic_command_timeout <= 0.0:
    parser.error("--sonic-command-timeout must be finite and positive")
if (
    not math.isfinite(args_cli.standing_command_timeout)
    or args_cli.standing_command_timeout < args_cli.sonic_command_timeout
):
    parser.error(
        "--standing-command-timeout must be finite and at least "
        "--sonic-command-timeout"
    )
for name in (
    "release_delay",
    "release_stable_duration",
    "release_max_joint_velocity",
    "release_max_target_rate",
    "release_max_torque_ratio",
    "command_blend_duration",
    "command_blend_trigger_delta",
    "bootstrap_target_slew_rate",
    "bootstrap_target_catchup_error",
    "support_attitude_fade_duration",
    "support_fade_duration",
    "fade_max_joint_velocity",
    "fade_velocity_violation_duration",
    "unsupported_stable_duration",
):
    value = float(getattr(args_cli, name))
    if not math.isfinite(value) or value < 0.0:
        parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
if not 0.0 <= args_cli.support_fade_min_handoff_progress <= 1.0:
    parser.error("--support-fade-min-handoff-progress must be within [0, 1]")
if (
    not math.isfinite(args_cli.bootstrap_damping_multiplier)
    or args_cli.bootstrap_damping_multiplier < 1.0
):
    parser.error("--bootstrap-damping-multiplier must be finite and at least 1.0")
if (
    not math.isfinite(args_cli.bootstrap_waist_pitch_hold_gain_multiplier)
    or args_cli.bootstrap_waist_pitch_hold_gain_multiplier < 1.0
):
    parser.error(
        "--bootstrap-waist-pitch-hold-gain-multiplier must be finite and "
        "at least 1.0"
    )

ASSET_PROFILE = load_asset_profile(args_cli.asset_profile, ASSET_PROFILE_DIR)
ASSET_PROFILE.assert_task(args_cli.task)
if args_cli.fixed_root_lowcmd_diagnostic:
    if args_cli.bootstrap_support != "fixed" or args_cli.no_bootstrap_support:
        parser.error("--fixed-root-lowcmd-diagnostic requires --bootstrap-support fixed")
    if args_cli.exit_after_command:
        parser.error("a fixed-root LowCmd diagnostic cannot execute an approved motion")
elif not ASSET_PROFILE.qualified:
    parser.error(
        f"asset profile {ASSET_PROFILE.profile_id} is {ASSET_PROFILE.qualification}; "
        "full motion execution is blocked"
    )
args_cli.asset_profile_contract = ASSET_PROFILE.runtime_contract()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if args_cli.video_quality_profile == "offline-high":
    import carb

    renderer_settings = carb.settings.get_settings()
    renderer_settings.set("/rtx/post/dlss/execMode", 2)
    print(
        "[pipeline-sim] offline video quality profile: "
        f"resolution={args_cli.video_width}x{args_cli.video_height}, "
        "DLSS=Quality, live physics/DDS path disabled by trace-replay gate",
        flush=True,
    )

import gymnasium as gym
import torch

# IsaacLab 2.1 still calls the deprecated wrapper from articulation accessors.
# It is exactly an alias for quat_apply but emits an omni.log warning on every
# call, which can make GUI execution slower than the 50 Hz SONIC reference.
import isaaclab.utils.math as isaac_math

isaac_math.quat_rotate = isaac_math.quat_apply

import tasks  # noqa: F401  Registers the environment.
from action_provider.action_provider_sonic_dds import G1_MOTOR_JOINTS, SonicDDSActionProvider
from dds.dds_master import dds_manager
from dds.g1_robot_dds import G1RobotDDS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states


TASK_NAME = args_cli.task
DIAGNOSTIC_MODE = bool(args_cli.fixed_root_lowcmd_diagnostic)


# Velocity limits from the pinned SONIC G1_CYLINDER_MODEL_12_DEX_CFG.  Runtime
# monitoring uses the per-joint contract instead of a single threshold that is
# too strict for wrists but too loose for hip pitch/roll and knees.
SONIC_VELOCITY_LIMIT_RAD_S = {
    "left_hip_pitch_joint": 20.0,
    "left_hip_roll_joint": 20.0,
    "left_hip_yaw_joint": 32.0,
    "left_knee_joint": 20.0,
    "left_ankle_pitch_joint": 37.0,
    "left_ankle_roll_joint": 37.0,
    "right_hip_pitch_joint": 20.0,
    "right_hip_roll_joint": 20.0,
    "right_hip_yaw_joint": 32.0,
    "right_knee_joint": 20.0,
    "right_ankle_pitch_joint": 37.0,
    "right_ankle_roll_joint": 37.0,
    "waist_yaw_joint": 32.0,
    "waist_roll_joint": 37.0,
    "waist_pitch_joint": 37.0,
    "left_shoulder_pitch_joint": 37.0,
    "left_shoulder_roll_joint": 37.0,
    "left_shoulder_yaw_joint": 37.0,
    "left_elbow_joint": 37.0,
    "left_wrist_roll_joint": 37.0,
    "left_wrist_pitch_joint": 22.0,
    "left_wrist_yaw_joint": 22.0,
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}

# Effort limits from the same pinned G1_CYLINDER_MODEL_12_DEX_CFG.  These are
# used only to measure/control bootstrap transients; PhysX retains the official
# actuator limits and remains authoritative for effort clamping.
SONIC_EFFORT_LIMIT_NM = {
    "left_hip_pitch_joint": 139.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 50.0,
    "left_ankle_roll_joint": 50.0,
    "right_hip_pitch_joint": 139.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 50.0,
    "right_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}


def quaternion_rotation_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert normalized wxyz quaternions to the shortest rotation vector."""

    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    quaternion = torch.where(quaternion[:, :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[:, 1:4]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[:, :1].clamp_min(1.0e-8))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    return torch.where(vector_norm > 1.0e-8, axis * angle, 2.0 * vector)


@torch.jit.script
def scripted_critical_metrics(
    root_state: torch.Tensor,
    all_joint_position: torch.Tensor,
    all_joint_velocity: torch.Tensor,
    actual_unitree: torch.Tensor,
    joint_velocity: torch.Tensor,
    desired_joint_position: torch.Tensor,
    applied_desired_joint_position: torch.Tensor,
    reference_joint_position: torch.Tensor,
    velocity_limits: torch.Tensor,
    applied_motor_torque: torch.Tensor,
    effort_limits: torch.Tensor,
    command_enabled: bool,
    tracking_enabled: bool,
) -> torch.Tensor:
    """Fuse the 200 Hz safety reductions into one TorchScript graph."""

    root_quaternion = root_state[3:7]
    upright_cosine = 1.0 - 2.0 * (
        root_quaternion[1] ** 2 + root_quaternion[2] ** 2
    )
    absolute_velocity = torch.abs(joint_velocity)
    max_velocity_index = torch.argmax(absolute_velocity)
    velocity_ratio = absolute_velocity / velocity_limits
    max_ratio_index = torch.argmax(velocity_ratio)
    absolute_torque = torch.abs(applied_motor_torque)
    max_torque_index = torch.argmax(absolute_torque)
    torque_ratio = absolute_torque / effort_limits
    max_torque_ratio_index = torch.argmax(torque_ratio)
    finite = torch.stack(
        (
            torch.isfinite(all_joint_position).all(),
            torch.isfinite(all_joint_velocity).all(),
            torch.isfinite(root_state).all(),
        )
    ).all()
    command_error = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    command_index = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    raw_command_error = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    raw_command_index = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    target_shaping_error = torch.zeros((), dtype=root_state.dtype, device=root_state.device)
    if command_enabled:
        command_errors = torch.abs(applied_desired_joint_position - actual_unitree)
        command_index = torch.argmax(command_errors).to(root_state.dtype)
        command_error = command_errors[command_index.to(torch.long)]
        raw_command_errors = torch.abs(desired_joint_position - actual_unitree)
        raw_command_index = torch.argmax(raw_command_errors).to(root_state.dtype)
        raw_command_error = raw_command_errors[raw_command_index.to(torch.long)]
        target_shaping_error = torch.max(
            torch.abs(desired_joint_position - applied_desired_joint_position)
        )
    tracking_error = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    tracking_index = torch.zeros((), dtype=root_state.dtype, device=root_state.device) - 1.0
    if tracking_enabled:
        tracking_errors = torch.abs(reference_joint_position - actual_unitree)
        tracking_index = torch.argmax(tracking_errors).to(root_state.dtype)
        tracking_error = tracking_errors[tracking_index.to(torch.long)]
    # Only values needed for immediate 200 Hz safety decisions cross the
    # GPU/CPU boundary.  Full root state is sampled separately at trace/status
    # cadence and remains present in the execution trace.
    return torch.stack(
        (
            root_state[2],
            root_state[9],
            upright_cosine,
            absolute_velocity[max_velocity_index],
            max_velocity_index.to(root_state.dtype),
            velocity_ratio[max_ratio_index],
            max_ratio_index.to(root_state.dtype),
            finite.to(root_state.dtype),
            command_error,
            command_index,
            raw_command_error,
            raw_command_index,
            target_shaping_error,
            tracking_error,
            tracking_index,
            absolute_torque[max_torque_index],
            max_torque_index.to(root_state.dtype),
            torque_ratio[max_torque_ratio_index],
            max_torque_ratio_index.to(root_state.dtype),
        )
    )


def qualified_sonic_step(env, action: torch.Tensor) -> None:
    """Run the qualified zero-reward task without unused RL bookkeeping.

    Physics, effort application, scene writes/updates, and the 200 Hz external
    safety monitor are unchanged.  The qualified task has no command, reward,
    termination, interval-event, or recorder terms, and the runner does not
    consume its policy observation, so recomputing those managers is pure
    overhead.
    """

    env.action_manager.process_action(action.to(env.device))
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    if (
        env._sim_step_counter % env.cfg.sim.render_interval == 0
        and getattr(env, "_sonic_rendering_enabled", False)
    ):
        env.sim.render()
    env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1


def main() -> int:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    process_cpu_started = time.process_time()
    env = None
    provider = None
    video_writer = None
    video_queue = None
    video_thread = None
    video_encoder_ready = None
    video_encoder_errors: list[str] = []
    video_frames = 0
    video_close_error = None
    viewport_api = None
    capture_viewport_frame = None
    schedule_viewport_frame = None
    video_capture_pending = False
    video_capture_handle = None
    running = True
    result = "STOPPED"
    unsafe_reason = None
    samples: list[dict[str, float | str | bool | None]] = []
    support_active = True
    trace_writer = None
    phase_samples_ms = {
        "lowstate_bridge": deque(maxlen=20000),
        "lowcmd_action": deque(maxlen=20000),
        "physics_step": deque(maxlen=20000),
        "critical_monitor": deque(maxlen=20000),
        "loop": deque(maxlen=20000),
    }
    runner_ready_monotonic = None
    playback_start_wall = None
    playback_end_wall = None
    video_finalization_s = None

    def request_stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_utc = datetime.now(timezone.utc)
    output_dir = PROJECT_ROOT.parent / "motion_exchange/executions"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = PROJECT_ROOT.parent / "motion_exchange/.runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_status_path = runtime_dir / "isaac_status.json"
    runtime_request_path = runtime_dir / "request.json"
    task_slug = TASK_NAME.lower().replace("isaac-flat-", "").replace("-", "_")
    run_id = started_utc.strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir / f"isaac_{task_slug}_{run_id}.json"
    trace_path = output_dir / f"isaac_{task_slug}_{run_id}.jsonl"
    video_path = None
    shared_video_path = None
    if args_cli.video:
        if args_cli.video_output:
            requested_video = Path(args_cli.video_output)
            if requested_video.is_absolute():
                if requested_video.parts[:2] == ("/", "motion_exchange"):
                    requested_video = (
                        PROJECT_ROOT.parent
                        / "motion_exchange"
                        / Path(*requested_video.parts[2:])
                    )
            else:
                requested_video = output_dir / requested_video
            video_path = requested_video.resolve()
        else:
            video_path = output_dir / f"isaac_{task_slug}_{run_id}.mp4"
        exchange_root = (PROJECT_ROOT.parent / "motion_exchange").resolve()
        if not video_path.is_relative_to(exchange_root):
            raise ValueError(f"video output is outside motion_exchange: {video_path}")
        if video_path.suffix.lower() != ".mp4":
            raise ValueError(f"video output must use an .mp4 suffix: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        shared_video_path = (
            "/motion_exchange/" + str(video_path.relative_to(exchange_root))
        )
    shared_log_path = f"/motion_exchange/executions/{log_path.name}"
    shared_trace_path = f"/motion_exchange/executions/{trace_path.name}"
    session_id = uuid.uuid4().hex
    runtime_performance_base: dict = {}

    def write_runtime_status(state: str, **values) -> None:
        if "performance" not in values and runtime_performance_base:
            values["performance"] = runtime_performance_base
        payload = {
            "schema_version": 1,
            "session_id": session_id,
            "state": state,
            "task": TASK_NAME,
            "asset_profile": ASSET_PROFILE.profile_id,
            "asset_profile_qualified": ASSET_PROFILE.qualified,
            "diagnostic_only": DIAGNOSTIC_MODE,
            "updated_epoch_s": time.time(),
            **values,
        }
        temporary = runtime_status_path.with_name(
            f".{runtime_status_path.name}.{session_id}.tmp"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, runtime_status_path)

    try:
        env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        # This task's runtime safety uses root/joint state, not contact-history
        # tensors.  Disabling the unused ContactSensor leaves physical contact,
        # friction, and collision response intact while avoiding a per-step GPU
        # contact-report copy that alone prevents 0.8x headless realtime.
        env_cfg.scene.contact_forces = None
        if args_cli.video:
            video_step_s = float(env_cfg.sim.dt * env_cfg.decimation)
            env_cfg.sim.render_interval = max(
                1, round(1.0 / (args_cli.video_fps * video_step_s))
            )
        else:
            env_cfg.sim.render_interval = max(1, int(args_cli.render_interval))
        print("[pipeline-sim] creating Isaac environment", flush=True)
        env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
        env._sonic_rendering_enabled = env.sim.has_gui() or env.sim.has_rtx_sensors()
        print("[pipeline-sim] resetting SimulationContext", flush=True)
        env.sim.reset()
        print("[pipeline-sim] resetting environment", flush=True)
        env.reset()
        print("[pipeline-sim] environment reset complete", flush=True)
        if env.termination_manager.active_terms or env.reward_manager.active_terms:
            raise RuntimeError("qualified SONIC runner requires no reward/termination terms")
        if env.command_manager.active_terms:
            raise RuntimeError("qualified SONIC runner requires no command terms")
        if "interval" in env.event_manager.available_modes:
            raise RuntimeError("qualified SONIC runner requires no interval events")
        if env.recorder_manager.active_terms:
            raise RuntimeError("qualified SONIC runner requires no recorder terms")

        if args_cli.video:
            import ctypes

            import imageio.v2 as imageio
            import numpy as np
            from omni.kit.viewport.utility import (
                capture_viewport_to_buffer,
                get_active_viewport,
            )
            from pxr import Gf, Sdf, UsdShade

            if args_cli.video_quality_profile == "offline-high":
                corrected = {
                    "DefaultMaterial": Gf.Vec3f(0.70, 0.70, 0.70),
                    "DefaultMaterial_0": Gf.Vec3f(0.05, 0.05, 0.05),
                }
                updated_shaders = 0
                for prim in env.sim.stage.Traverse():
                    if not prim.IsA(UsdShade.Shader):
                        continue
                    color = corrected.get(prim.GetParent().GetName())
                    if color is None:
                        continue
                    diffuse = UsdShade.Shader(prim).GetInput(
                        "diffuse_color_constant"
                    )
                    if diffuse and diffuse.Set(color):
                        updated_shaders += 1
                if updated_shaders != 2:
                    raise RuntimeError(
                        "offline-high expected two imported G1 shaders, updated "
                        f"{updated_shaders}"
                    )
                print(
                    "[pipeline-sim] offline-high corrected two existing G1 "
                    "shader constants without new materials or draw calls",
                    flush=True,
                )

            viewport_api = get_active_viewport()
            if viewport_api is None:
                raise RuntimeError("headless rendering experience did not create an active viewport")
            viewport_api.set_texture_resolution(
                (int(args_cli.video_width), int(args_cli.video_height))
            )
            viewport_api.camera_path = Sdf.Path("/OmniverseKit_Persp")
            env.sim.set_camera_view(
                eye=list(args_cli.video_camera_eye),
                target=list(args_cli.video_camera_lookat),
                camera_prim_path="/OmniverseKit_Persp",
            )

            def copy_viewport_buffer(buffer, buffer_size, width, height):
                ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(
                    ctypes.c_byte * buffer_size
                )
                ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
                    ctypes.py_object,
                    ctypes.c_char_p,
                ]
                content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                rgba = np.frombuffer(content.contents, dtype=np.uint8).reshape(
                    int(height), int(width), 4
                )
                return rgba[:, :, :3].copy()

            def capture_fixed_viewport_frame(timeout_s: float = 30.0):
                captured: list[np.ndarray] = []
                errors: list[str] = []
                completed = threading.Event()

                def on_capture(buffer, buffer_size, width, height, _format):
                    try:
                        captured.append(
                            copy_viewport_buffer(buffer, buffer_size, width, height)
                        )
                    except Exception as exc:
                        errors.append(str(exc))
                    finally:
                        completed.set()

                capture_handle = capture_viewport_to_buffer(viewport_api, on_capture)
                if capture_handle is None:
                    raise RuntimeError("failed to schedule fixed viewport capture")
                deadline = time.monotonic() + timeout_s
                while not completed.is_set() and time.monotonic() < deadline:
                    simulation_app.update()
                if not completed.is_set():
                    raise RuntimeError("timed out capturing fixed headless viewport")
                if errors:
                    raise RuntimeError(f"fixed viewport capture failed: {errors[0]}")
                if not captured:
                    raise RuntimeError("fixed viewport capture callback returned no frame")
                return captured[0]

            def schedule_fixed_viewport_frame():
                nonlocal video_capture_handle
                nonlocal video_capture_pending
                nonlocal video_frames
                if video_capture_pending:
                    return
                video_capture_pending = True

                def on_capture(buffer, buffer_size, width, height, _format):
                    nonlocal video_capture_handle
                    nonlocal video_capture_pending
                    nonlocal video_frames
                    try:
                        frame = copy_viewport_buffer(
                            buffer, buffer_size, width, height
                        )
                        if frame.size == 0:
                            raise RuntimeError("fixed viewport callback returned an empty frame")
                        video_queue.put_nowait(frame)
                        video_frames += 1
                    except Exception as exc:
                        video_encoder_errors.append(str(exc))
                    finally:
                        video_capture_pending = False
                        video_capture_handle = None

                video_capture_handle = capture_viewport_to_buffer(
                    viewport_api, on_capture
                )
                if video_capture_handle is None:
                    video_capture_pending = False
                    raise RuntimeError("failed to schedule asynchronous viewport capture")

            capture_viewport_frame = capture_fixed_viewport_frame
            schedule_viewport_frame = schedule_fixed_viewport_frame
            print("[pipeline-sim] rendering fixed viewport warm-up frame", flush=True)
            for _ in range(3):
                simulation_app.update()
            warmup_frame = capture_viewport_frame()
            print("[pipeline-sim] fixed viewport warm-up render complete", flush=True)
            if warmup_frame.size == 0 or not warmup_frame.any():
                raise RuntimeError("fixed viewport returned an empty or black warm-up frame")
            video_writer = imageio.get_writer(
                video_path,
                fps=float(args_cli.video_fps),
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=None,
                ffmpeg_log_level="error",
            )
            video_queue = queue.Queue(maxsize=64)
            video_encoder_ready = threading.Event()

            def encode_video_frames() -> None:
                try:
                    while True:
                        frame = video_queue.get()
                        if frame is None:
                            break
                        video_writer.append_data(frame)
                        video_encoder_ready.set()
                except Exception as exc:
                    video_encoder_errors.append(str(exc))
                    video_encoder_ready.set()
                finally:
                    try:
                        video_writer.close()
                    except Exception as exc:
                        video_encoder_errors.append(str(exc))

            video_thread = threading.Thread(
                target=encode_video_frames,
                name="isaac-video-encoder",
                daemon=True,
            )
            video_thread.start()
            # Starting ffmpeg on the first playback frame can pause the Isaac
            # Python thread long enough for SONIC to declare LowState lost.
            # Prime the encoder before DDS starts; the one neutral lead-in
            # frame also makes the beginning of the motion easier to read.
            video_queue.put(warmup_frame.copy())
            video_frames += 1
            if not video_encoder_ready.wait(timeout=30.0):
                raise RuntimeError("timed out warming the fixed-camera video encoder")
            if video_encoder_errors:
                raise RuntimeError(
                    f"fixed-camera video encoder failed during warm-up: {video_encoder_errors[0]}"
                )
            print(
                "[pipeline-sim] fixed-camera video armed: "
                f"path={video_path}, resolution={args_cli.video_width}x{args_cli.video_height}, "
                f"fps={args_cli.video_fps:g}, eye={tuple(args_cli.video_camera_eye)}, "
                f"lookat={tuple(args_cli.video_camera_lookat)}, "
                f"quality_profile={args_cli.video_quality_profile}"
            )

        robot = env.scene["robot"]
        names = list(robot.data.joint_names)
        ASSET_PROFILE.assert_articulation(names)
        body_names = list(robot.data.body_names)
        if ASSET_PROFILE.body_mass_override_kg:
            body_index = {name: index for index, name in enumerate(body_names)}
            missing_body_overrides = sorted(
                set(ASSET_PROFILE.body_mass_override_kg) - set(body_index)
            )
            if missing_body_overrides:
                raise ValueError(
                    f"asset profile {ASSET_PROFILE.profile_id} body-mass overrides "
                    f"refer to missing bodies: {missing_body_overrides}"
                )
            masses = robot.root_physx_view.get_masses()
            inertias = robot.root_physx_view.get_inertias()
            applied_mass_overrides = {}
            for body_name, target_mass_kg in ASSET_PROFILE.body_mass_override_kg.items():
                index = body_index[body_name]
                original_mass = masses[:, index].clone()
                ratio = float(target_mass_kg) / original_mass
                masses[:, index] = float(target_mass_kg)
                inertias[:, index] *= ratio[:, None]
                applied_mass_overrides[body_name] = {
                    "authored_mass_kg": float(original_mass[0]),
                    "runtime_mass_kg": float(target_mass_kg),
                }
            env_ids_cpu = torch.arange(env.num_envs, dtype=torch.int32, device="cpu")
            robot.root_physx_view.set_masses(masses, env_ids_cpu)
            robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
            print(
                "[baseline] applied deterministic body-mass compatibility overrides: "
                + json.dumps(applied_mass_overrides, sort_keys=True)
            )
        mapped_asset_joints = [
            ASSET_PROFILE.contract_to_asset_joint[name] for name in G1_MOTOR_JOINTS
        ]
        extras = [name for name in names if name not in mapped_asset_joints]
        print(
            f"[baseline] verified asset profile {ASSET_PROFILE.profile_id}; "
            f"articulation_count={len(names)}, extra_joints={extras}"
        )
        print(f"[baseline] native order={names}")
        print(f"[baseline] initial root height={float(robot.data.root_pos_w[0, 2]):.4f} m")

        if args_cli.body_properties_output:
            body_properties_path = Path(args_cli.body_properties_output)
            if body_properties_path.is_absolute() and body_properties_path.parts[:2] == (
                "/",
                "motion_exchange",
            ):
                body_properties_path = (
                    PROJECT_ROOT.parent
                    / "motion_exchange"
                    / Path(*body_properties_path.parts[2:])
                )
            body_properties_path = body_properties_path.resolve()
            exchange_root = (PROJECT_ROOT.parent / "motion_exchange").resolve()
            if not body_properties_path.is_relative_to(exchange_root):
                raise ValueError(
                    "body-properties output is outside motion_exchange: "
                    f"{body_properties_path}"
                )

            def _tensor_first_env(name: str):
                value = getattr(robot.data, name, None)
                if value is None:
                    return None
                data = value.detach().cpu()
                if data.ndim >= 2 and data.shape[0] == env.num_envs:
                    data = data[0]
                return data.tolist()

            masses = robot.root_physx_view.get_masses().detach().cpu()[0].tolist()
            com_pos_b = _tensor_first_env("body_com_pos_b")
            inertias = robot.root_physx_view.get_inertias().detach().cpu()[0].tolist()
            diagnostic = {
                "schema_version": 1,
                "asset_profile": ASSET_PROFILE.profile_id,
                "task": args_cli.task,
                "body_count": len(body_names),
                "total_mass_kg": float(sum(float(value) for value in masses)),
                "bodies": [
                    {
                        "name": name,
                        "mass_kg": float(masses[index]),
                        "com_pos_b_m": None if com_pos_b is None else com_pos_b[index],
                        "inertia": None if inertias is None else inertias[index],
                    }
                    for index, name in enumerate(body_names)
                ],
            }
            body_properties_path.parent.mkdir(parents=True, exist_ok=True)
            body_properties_path.write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
            )
            print(
                "[baseline] wrote rigid-body properties: "
                f"{body_properties_path} total_mass={diagnostic['total_mass_kg']:.4f}kg"
            )

        if args_cli.replay_trace:
            replay_trace_path = Path(args_cli.replay_trace)
            if replay_trace_path.is_absolute() and replay_trace_path.parts[:2] == (
                "/",
                "motion_exchange",
            ):
                replay_trace_path = (
                    PROJECT_ROOT.parent
                    / "motion_exchange"
                    / Path(*replay_trace_path.parts[2:])
                )
            replay_trace_path = replay_trace_path.resolve()
            exchange_root = (PROJECT_ROOT.parent / "motion_exchange").resolve()
            if not replay_trace_path.is_relative_to(exchange_root):
                raise ValueError(
                    f"replay trace is outside motion_exchange: {replay_trace_path}"
                )
            replay_rows = [
                json.loads(line)
                for line in replay_trace_path.read_text().splitlines()
                if line.strip()
            ]
            replay_rows = [
                row
                for row in replay_rows
                if row.get("reference_frame") is not None
                and not row.get("shutdown_requested", False)
            ]
            if not replay_rows:
                raise ValueError(
                    "replay trace contains no completed reference-playback samples"
                )
            replay_start = float(replay_rows[0]["simulation_time_s"])
            replay_end = float(replay_rows[-1]["simulation_time_s"])
            replay_samples = []
            replay_source_index = 0
            replay_target_time = replay_start
            replay_period = 1.0 / float(args_cli.video_fps)
            while replay_target_time <= replay_end + 1.0e-9:
                while (
                    replay_source_index + 1 < len(replay_rows)
                    and abs(
                        float(
                            replay_rows[replay_source_index + 1]["simulation_time_s"]
                        )
                        - replay_target_time
                    )
                    <= abs(
                        float(
                            replay_rows[replay_source_index]["simulation_time_s"]
                        )
                        - replay_target_time
                    )
                ):
                    replay_source_index += 1
                replay_samples.append(replay_rows[replay_source_index])
                replay_target_time += replay_period
            if replay_samples[-1] is not replay_rows[-1]:
                replay_samples.append(replay_rows[-1])
            if args_cli.replay_max_frames > 0:
                replay_samples = replay_samples[: args_cli.replay_max_frames]

            asset_index = {name: index for index, name in enumerate(names)}
            replay_asset_indices = [
                asset_index[ASSET_PROFILE.contract_to_asset_joint[name]]
                for name in G1_MOTOR_JOINTS
            ]
            replay_axis_signs = torch.tensor(
                [ASSET_PROFILE.joint_axis_sign[name] for name in G1_MOTOR_JOINTS],
                dtype=torch.float32,
                device=env.device,
            )
            print(
                f"[pipeline-sim] offline replay: {len(replay_samples)} frames "
                f"from {replay_trace_path}",
                flush=True,
            )
            for replay_index, replay_sample in enumerate(replay_samples):
                q_contract = torch.tensor(
                    replay_sample["joint_pos_unitree_order"],
                    dtype=torch.float32,
                    device=env.device,
                )
                qd_contract = torch.tensor(
                    replay_sample["joint_vel_unitree_order"],
                    dtype=torch.float32,
                    device=env.device,
                )
                q_asset = torch.zeros((1, len(names)), device=env.device)
                qd_asset = torch.zeros_like(q_asset)
                q_asset[0, replay_asset_indices] = q_contract * replay_axis_signs
                qd_asset[0, replay_asset_indices] = qd_contract * replay_axis_signs
                root_state = torch.tensor(
                    [replay_sample["root_state_w"]],
                    dtype=torch.float32,
                    device=env.device,
                )
                robot.write_root_state_to_sim(root_state)
                robot.write_joint_state_to_sim(q_asset, qd_asset)
                env.sim.forward()
                frame = capture_viewport_frame()
                if frame.size == 0:
                    raise RuntimeError("offline replay returned an empty video frame")
                video_queue.put(frame, timeout=5.0)
                video_frames += 1
                if replay_index % 20 == 0 or replay_index + 1 == len(replay_samples):
                    print(
                        f"[pipeline-sim] replay frame {replay_index + 1}/"
                        f"{len(replay_samples)} reference="
                        f"{replay_sample['reference_frame']}",
                        flush=True,
                    )
            result = "COMPLETED"
            support_active = False
            return 0

        g1_dds = G1RobotDDS(node_name="g1_sonic_motion_pipeline")
        if not dds_manager.register_object("g129", g1_dds):
            raise RuntimeError("DDS object g129 is already registered")
        dds_manager.set_publish_rate("g129", 200.0)
        dds_manager.start_publishing(["g129"])
        dds_manager.start_subscribing(["g129"])

        provider = SonicDDSActionProvider(env, args_cli)
        start = time.monotonic()
        next_stats = start
        first_step = True
        support_mode = "none" if args_cli.no_bootstrap_support else args_cli.bootstrap_support
        support_active = support_mode != "none"
        support_release_step_target = None
        support_release_step = None
        support_stable_start_step = None
        support_fade_start_step = None
        fade_velocity_violation_steps = 0
        support_scale = 1.0 if support_active else 0.0
        support_attitude_scale = 1.0 if support_active else 0.0
        bootstrap_phase = "IDLE_SUPPORTED" if support_active else "UNSUPPORTED"
        post_release_stable_start_step = None
        playback_gate_step = None
        playback_start_step = None
        reference_playback_end_step = None
        reference_playback_end_wall = None
        post_hold_ready_reported = False
        shutdown_requested = False
        completed_waiting_for_idle = False
        preempt_support_hold = False
        preempt_request: dict | None = None
        completed_performance: dict = {}
        active_report_path: Path | None = None
        active_shared_report_path: str | None = None
        last_motion_id: str | None = None
        execution_count = 0
        support_pose = robot.data.root_state_w[:, :7].clone()
        support_velocity = torch.zeros_like(robot.data.root_state_w[:, 7:13])
        pelvis_ids, _ = robot.find_bodies("pelvis")
        if len(pelvis_ids) != 1:
            raise RuntimeError(f"expected exactly one pelvis body, found {pelvis_ids}")
        pelvis_id = int(pelvis_ids[0])
        elastic_target_w = support_pose[:, :3].clone()
        elastic_target_w[:, 2] = resolve_elastic_target_height(
            args_cli.elastic_target_height,
            float(support_pose[0, 2]),
        )
        zero_external_wrench = torch.zeros((1, 1, 3), device=env.device)

        def set_elastic_support(
            vertical_scale: float,
            attitude_scale: float,
        ) -> None:
            vertical_scale = max(0.0, min(1.0, float(vertical_scale)))
            attitude_scale = max(0.0, min(1.0, float(attitude_scale)))
            if vertical_scale <= 0.0 and attitude_scale <= 0.0:
                robot.set_external_force_and_torque(
                    zero_external_wrench,
                    zero_external_wrench,
                    body_ids=[pelvis_id],
                )
                return
            pelvis_pos_w = robot.data.body_pos_w[:, pelvis_id]
            pelvis_quat_w = robot.data.body_quat_w[:, pelvis_id]
            pelvis_lin_vel_w = robot.data.body_lin_vel_w[:, pelvis_id]
            pelvis_ang_vel_w = robot.data.body_ang_vel_w[:, pelvis_id]
            # Exact constants from gear_sonic.utils.mujoco_sim.ElasticBand.
            # Release XY/attitude first so the free-base balance policy can
            # take control while vertical fall protection remains.  Damping
            # scales with sqrt(stiffness scale), preserving damping ratio as
            # each spring fades instead of creating an under-damped band.
            spring_force_w = 10000.0 * (elastic_target_w - pelvis_pos_w)
            damping_force_w = -1000.0 * pelvis_lin_vel_w
            spring_force_w[:, :2] *= attitude_scale
            damping_force_w[:, :2] *= math.sqrt(attitude_scale)
            spring_force_w[:, 2] *= vertical_scale
            damping_force_w[:, 2] *= math.sqrt(vertical_scale)
            force_w = spring_force_w + damping_force_w
            spring_torque_w = (
                -1000.0 * quaternion_rotation_vector(pelvis_quat_w)
            )
            damping_torque_w = -10.0 * pelvis_ang_vel_w
            torque_w = (
                attitude_scale * spring_torque_w
                + math.sqrt(attitude_scale) * damping_torque_w
            )
            # IsaacLab 2.1 stores external wrenches in each body's local frame.
            force_b = isaac_math.quat_apply_inverse(pelvis_quat_w, force_w)
            torque_b = isaac_math.quat_apply_inverse(pelvis_quat_w, torque_w)
            robot.set_external_force_and_torque(
                force_b.unsqueeze(1), torque_b.unsqueeze(1), body_ids=[pelvis_id]
            )

        step_count = 0
        sim_step_s = float(env_cfg.sim.dt * env_cfg.decimation)
        video_capture_interval_steps = max(
            1, round(1.0 / (args_cli.video_fps * sim_step_s))
        )
        next_step_deadline = time.monotonic()
        lowstate_interval_steps = max(1, round(1.0 / (50.0 * sim_step_s)))
        trace_interval_steps = max(1, round(1.0 / (max(args_cli.trace_hz, 1.0) * sim_step_s)))
        command_seen = False
        interactive_mode = False
        max_tracking_error = 0.0
        max_tracking_joint = None
        max_tracking_step = None
        max_command_position_error = 0.0
        max_command_position_joint = None
        max_command_position_step = None
        max_settling_joint_velocity = 0.0
        max_settling_joint_velocity_joint = None
        max_settling_torque_ratio = 0.0
        max_settling_torque_ratio_joint = None
        max_settling_target_rate = 0.0
        reference_root_heights: list[float] = []
        reference_joint_positions: list[list[float]] = []
        reference_joint_tensor = None
        reference_hz = 50.0
        post_hold_duration_s = 2.0
        expected_reference_root_height = None
        reference_root_height_error = None
        max_reference_root_height_error = 0.0
        max_reference_root_height_error_frame = None
        active_request: dict | None = None
        settle_start_wall = None
        settle_start_step = None
        release_wall = None
        release_realtime_factor = None
        next_request_check = 0.0
        joint_velocity_limits = torch.tensor(
            [SONIC_VELOCITY_LIMIT_RAD_S[name] for name in G1_MOTOR_JOINTS],
            dtype=torch.float32,
            device=env.device,
        )
        joint_effort_limits = torch.tensor(
            [SONIC_EFFORT_LIMIT_NM[name] for name in G1_MOTOR_JOINTS],
            dtype=torch.float32,
            device=env.device,
        )
        safety_joint_placeholder = torch.zeros(
            29, dtype=torch.float32, device=env.device
        )
        root_state_values = [0.0] * 13
        root_linear_velocity = [0.0, 0.0, 0.0]
        root_angular_velocity = [0.0, 0.0, 0.0]

        def write_persistent_execution_report(finished_wall: float) -> dict:
            """Finalize one command without closing the long-lived simulator."""

            reference_duration_s = (
                len(reference_joint_positions) / reference_hz
                if reference_joint_positions and reference_hz > 0.0
                else None
            )
            playback_wall_s = (
                reference_playback_end_wall - playback_start_wall
                if reference_playback_end_wall is not None
                and playback_start_wall is not None
                else None
            )
            performance = {
                **runtime_performance_base,
                "persistent_session": True,
                "execution_index": execution_count,
                "reference_duration_s": reference_duration_s,
                "reference_playback_wall_s": playback_wall_s,
                "unsupported_playback_realtime_factor": (
                    reference_duration_s / playback_wall_s
                    if reference_duration_s is not None
                    and playback_wall_s is not None
                    and playback_wall_s > 0.0
                    else None
                ),
                "post_hold_wall_s": (
                    max(0.0, finished_wall - reference_playback_end_wall)
                    if reference_playback_end_wall is not None
                    else None
                ),
                "settling_wall_s": (
                    release_wall - settle_start_wall
                    if release_wall is not None and settle_start_wall is not None
                    else 0.0
                ),
                "settling_realtime_factor": release_realtime_factor,
                "phase_latency": {
                    name: summarize_ms(values)
                    for name, values in phase_samples_ms.items()
                },
                "dds": g1_dds.performance_stats()
                if hasattr(g1_dds, "performance_stats")
                else None,
                "lowcmd": provider.performance_stats()
                if hasattr(provider, "performance_stats")
                else None,
                "trace": trace_writer.stats() if trace_writer is not None else None,
                "report_trace_finalization_s": 0.0,
                "video_finalization_s": None,
            }
            report = {
                "schema_version": "1.0",
                "task": TASK_NAME,
                "asset_profile": ASSET_PROFILE.profile_id,
                "asset_profile_contract": ASSET_PROFILE.runtime_contract(),
                "asset_profile_qualified": ASSET_PROFILE.qualified,
                "session_id": session_id,
                "execution_index": execution_count,
                "request_id": (active_request or {}).get("request_id"),
                "motion_id": (active_request or {}).get("motion_id"),
                "started_at": started_utc.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": RuntimeState.COMPLETED.value,
                "bootstrap_support_released": not support_active,
                "bootstrap_support_mode": support_mode,
                "bootstrap_phase": bootstrap_phase,
                "trace_path": str(trace_path),
                "performance": performance,
                "samples": samples,
            }
            if active_report_path is None:
                raise RuntimeError("persistent execution has no report path")
            report_started = time.monotonic()
            active_report_path.write_text(json.dumps(report, indent=2) + "\n")
            performance["report_trace_finalization_s"] = (
                time.monotonic() - report_started
            )
            active_report_path.write_text(json.dumps(report, indent=2) + "\n")
            return performance

        if DIAGNOSTIC_MODE:
            print(
                "[pipeline-sim] fixed-root LowCmd diagnostic endpoint ready; "
                "approved motion execution is disabled"
            )
        else:
            print(
                f"[pipeline-sim] qualified asset {ASSET_PROFILE.profile_id} ready; "
                "waiting for an approved motion"
            )
        print(f"[pipeline-sim] report={log_path}")
        print(f"[pipeline-sim] trace={trace_path}")
        runner_ready_monotonic = time.monotonic()
        runner_startup_s = runner_ready_monotonic - PROCESS_STARTED_MONOTONIC
        startup_performance = {
            "isaac_runner_startup_s": runner_startup_s,
            "runner_started_at": PROCESS_STARTED_UTC.isoformat(),
            "runner_ready_at": datetime.now(timezone.utc).isoformat(),
        }
        runtime_performance_base.update(startup_performance)
        write_runtime_status(
            "DIAGNOSTIC_ONLY" if DIAGNOSTIC_MODE else RuntimeState.READY.value,
            report_path=shared_log_path,
            trace_path=shared_trace_path,
            render_interval=env_cfg.sim.render_interval,
            physics_dt_s=sim_step_s,
            bootstrap_support_mode=support_mode,
            elastic_target_height_m=float(elastic_target_w[0, 2]),
            video_path=shared_video_path,
            performance=startup_performance,
        )

        trace_writer = AsyncTraceWriter(trace_path)
        with trace_writer as trace_handle:
            with torch.inference_mode():
                while running and simulation_app.is_running():
                    loop_started = time.perf_counter()
                    phase_started = loop_started
                    if step_count % lowstate_interval_steps == 0:
                        get_robot_boy_joint_states(
                            env,
                            enable_dds=True,
                            joint_name_map=ASSET_PROFILE.contract_to_asset_joint,
                            joint_axis_signs=ASSET_PROFILE.joint_axis_sign,
                        )
                        phase_samples_ms["lowstate_bridge"].append(
                            (time.perf_counter() - phase_started) * 1000.0
                        )
                    phase_started = time.perf_counter()
                    action = provider.get_action(env)
                    phase_samples_ms["lowcmd_action"].append(
                        (time.perf_counter() - phase_started) * 1000.0
                    )
                    now = time.monotonic()
                    if provider.has_fresh_command and not command_seen:
                        command_seen = True
                        if DIAGNOSTIC_MODE:
                            active_request = {
                                "request_id": "fixed-root-lowcmd-diagnostic",
                                "motion_id": "fixed-root-lowcmd-diagnostic",
                            }
                            settle_start_wall = now
                            settle_start_step = step_count
                            # Diagnostic probes have no supervisor SETTLING
                            # handshake, so they must explicitly leave the
                            # bootstrap neutral hold.  Without this call the
                            # provider caches the bounded LowCmd but correctly
                            # keeps commanding its default pose, producing a
                            # false near-zero response after the production
                            # handoff gate was introduced.
                            provider.begin_control_handoff()
                            print("[pipeline-sim] accepted bounded diagnostic LowCmd stream")
                        if not DIAGNOSTIC_MODE:
                            try:
                                candidate = json.loads(runtime_request_path.read_text())
                            except (FileNotFoundError, json.JSONDecodeError) as exc:
                                unsafe_reason = f"missing or invalid approved runtime request: {exc}"
                                result = "UNSAFE"
                                print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                break
                            if candidate.get("state") == "INTERACTIVE":
                                if candidate.get("isaac_session_id") != session_id:
                                    # The persistent runner can become READY
                                    # before the supervisor notices the new
                                    # session. Keep bootstrap support active
                                    # while that stale request is replaced.
                                    command_seen = False
                                elif candidate.get("asset_profile") != ASSET_PROFILE.profile_id:
                                    unsafe_reason = (
                                        "interactive request asset profile mismatch: "
                                        f"expected={ASSET_PROFILE.profile_id}, "
                                        f"actual={candidate.get('asset_profile')}"
                                    )
                                    result = "UNSAFE"
                                    print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                    break
                                else:
                                    interactive_mode = True
                                    active_request = candidate
                                    provider.set_standing_idle(False)
                                    settle_start_wall = now
                                    settle_start_step = step_count
                                    playback_gate_step = None
                                    post_release_stable_start_step = None
                                    provider.begin_control_handoff()
                                    bootstrap_phase = (
                                        "SUPPORTED_WARMUP"
                                        if support_active
                                        else "INTERACTIVE_GROUNDING"
                                    )
                                    write_runtime_status(
                                        "SETTLING" if support_active else "GROUNDING",
                                        request_id=candidate.get("request_id"),
                                        motion_id=candidate.get("motion_id"),
                                        interactive_source=candidate.get("interactive_source"),
                                        bootstrap_phase=bootstrap_phase,
                                        control_handoff_progress=provider.handoff_progress,
                                    )
                                    print(
                                        "[pipeline-sim] accepted interactive SONIC "
                                        "joystick/planner runtime",
                                        flush=True,
                                    )
                            elif candidate.get("state") in {
                                "STARTING", "IDLE", "STOPPING"
                            }:
                                # SONIC publishes its INIT-ramp command before the
                                # supervisor has entered CONTROL. A persistent
                                # SONIC process also publishes while no execution
                                # is active. Keep support and wait for the explicit
                                # SETTLING handshake instead of accepting either.
                                command_seen = False
                            elif candidate.get("state") != "SETTLING":
                                unsafe_reason = "runtime request is not SETTLING"
                                result = "UNSAFE"
                                print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                break
                            elif candidate.get("isaac_session_id") != session_id:
                                # This can be an approved request from the
                                # previous generation. Do not admit it into a
                                # new physics session; wait under support for
                                # the supervisor to republish against this one.
                                command_seen = False
                            elif candidate.get("asset_profile") != ASSET_PROFILE.profile_id:
                                unsafe_reason = (
                                    "runtime request asset profile mismatch: "
                                    f"expected={ASSET_PROFILE.profile_id}, "
                                    f"actual={candidate.get('asset_profile')}"
                                )
                                result = "UNSAFE"
                                print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                break
                            else:
                                execution_count += 1
                                command_slug = re.sub(
                                    r"[^A-Za-z0-9_.-]+",
                                    "_",
                                    str(candidate.get("motion_id") or "motion"),
                                )
                                active_report_path = output_dir / (
                                    f"isaac_{task_slug}_{run_id}_"
                                    f"{execution_count:04d}_{command_slug}.json"
                                )
                                active_shared_report_path = (
                                    "/motion_exchange/executions/"
                                    + active_report_path.name
                                )
                                samples = []
                                for latency_samples in phase_samples_ms.values():
                                    latency_samples.clear()
                                max_tracking_error = 0.0
                                max_tracking_joint = None
                                max_tracking_step = None
                                max_command_position_error = 0.0
                                max_command_position_joint = None
                                max_command_position_step = None
                                max_settling_joint_velocity = 0.0
                                max_settling_joint_velocity_joint = None
                                max_settling_torque_ratio = 0.0
                                max_settling_torque_ratio_joint = None
                                max_settling_target_rate = 0.0
                                max_reference_root_height_error = 0.0
                                max_reference_root_height_error_frame = None
                                playback_start_step = None
                                playback_start_wall = None
                                playback_end_wall = None
                                reference_playback_end_step = None
                                reference_playback_end_wall = None
                                post_hold_ready_reported = False
                                shutdown_requested = False
                                completed_waiting_for_idle = False
                                completed_performance = {}
                                playback_gate_step = None
                                post_release_stable_start_step = None
                                settle_start_wall = None
                                settle_start_step = None
                                release_wall = None
                                release_realtime_factor = (
                                    None if support_active else 1.0
                                )
                                unsafe_reason = None
                                provider.set_standing_idle(False)
                                interactive_mode = False
                                active_request = candidate
                                try:
                                    reference_root_heights = [
                                        float(value)
                                        for value in candidate["reference_root_heights_m"]
                                    ]
                                    reference_hz = float(candidate.get("reference_hz", 50.0))
                                    post_hold_duration_s = float(
                                        candidate.get("post_hold_s", 2.0)
                                    )
                                    reference_joint_path = resolve_reference_path(
                                        str(candidate["reference_joint_pos_path"])
                                    )
                                    reference_joint_positions = (
                                        load_reference_joint_positions(reference_joint_path)
                                    )
                                    reference_joint_tensor = torch.tensor(
                                        reference_joint_positions,
                                        dtype=torch.float32,
                                        device=env.device,
                                    )
                                except (KeyError, TypeError, ValueError) as exc:
                                    unsafe_reason = f"invalid reference contract: {exc}"
                                    result = "UNSAFE"
                                    print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                    break
                                if (
                                    not reference_root_heights
                                    or not reference_joint_positions
                                    or len(reference_root_heights)
                                        != len(reference_joint_positions)
                                    or reference_hz <= 0.0
                                    or post_hold_duration_s < 0.0
                                ):
                                    unsafe_reason = (
                                        "reference root/joint contract is empty or frame-mismatched"
                                    )
                                    result = "UNSAFE"
                                    print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                    break
                                elastic_target_w[:, 2] = resolve_elastic_target_height(
                                    args_cli.elastic_target_height,
                                    float(support_pose[0, 2]),
                                    reference_root_heights[0],
                                )
                                print(
                                    "[pipeline-sim] elastic support target aligned to "
                                    f"reference frame zero: {float(elastic_target_w[0, 2]):.4f}m"
                                )
                                settle_start_wall = now
                                settle_start_step = step_count
                                bootstrap_phase = (
                                    "SUPPORTED_WARMUP"
                                    if support_active
                                    else "READY_STANDING"
                                )
                                provider.begin_control_handoff()
                                write_runtime_status(
                                    "SETTLING",
                                    request_id=active_request.get("request_id"),
                                    motion_id=active_request.get("motion_id"),
                                    elastic_target_height_m=float(
                                        elastic_target_w[0, 2]
                                    ),
                                    bootstrap_phase=bootstrap_phase,
                                    control_handoff_progress=provider.handoff_progress,
                                )
                    if command_seen and not DIAGNOSTIC_MODE and now >= next_request_check:
                        next_request_check = now + 0.1
                        try:
                            control_request = json.loads(runtime_request_path.read_text())
                        except (FileNotFoundError, json.JSONDecodeError):
                            control_request = {}
                        if (
                            interactive_mode
                            and control_request.get("state") == "STARTING"
                            and control_request.get("isaac_session_id") == session_id
                        ):
                            # A fresh supervisor must never inherit an already
                            # unsupported interactive physics session.  Its
                            # first INIT commands can arrive before the LowCmd
                            # stale timeout and would otherwise look like a
                            # continuation of the old controller.  End this
                            # generation explicitly so the persistent runner
                            # restarts with bootstrap support engaged.
                            result = "SAFE_STOP"
                            unsafe_reason = (
                                "fresh SONIC supervisor requested interactive "
                                "bootstrap"
                            )
                            print(f"[pipeline-sim] {unsafe_reason}", flush=True)
                            write_runtime_status(
                                RuntimeState.SAFE_STOP.value,
                                request_id=control_request.get("request_id"),
                                motion_id=control_request.get("motion_id"),
                                reason=unsafe_reason,
                            )
                            break
                        elif (
                            interactive_mode
                            and control_request.get("state")
                                == "REFERENCE_PREEMPT"
                            and control_request.get("isaac_session_id") == session_id
                            and (
                                not preempt_support_hold
                                or preempt_request is None
                                or preempt_request.get("request_id")
                                    != control_request.get("request_id")
                            )
                        ):
                            # Offline references explicitly require the
                            # bootstrap root band during frame-zero settling.
                            # Reacquire it around the robot's current pose
                            # before SONIC disables the joystick planner.
                            preempt_support_hold = True
                            preempt_request = control_request
                            support_active = support_mode != "none"
                            support_scale = 1.0 if support_active else 0.0
                            # Runtime preemption mirrors the planned physical
                            # suspension: restore vertical fall protection but
                            # leave XY and attitude under SONIC authority. A
                            # full 6-DoF band fights the already-active balance
                            # policy and prevents the neutral gate converging.
                            support_attitude_scale = 0.0
                            support_release_step_target = None
                            support_release_step = None
                            support_stable_start_step = None
                            support_fade_start_step = None
                            fade_velocity_violation_steps = 0
                            post_release_stable_start_step = None
                            playback_gate_step = None
                            elastic_target_w.copy_(
                                robot.data.body_pos_w[:, pelvis_id]
                            )
                            bootstrap_phase = "REFERENCE_PREEMPT_SUPPORTED"
                            write_runtime_status(
                                "PREEMPT_SUPPORTED",
                                request_id=control_request.get("request_id"),
                                motion_id=control_request.get("motion_id"),
                                bootstrap_phase=bootstrap_phase,
                                elastic_support_scale=support_scale,
                                elastic_support_attitude_scale=(
                                    support_attitude_scale
                                ),
                                simulation_time_s=round(
                                    step_count * sim_step_s, 4
                                ),
                            )
                            print(
                                "[pipeline-sim] elastic support reacquired "
                                "for reference preemption",
                                flush=True,
                            )
                        elif (
                            interactive_mode
                            and control_request.get("state") == "SETTLING"
                            and control_request.get("isaac_session_id") == session_id
                        ):
                            # An approved offline reference preempts joystick
                            # locomotion without restarting Isaac or SONIC. The
                            # regular SETTLING admission path runs next tick.
                            interactive_mode = False
                            preempt_support_hold = False
                            preempt_request = None
                            command_seen = False
                            active_request = None
                            playback_gate_step = None
                            post_release_stable_start_step = None
                            reference_root_heights = []
                            reference_joint_positions = []
                            reference_joint_tensor = None
                            expected_reference_root_height = None
                            reference_root_height_error = None
                            provider.set_standing_idle(False)
                            bootstrap_phase = "REFERENCE_PREEMPT"
                            print(
                                "[pipeline-sim] approved reference preempted "
                                "interactive joystick mode",
                                flush=True,
                            )
                        elif (
                            completed_waiting_for_idle
                            and control_request.get("state") == "IDLE"
                            and active_request is not None
                            and control_request.get("motion_id")
                                == active_request.get("motion_id")
                            and control_request.get("isaac_session_id") == session_id
                        ):
                            last_motion_id = str(active_request.get("motion_id"))
                            completed_waiting_for_idle = False
                            command_seen = False
                            interactive_mode = False
                            shutdown_requested = False
                            active_request = None
                            playback_start_step = None
                            playback_start_wall = None
                            playback_end_wall = None
                            reference_playback_end_step = None
                            reference_playback_end_wall = None
                            reference_root_heights = []
                            reference_joint_positions = []
                            reference_joint_tensor = None
                            expected_reference_root_height = None
                            reference_root_height_error = None
                            post_hold_ready_reported = False
                            playback_gate_step = None
                            post_release_stable_start_step = None
                            bootstrap_phase = "READY_STANDING"
                            provider.set_standing_idle(True)
                            write_runtime_status(
                                RuntimeState.READY_STANDING.value,
                                last_motion_id=last_motion_id,
                                execution_count=execution_count,
                                report_path=active_shared_report_path,
                                trace_path=shared_trace_path,
                                simulation_time_s=round(step_count * sim_step_s, 4),
                                bootstrap_support_mode=support_mode,
                                bootstrap_phase="READY_STANDING",
                                performance=completed_performance,
                            )
                            print(
                                "[pipeline-sim] same session returned to READY_STANDING "
                                f"after motion={last_motion_id}",
                                flush=True,
                            )
                        elif control_request.get("state") == "ABORTED":
                            result = "ABORTED"
                            unsafe_reason = "execution aborted by DDS control command"
                            print(f"[pipeline-sim] {unsafe_reason}")
                            write_runtime_status(
                                RuntimeState.SAFE_STOP.value,
                                request_id=(active_request or {}).get("request_id"),
                                motion_id=(active_request or {}).get("motion_id"),
                                reason=unsafe_reason,
                            )
                            break
                        elif (
                            control_request.get("state") == "STOPPING"
                            and active_request is not None
                            and control_request.get("motion_id")
                                == active_request.get("motion_id")
                        ):
                            shutdown_requested = True
                        if (
                            not completed_waiting_for_idle
                            and
                            control_request.get("state") == "PLAYING"
                            and playback_start_step is None
                            and active_request is not None
                            and control_request.get("motion_id")
                                == active_request.get("motion_id")
                        ):
                            playback_start_step = step_count
                            playback_start_wall = now
                            active_request = control_request
                            print(
                                "[pipeline-sim] reference playback clock started at "
                                f"physics step {playback_start_step}"
                            )
                    if args_cli.exit_after_command and command_seen and provider.command_is_stale:
                        playback_end_wall = now
                        playback_elapsed_sim_s = (
                            None
                            if playback_start_step is None
                            else (step_count - playback_start_step) * sim_step_s
                        )
                        required_reference_sim_s = (
                            None
                            if not reference_joint_positions
                            else len(reference_joint_positions) / reference_hz
                        )
                        if playback_start_step is None:
                            result = "UNSAFE"
                            unsafe_reason = (
                                "SONIC command stream ended before reference playback started"
                            )
                            print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                        elif (
                            required_reference_sim_s is None
                            or playback_elapsed_sim_s + sim_step_s
                                < required_reference_sim_s
                        ):
                            result = "UNSAFE"
                            unsafe_reason = (
                                "SONIC command stream ended before the full reference "
                                "duration: "
                                f"elapsed={playback_elapsed_sim_s:.3f}s, "
                                f"required={required_reference_sim_s:.3f}s"
                            )
                            print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                        else:
                            result = "COMPLETED"
                            print("[pipeline-sim] SONIC command stream ended after approved execution")
                        break
                    if (
                        not args_cli.exit_after_command
                        and (execution_count > 0 or interactive_mode)
                        and provider.command_is_stale
                    ):
                        # A persistent interactive controller is just as
                        # safety-critical as offline reference playback.  If
                        # SONIC exits or is recreated, do not keep applying its
                        # last cached LowCmd while the robot is unsupported.
                        # End this generation immediately; the runner service
                        # will start a clean, supported session and the SONIC
                        # supervisor will reconnect to it.
                        result = "SAFE_STOP"
                        unsafe_reason = (
                            "persistent SONIC LowCmd stream became stale: "
                            f"age={provider.command_age_s:.3f}s"
                        )
                        print(f"[pipeline-sim] SAFE_STOP: {unsafe_reason}")
                        write_runtime_status(
                            RuntimeState.SAFE_STOP.value,
                            request_id=(active_request or {}).get("request_id"),
                            motion_id=(active_request or {}).get("motion_id"),
                            reason=unsafe_reason,
                        )
                        break
                    if (
                        support_active
                        and provider.has_fresh_command
                        and not DIAGNOSTIC_MODE
                        and not preempt_support_hold
                    ):
                        if command_seen and support_release_step_target is None:
                            settle_steps = round(
                                max(0.0, args_cli.release_delay) / sim_step_s
                            )
                            support_release_step_target = step_count + settle_steps
                            print(
                                "[pipeline-sim] first live LowCmd received; "
                                "support release eligibility begins after "
                                f"{args_cli.release_delay:.1f}s of simulation time; "
                                f"requires {args_cli.release_stable_duration:.1f}s "
                                "of quiet root/joint/target/effort state"
                            )
                    if support_active:
                        if support_mode == "fixed":
                            robot.write_root_pose_to_sim(support_pose)
                            robot.write_root_velocity_to_sim(support_velocity)
                        elif support_mode == "elastic":
                            if support_fade_start_step is not None:
                                fade_elapsed_s = (
                                    step_count - support_fade_start_step
                                ) * sim_step_s
                                support_attitude_scale = elastic_support_scale(
                                    fade_elapsed_s,
                                    args_cli.support_attitude_fade_duration,
                                )
                                vertical_fade_elapsed_s = max(
                                    0.0,
                                    fade_elapsed_s
                                    - args_cli.support_attitude_fade_duration,
                                )
                                support_scale = elastic_support_scale(
                                    vertical_fade_elapsed_s,
                                    args_cli.support_fade_duration,
                                )
                                bootstrap_phase = (
                                    "ATTITUDE_FADE"
                                    if support_attitude_scale > 0.0
                                    else "VERTICAL_FADE"
                                )
                            set_elastic_support(
                                support_scale,
                                support_attitude_scale,
                            )
                    if first_step:
                        q_error = torch.abs(robot.data.default_joint_pos[0] - robot.data.joint_pos[0])
                        error_index = int(torch.argmax(q_error).item())
                        torque_index = int(torch.argmax(torch.abs(action[0])).item())
                        print(
                            "[pipeline-sim] first-step command: "
                            f"max|q_default-q|={float(q_error[error_index]):.6f}rad "
                            f"joint={names[error_index]}, "
                            f"max|tau|={float(torch.abs(action[0, torque_index])):.6f}Nm "
                            f"joint={names[torque_index]}"
                        )
                        first_step = False
                    phase_started = time.perf_counter()
                    qualified_sonic_step(env, action)
                    phase_samples_ms["physics_step"].append(
                        (time.perf_counter() - phase_started) * 1000.0
                    )
                    step_count += 1
                    monitor_started = time.perf_counter()
                    if (
                        video_writer is not None
                        and playback_start_step is not None
                        and (step_count - playback_start_step)
                            % video_capture_interval_steps == 0
                    ):
                        if video_encoder_errors:
                            raise RuntimeError(
                                f"fixed-camera video encoder failed: {video_encoder_errors[0]}"
                            )
                        schedule_viewport_frame()
                    if support_active and support_mode == "fixed":
                        # Fixed support is retained as a diagnostic, but unlike
                        # the elastic band it suppresses all base feedback.
                        robot.write_root_pose_to_sim(support_pose)
                        robot.write_root_velocity_to_sim(support_velocity)

                    now = time.monotonic()
                    unsupported_sim_time = (
                        None
                        if support_release_step is None
                        else (step_count - support_release_step) * sim_step_s
                    )
                    reference_frame = None
                    expected_reference_root_height = None
                    reference_root_height_error = None
                    tracking_error = None
                    tracking_joint = None
                    if playback_start_step is not None and reference_root_heights:
                        reference_frame = min(
                            int(
                                (step_count - playback_start_step)
                                * sim_step_s
                                * reference_hz
                            ),
                            len(reference_root_heights) - 1,
                        )
                        expected_reference_root_height = reference_root_heights[
                            reference_frame
                        ]

                    # Critical monitoring remains at 200 Hz, but all scalar
                    # decisions cross the GPU/CPU boundary in one packed copy.
                    root_state = robot.data.root_state_w[0]
                    joint_vel = provider.actual_joint_velocities()
                    actual_unitree = provider.actual_joint_positions()
                    # STOPPING intentionally switches SONIC to damping-only
                    # LowCmd. Reference/q_target tracking is then meaningless.
                    tracking_enabled = bool(
                        reference_frame is not None
                        and not shutdown_requested
                        and reference_joint_tensor is not None
                    )
                    desired_for_safety = (
                        provider.desired_joint_positions
                        if provider.desired_joint_positions is not None
                        else safety_joint_placeholder
                    )
                    applied_desired_for_safety = (
                        provider.applied_desired_joint_positions
                        if provider.applied_desired_joint_positions is not None
                        else safety_joint_placeholder
                    )
                    reference_for_safety = (
                        reference_joint_tensor[reference_frame]
                        if tracking_enabled
                        else safety_joint_placeholder
                    )
                    packed_metrics = scripted_critical_metrics(
                        root_state,
                        robot.data.joint_pos,
                        robot.data.joint_vel,
                        actual_unitree,
                        joint_vel,
                        desired_for_safety,
                        applied_desired_for_safety,
                        reference_for_safety,
                        joint_velocity_limits,
                        provider.applied_motor_torques,
                        joint_effort_limits,
                        provider.desired_joint_positions is not None,
                        tracking_enabled,
                    ).detach().cpu().tolist()
                    root_height = packed_metrics[0]
                    root_linear_velocity[2] = packed_metrics[1]
                    upright_cosine = packed_metrics[2]
                    root_tilt = math.acos(max(-1.0, min(1.0, upright_cosine)))
                    max_dq = packed_metrics[3]
                    max_index = int(packed_metrics[4])
                    max_velocity_ratio = packed_metrics[5]
                    max_velocity_ratio_index = int(packed_metrics[6])
                    finite = bool(packed_metrics[7])
                    command_position_error = (
                        None if packed_metrics[8] < 0.0 else packed_metrics[8]
                    )
                    command_index = int(packed_metrics[9])
                    command_position_joint = (
                        None
                        if command_index < 0
                        else G1_MOTOR_JOINTS[command_index]
                    )
                    raw_command_position_error = (
                        None if packed_metrics[10] < 0.0 else packed_metrics[10]
                    )
                    raw_command_index = int(packed_metrics[11])
                    raw_command_position_joint = (
                        None
                        if raw_command_index < 0
                        else G1_MOTOR_JOINTS[raw_command_index]
                    )
                    target_shaping_error = packed_metrics[12]
                    tracking_error = (
                        None if packed_metrics[13] < 0.0 else packed_metrics[13]
                    )
                    tracking_index = int(packed_metrics[14])
                    tracking_joint = (
                        None
                        if tracking_index < 0
                        else G1_MOTOR_JOINTS[tracking_index]
                    )
                    max_applied_torque = packed_metrics[15]
                    max_torque_index = int(packed_metrics[16])
                    max_applied_torque_joint = G1_MOTOR_JOINTS[max_torque_index]
                    max_torque_ratio = packed_metrics[17]
                    max_torque_ratio_index = int(packed_metrics[18])
                    max_torque_ratio_joint = G1_MOTOR_JOINTS[
                        max_torque_ratio_index
                    ]
                    target_rate_rad_s = provider.latest_target_rate_rad_s
                    handoff_progress = provider.handoff_progress

                    if command_seen and support_active:
                        if max_dq > max_settling_joint_velocity:
                            max_settling_joint_velocity = max_dq
                            max_settling_joint_velocity_joint = G1_MOTOR_JOINTS[
                                max_index
                            ]
                        if max_torque_ratio > max_settling_torque_ratio:
                            max_settling_torque_ratio = max_torque_ratio
                            max_settling_torque_ratio_joint = max_torque_ratio_joint
                        max_settling_target_rate = max(
                            max_settling_target_rate, target_rate_rad_s
                        )

                    if command_position_error is not None and (
                        command_position_error > max_command_position_error
                    ):
                        max_command_position_error = command_position_error
                        max_command_position_joint = command_position_joint
                        max_command_position_step = step_count
                    if tracking_error is not None and tracking_error > max_tracking_error:
                        max_tracking_error = tracking_error
                        max_tracking_joint = tracking_joint
                        max_tracking_step = step_count
                    if expected_reference_root_height is not None:
                        reference_root_height_error = (
                            expected_reference_root_height - root_height
                        )
                        if (
                            reference_root_height_error
                            > max_reference_root_height_error
                        ):
                            max_reference_root_height_error = reference_root_height_error
                            max_reference_root_height_error_frame = reference_frame
                    if (
                        playback_start_step is not None
                        and reference_playback_end_step is None
                        and reference_joint_positions
                        and (step_count - playback_start_step) * sim_step_s
                            >= len(reference_joint_positions) / reference_hz
                    ):
                        reference_playback_end_step = step_count
                        reference_playback_end_wall = now

                    if (
                        reference_playback_end_step is not None
                        and not post_hold_ready_reported
                        and (step_count - reference_playback_end_step) * sim_step_s
                            >= post_hold_duration_s
                    ):
                        post_hold_ready_reported = True
                        write_runtime_status(
                            "POST_HOLD_COMPLETE",
                            request_id=(active_request or {}).get("request_id"),
                            motion_id=(active_request or {}).get("motion_id"),
                            simulation_time_s=round(step_count * sim_step_s, 4),
                            post_hold_simulation_time_s=round(
                                (step_count - reference_playback_end_step)
                                * sim_step_s,
                                4,
                            ),
                        )

                    if (
                        command_seen
                        and not support_active
                        and support_release_step is not None
                        and playback_gate_step is None
                    ):
                        grounded = bool(
                            0.70 <= root_height <= 0.90
                            and abs(root_linear_velocity[2]) <= 0.15
                            and root_tilt <= 0.25
                            and bootstrap_controller_is_quiet(
                                max_joint_velocity_rad_s=max_dq,
                                max_target_rate_rad_s=target_rate_rad_s,
                                max_torque_ratio=max_torque_ratio,
                                handoff_progress=handoff_progress,
                                joint_velocity_limit_rad_s=(
                                    args_cli.release_max_joint_velocity
                                ),
                                target_rate_limit_rad_s=(
                                    args_cli.release_max_target_rate
                                ),
                                torque_ratio_limit=(
                                    args_cli.release_max_torque_ratio
                                ),
                            )
                        )
                        if grounded:
                            if post_release_stable_start_step is None:
                                post_release_stable_start_step = step_count
                            stable_sim_s = (
                                step_count - post_release_stable_start_step
                            ) * sim_step_s
                            if stable_sim_s >= args_cli.unsupported_stable_duration:
                                playback_gate_step = step_count
                                bootstrap_phase = (
                                    "INTERACTIVE"
                                    if interactive_mode
                                    else "UNSUPPORTED_QUIET"
                                )
                                provider.end_control_handoff()
                                print(
                                    "[pipeline-sim] unsupported ground gate passed "
                                    f"after {stable_sim_s:.2f}s stable"
                                )
                                write_runtime_status(
                                    "INTERACTIVE" if interactive_mode else "EXECUTING",
                                    request_id=active_request.get("request_id"),
                                    motion_id=active_request.get("motion_id"),
                                    interactive_source=active_request.get(
                                        "interactive_source"
                                    ),
                                    release_realtime_factor=release_realtime_factor,
                                    reference_clock="lowstate_tick",
                                    simulation_time_s=round(step_count * sim_step_s, 4),
                                    bootstrap_support_mode=support_mode,
                                    root_height_m=root_height,
                                    root_tilt_rad=root_tilt,
                                    bootstrap_phase=bootstrap_phase,
                                    max_joint_velocity_rad_s=max_dq,
                                    max_target_rate_rad_s=target_rate_rad_s,
                                    max_torque_ratio=max_torque_ratio,
                                )
                        else:
                            post_release_stable_start_step = None
                    settling_trace_tick = bool(
                        command_seen and playback_gate_step is None
                    )
                    if (
                        settling_trace_tick
                        or step_count % trace_interval_steps == 0
                        or now >= next_stats
                    ):
                        root_state_values = root_state.detach().cpu().tolist()
                        root_linear_velocity = root_state_values[7:10]
                        root_angular_velocity = root_state_values[10:13]
                    if settling_trace_tick or step_count % trace_interval_steps == 0:
                        trace_tensors = [
                            actual_unitree,
                            joint_vel,
                            provider.applied_motor_torques,
                        ]
                        if provider.desired_joint_positions is not None:
                            trace_tensors.append(provider.desired_joint_positions)
                            trace_tensors.append(
                                provider.applied_desired_joint_positions
                            )
                        trace_values = torch.cat(trace_tensors).detach().cpu().tolist()
                        actual_values = trace_values[:29]
                        velocity_values = trace_values[29:58]
                        applied_torque_values = trace_values[58:87]
                        desired_values = (
                            trace_values[87:116]
                            if provider.desired_joint_positions is not None
                            else None
                        )
                        applied_desired_values = (
                            trace_values[116:145]
                            if provider.desired_joint_positions is not None
                            else None
                        )
                        trace = {
                            "simulation_time_s": round(step_count * sim_step_s, 4),
                            "unsupported_simulation_time_s": (
                                None if unsupported_sim_time is None else round(unsupported_sim_time, 4)
                            ),
                            "root_state_w": root_state_values,
                            "root_tilt_rad": root_tilt,
                            "reference_frame": reference_frame,
                            "expected_reference_root_height_m": expected_reference_root_height,
                            "reference_root_height_error_m": reference_root_height_error,
                            "joint_pos_unitree_order": actual_values,
                            "joint_vel_unitree_order": velocity_values,
                            "desired_joint_pos_unitree_order": desired_values,
                            "applied_desired_joint_pos_unitree_order": (
                                applied_desired_values
                            ),
                            "applied_torque_unitree_order_nm": applied_torque_values,
                            "reference_tracking_error_rad": tracking_error,
                            "reference_tracking_error_joint": tracking_joint,
                            "command_position_error_rad": command_position_error,
                            "command_position_error_joint": command_position_joint,
                            "raw_command_position_error_rad": (
                                raw_command_position_error
                            ),
                            "raw_command_position_error_joint": (
                                raw_command_position_joint
                            ),
                            "bootstrap_target_shaping_error_rad": (
                                target_shaping_error
                            ),
                            "sonic_startup_ready": provider.sonic_startup_ready,
                            "bootstrap_support_active": support_active,
                            "bootstrap_support_mode": support_mode,
                            "bootstrap_phase": bootstrap_phase,
                            "elastic_support_scale": support_scale,
                            "elastic_support_attitude_scale": (
                                support_attitude_scale
                            ),
                            "control_handoff_progress": handoff_progress,
                            "bootstrap_damping_multiplier": (
                                provider.bootstrap_damping_multiplier
                            ),
                            "max_target_rate_rad_s": target_rate_rad_s,
                            "max_applied_torque_nm": max_applied_torque,
                            "max_applied_torque_joint": max_applied_torque_joint,
                            "max_torque_limit_ratio": max_torque_ratio,
                            "max_torque_ratio_joint": max_torque_ratio_joint,
                            "shutdown_requested": shutdown_requested,
                            "max_joint_velocity_limit_ratio": max_velocity_ratio,
                        }
                        trace_handle.write(json.dumps(trace, separators=(",", ":")) + "\n")

                    if now >= next_stats:
                        sample = {
                            "elapsed_s": round(now - start, 3),
                            "simulation_time_s": round(step_count * sim_step_s, 3),
                            "unsupported_simulation_time_s": (
                                None if unsupported_sim_time is None else round(unsupported_sim_time, 3)
                            ),
                            "root_height_m": root_height,
                            "root_tilt_rad": root_tilt,
                            "reference_frame": reference_frame,
                            "expected_reference_root_height_m":
                                expected_reference_root_height,
                            "reference_root_height_error_m":
                                reference_root_height_error,
                            "max_abs_joint_velocity_rad_s": max_dq,
                            "max_velocity_joint": G1_MOTOR_JOINTS[max_index],
                            "reference_tracking_error_rad": tracking_error,
                            "reference_tracking_error_joint": tracking_joint,
                            "command_position_error_rad": command_position_error,
                            "command_position_error_joint": command_position_joint,
                            "raw_command_position_error_rad": (
                                raw_command_position_error
                            ),
                            "raw_command_position_error_joint": (
                                raw_command_position_joint
                            ),
                            "bootstrap_target_shaping_error_rad": (
                                target_shaping_error
                            ),
                            "bootstrap_support_active": support_active,
                            "bootstrap_support_mode": support_mode,
                            "bootstrap_phase": bootstrap_phase,
                            "elastic_support_scale": support_scale,
                            "elastic_support_attitude_scale": (
                                support_attitude_scale
                            ),
                            "control_handoff_progress": handoff_progress,
                            "bootstrap_damping_multiplier": (
                                provider.bootstrap_damping_multiplier
                            ),
                            "max_target_rate_rad_s": target_rate_rad_s,
                            "max_applied_torque_nm": max_applied_torque,
                            "max_applied_torque_joint": max_applied_torque_joint,
                            "max_torque_limit_ratio": max_torque_ratio,
                            "max_torque_ratio_joint": max_torque_ratio_joint,
                            "shutdown_requested": shutdown_requested,
                            "max_joint_velocity_limit_ratio": max_velocity_ratio,
                        }
                        samples.append(sample)
                        print(
                            "[pipeline-sim] "
                            f"t={sample['elapsed_s']:.1f}s root_z={root_height:.3f}m "
                            f"max|dq|={max_dq:.3f}rad/s joint={G1_MOTOR_JOINTS[max_index]} "
                            f"tau_ratio={max_torque_ratio:.3f} "
                            f"blend={handoff_progress:.2f} support={support_scale:.2f} "
                            f"kd_x={provider.bootstrap_damping_multiplier:.1f} "
                            f"ref_track={tracking_error if tracking_error is not None else 0.0:.3f}rad "
                            f"cmd_err={command_position_error if command_position_error is not None else 0.0:.3f}rad"
                        )
                        next_stats = now + max(0.1, args_cli.stats_interval)
                        runtime_state = (
                            "DIAGNOSTIC_ONLY"
                            if DIAGNOSTIC_MODE
                            else (
                                RuntimeState.READY_STANDING.value
                                if (
                                    not support_active
                                    and provider.has_fresh_command
                                    and not provider.command_is_stale
                                )
                                else RuntimeState.READY.value
                            )
                        )
                        if preempt_support_hold and preempt_request is not None:
                            runtime_state = "PREEMPT_SUPPORTED"
                        elif completed_waiting_for_idle:
                            runtime_state = RuntimeState.COMPLETED.value
                        elif command_seen:
                            if interactive_mode:
                                runtime_state = (
                                    "INTERACTIVE"
                                    if playback_gate_step is not None
                                    else (
                                        "SETTLING"
                                        if support_active
                                        else "GROUNDING"
                                    )
                                )
                            elif shutdown_requested:
                                runtime_state = "STOPPING"
                            elif post_hold_ready_reported:
                                runtime_state = "POST_HOLD_COMPLETE"
                            elif support_active:
                                runtime_state = "SETTLING"
                            elif playback_gate_step is None:
                                runtime_state = "GROUNDING"
                            else:
                                runtime_state = "EXECUTING"
                        write_runtime_status(
                            runtime_state,
                            request_id=(
                                preempt_request or active_request or {}
                            ).get("request_id"),
                            motion_id=(
                                preempt_request or active_request or {}
                            ).get("motion_id"),
                            report_path=(
                                active_shared_report_path or shared_log_path
                            ),
                            trace_path=shared_trace_path,
                            simulation_time_s=sample["simulation_time_s"],
                            root_height_m=root_height,
                            root_tilt_rad=root_tilt,
                            root_linear_velocity_m_s=root_linear_velocity,
                            root_angular_velocity_rad_s=root_angular_velocity,
                            reference_frame=reference_frame,
                            expected_reference_root_height_m=expected_reference_root_height,
                            reference_root_height_error_m=reference_root_height_error,
                            reference_tracking_error_rad=tracking_error,
                            reference_tracking_error_joint=tracking_joint,
                            command_position_error_rad=command_position_error,
                            command_position_error_joint=command_position_joint,
                            raw_command_position_error_rad=(
                                raw_command_position_error
                            ),
                            raw_command_position_error_joint=(
                                raw_command_position_joint
                            ),
                            bootstrap_target_shaping_error_rad=(
                                target_shaping_error
                            ),
                            release_realtime_factor=release_realtime_factor,
                            bootstrap_support_mode=support_mode,
                            bootstrap_phase=bootstrap_phase,
                            elastic_support_scale=support_scale,
                            elastic_support_attitude_scale=(
                                support_attitude_scale
                            ),
                            control_handoff_progress=handoff_progress,
                            bootstrap_damping_multiplier=(
                                provider.bootstrap_damping_multiplier
                            ),
                            max_joint_velocity_rad_s=max_dq,
                            max_target_rate_rad_s=target_rate_rad_s,
                            max_applied_torque_nm=max_applied_torque,
                            max_torque_limit_ratio=max_torque_ratio,
                            elastic_target_height_m=float(elastic_target_w[0, 2]),
                            last_motion_id=last_motion_id,
                            execution_count=execution_count,
                            performance=(
                                completed_performance
                                if completed_waiting_for_idle
                                else runtime_performance_base
                            ),
                        )

                    if not finite:
                        unsafe_reason = "non-finite robot state"
                    elif playback_start_step is None and root_height < 0.51:
                        unsafe_reason = f"fall threshold crossed: root_height={root_height:.4f}m"
                    elif (
                        playback_start_step is not None
                        and reference_root_height_error is not None
                        and reference_root_height_error > 0.25
                    ):
                        unsafe_reason = (
                            "reference-relative fall threshold crossed: "
                            f"frame={reference_frame}, "
                            f"reference_root_height={expected_reference_root_height:.4f}m, "
                            f"actual_root_height={root_height:.4f}m, "
                            f"error={reference_root_height_error:.4f}m"
                        )
                    elif not support_active and root_tilt > 0.80:
                        unsafe_reason = (
                            f"unsupported root tilt threshold crossed: {root_tilt:.4f}rad"
                        )
                    elif max_velocity_ratio > 1.05:
                        unsafe_reason = (
                            "SONIC actuator velocity limit crossed: "
                            f"joint={G1_MOTOR_JOINTS[max_velocity_ratio_index]}, "
                            f"dq={float(torch.abs(joint_vel[max_velocity_ratio_index])):.4f}rad/s, "
                            f"limit={float(joint_velocity_limits[max_velocity_ratio_index]):.4f}rad/s"
                        )
                    elif (
                        not support_active
                        and not shutdown_requested
                        and command_position_error is not None
                        and command_position_error
                            > args_cli.max_runtime_command_position_error
                    ):
                        unsafe_reason = (
                            "unsupported LowCmd position error threshold crossed: "
                            f"joint={command_position_joint}, "
                            f"error={command_position_error:.4f}rad"
                        )
                    elif (
                        raw_command_position_error is not None
                        and raw_command_position_error
                            > args_cli.max_runtime_command_position_error
                    ):
                        unsafe_reason = (
                            "raw SONIC LowCmd position error threshold crossed: "
                            f"joint={raw_command_position_joint}, "
                            f"error={raw_command_position_error:.4f}rad"
                        )
                    elif (
                        not support_active
                        and tracking_error is not None
                        and tracking_error > args_cli.max_runtime_tracking_error
                    ):
                        unsafe_reason = (
                            "unsupported reference tracking threshold crossed: "
                            f"joint={tracking_joint}, error={tracking_error:.4f}rad"
                        )

                    if unsafe_reason is not None:
                        result = "UNSAFE"
                        print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                        write_runtime_status(
                            "UNSAFE",
                            request_id=(active_request or {}).get("request_id"),
                            motion_id=(active_request or {}).get("motion_id"),
                            reason=unsafe_reason,
                        )
                        break
                    if (
                        not completed_waiting_for_idle
                        and
                        shutdown_requested
                        and reference_playback_end_step is not None
                    ):
                        # The supervisor sets STOPPING only after SONIC emitted
                        # the exact motion-complete event and the full 2 seconds
                        # of monitored simulation post-hold elapsed.  Finish this
                        # runner without killing the preloaded SONIC process.
                        playback_end_wall = now
                        result = "COMPLETED"
                        print(
                            "[pipeline-sim] persistent SONIC execution completed "
                            "after monitored post-hold"
                        )
                        if args_cli.exit_after_command:
                            break
                        completed_performance = write_persistent_execution_report(now)
                        completed_waiting_for_idle = True
                        write_runtime_status(
                            RuntimeState.COMPLETED.value,
                            request_id=(active_request or {}).get("request_id"),
                            motion_id=(active_request or {}).get("motion_id"),
                            report_path=active_shared_report_path,
                            trace_path=shared_trace_path,
                            simulation_time_s=round(step_count * sim_step_s, 4),
                            release_realtime_factor=release_realtime_factor,
                            max_tracking_error_rad=max_tracking_error,
                            max_command_position_error_rad=(
                                max_command_position_error
                            ),
                            max_reference_root_height_error_m=(
                                max_reference_root_height_error
                            ),
                            performance=completed_performance,
                        )
                    if (
                        support_active
                        and command_seen
                        and not DIAGNOSTIC_MODE
                        and not preempt_support_hold
                    ):
                        settle_elapsed_sim_s = (step_count - settle_start_step) * sim_step_s
                        if settle_elapsed_sim_s > args_cli.max_settle_duration:
                            unsafe_reason = (
                                "bootstrap controller did not converge before timeout: "
                                f"settle_simulation_time={settle_elapsed_sim_s:.3f}s"
                            )
                        if (
                            unsafe_reason is None
                            and
                            command_position_error is not None
                            and command_position_error
                                > args_cli.max_settle_tracking_error
                        ):
                            unsafe_reason = (
                                "settle LowCmd position error threshold crossed before "
                                "support release: "
                                f"joint={command_position_joint}, "
                                f"error={command_position_error:.4f}rad"
                            )
                        elif (
                            unsafe_reason is None
                            and
                            args_cli.max_settle_joint_velocity > 0.0
                            and max_dq > args_cli.max_settle_joint_velocity
                        ):
                            unsafe_reason = (
                                "settle velocity threshold crossed before support release: "
                                f"joint={G1_MOTOR_JOINTS[max_index]}, dq={max_dq:.4f}rad/s"
                            )
                        if unsafe_reason is not None:
                            result = "UNSAFE"
                            print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                            write_runtime_status(
                                "UNSAFE",
                                request_id=(active_request or {}).get("request_id"),
                                motion_id=(active_request or {}).get("motion_id"),
                                reason=unsafe_reason,
                            )
                            break
                        root_release_stable = bootstrap_root_is_stable(
                            root_height_m=root_height,
                            target_height_m=float(elastic_target_w[0, 2]),
                            root_tilt_rad=root_tilt,
                            root_vertical_velocity_m_s=root_linear_velocity[2],
                            max_root_height_error_m=(
                                args_cli.release_max_root_height_error
                            ),
                            max_root_tilt_rad=args_cli.release_max_root_tilt,
                            max_root_vertical_velocity_m_s=(
                                args_cli.release_max_root_vertical_velocity
                            ),
                        )
                        controller_quiet = bootstrap_controller_is_quiet(
                            max_joint_velocity_rad_s=max_dq,
                            max_target_rate_rad_s=target_rate_rad_s,
                            max_torque_ratio=max_torque_ratio,
                            handoff_progress=handoff_progress,
                            joint_velocity_limit_rad_s=(
                                args_cli.release_max_joint_velocity
                            ),
                            target_rate_limit_rad_s=(
                                args_cli.release_max_target_rate
                            ),
                            torque_ratio_limit=args_cli.release_max_torque_ratio,
                        )
                        release_stable = root_release_stable and controller_quiet
                        # Every physics tick must remain quiet.  A joint/action
                        # spike now resets this rolling window instead of being
                        # masked by an otherwise stationary supported pelvis.
                        if support_fade_start_step is None:
                            if release_stable:
                                if support_stable_start_step is None:
                                    support_stable_start_step = step_count
                            else:
                                support_stable_start_step = None
                        stable_sim_s = (
                            0.0
                            if support_stable_start_step is None
                            else (step_count - support_stable_start_step)
                            * sim_step_s
                        )
                        # SONIC is a free-base policy and cannot reach a
                        # continuous quiet window against a full pelvis band.
                        # Start the deliberately slow support fade only from a
                        # tight bounded envelope; temporary damping remains
                        # active until the unsupported quiet gate passes.
                        support_warmup_safe = bool(
                            root_release_stable
                            and handoff_progress
                                >= args_cli.support_fade_min_handoff_progress
                            and max_dq <= min(
                                1.5,
                                args_cli.fade_max_joint_velocity,
                            )
                            and max_torque_ratio
                                <= args_cli.release_max_torque_ratio
                            and target_rate_rad_s
                                <= args_cli.release_max_target_rate
                            and target_shaping_error
                                <= args_cli.bootstrap_target_catchup_error
                            and provider.sonic_startup_ready
                        )
                        release_gate_passed = bool(
                            support_release_step_target is not None
                            and step_count >= support_release_step_target
                            and support_warmup_safe
                        )
                        if release_gate_passed and support_fade_start_step is None:
                            if (
                                support_mode == "elastic"
                                and (
                                    args_cli.support_attitude_fade_duration > 0.0
                                    or args_cli.support_fade_duration > 0.0
                                )
                            ):
                                support_fade_start_step = step_count
                                bootstrap_phase = "SUPPORT_FADE"
                                print(
                                    "[pipeline-sim] supported warm-up envelope passed; "
                                    "fading XY/attitude support over "
                                    f"{args_cli.support_attitude_fade_duration:.2f}s, "
                                    "then vertical support over "
                                    f"{args_cli.support_fade_duration:.2f}s"
                                )
                                write_runtime_status(
                                    "SETTLING",
                                    request_id=active_request.get("request_id"),
                                    motion_id=active_request.get("motion_id"),
                                    bootstrap_phase=bootstrap_phase,
                                    elastic_support_scale=support_scale,
                                    elastic_support_attitude_scale=(
                                        support_attitude_scale
                                    ),
                                    max_joint_velocity_rad_s=max_dq,
                                    max_target_rate_rad_s=target_rate_rad_s,
                                    max_torque_ratio=max_torque_ratio,
                                    sonic_startup_ready=(
                                        provider.sonic_startup_ready
                                    ),
                                )
                            else:
                                support_scale = 0.0
                                support_attitude_scale = 0.0

                        if support_fade_start_step is not None:
                            fade_unsafe_reason = None
                            if max_dq > args_cli.fade_max_joint_velocity:
                                fade_velocity_violation_steps += 1
                            else:
                                fade_velocity_violation_steps = 0
                            fade_velocity_violation_s = (
                                fade_velocity_violation_steps * sim_step_s
                            )
                            if root_height < 0.65 or root_tilt > 0.35:
                                fade_unsafe_reason = (
                                    "root state exceeded support-fade envelope: "
                                    f"height={root_height:.4f}m, tilt={root_tilt:.4f}rad"
                                )
                            elif (
                                max_dq > args_cli.fade_max_joint_velocity
                                and (
                                    max_dq >= 2.0
                                        * args_cli.fade_max_joint_velocity
                                    or fade_velocity_violation_s
                                        >= args_cli.fade_velocity_violation_duration
                                )
                            ):
                                fade_unsafe_reason = (
                                    "joint velocity exceeded support-fade envelope: "
                                    f"joint={G1_MOTOR_JOINTS[max_index]}, "
                                    f"dq={max_dq:.4f}rad/s, "
                                    "continuous_duration="
                                    f"{fade_velocity_violation_s:.4f}s"
                                )
                            elif max_torque_ratio > 1.05:
                                fade_unsafe_reason = (
                                    "requested effort exceeded support-fade envelope: "
                                    f"joint={max_torque_ratio_joint}, "
                                    f"ratio={max_torque_ratio:.4f}"
                                )
                            if fade_unsafe_reason is not None:
                                unsafe_reason = fade_unsafe_reason
                                result = "UNSAFE"
                                print(f"[pipeline-sim] UNSAFE: {unsafe_reason}")
                                write_runtime_status(
                                    "UNSAFE",
                                    request_id=(active_request or {}).get("request_id"),
                                    motion_id=(active_request or {}).get("motion_id"),
                                    reason=unsafe_reason,
                                    bootstrap_phase=bootstrap_phase,
                                    elastic_support_scale=support_scale,
                                    elastic_support_attitude_scale=(
                                        support_attitude_scale
                                    ),
                                )
                                break

                        release_now = bool(
                            not support_active
                            or (
                                support_fade_start_step is not None
                                and support_scale <= 0.0
                                and support_attitude_scale <= 0.0
                            )
                            or (
                                release_gate_passed
                                and support_mode != "elastic"
                            )
                            or (
                                release_gate_passed
                                and args_cli.support_attitude_fade_duration <= 0.0
                                and args_cli.support_fade_duration <= 0.0
                            )
                        )
                        if release_now and support_active:
                            settle_sim_s = (
                                step_count - settle_start_step
                            ) * sim_step_s
                            settle_wall_s = max(now - settle_start_wall, 1e-6)
                            release_realtime_factor = settle_sim_s / settle_wall_s
                            support_active = False
                            support_scale = 0.0
                            support_attitude_scale = 0.0
                            support_release_step = step_count
                            release_wall = now
                            bootstrap_phase = "UNSUPPORTED_SETTLE"
                            if support_mode == "elastic":
                                set_elastic_support(0.0, 0.0)
                            print(
                                "[pipeline-sim] bootstrap support fully released "
                                "after quiet gate and gradual fade; "
                                f"realtime_factor={release_realtime_factor:.3f}"
                            )
                            write_runtime_status(
                                "GROUNDING",
                                request_id=active_request.get("request_id"),
                                motion_id=active_request.get("motion_id"),
                                release_realtime_factor=release_realtime_factor,
                                reference_clock="lowstate_tick",
                                simulation_time_s=round(step_count * sim_step_s, 4),
                                bootstrap_support_mode=support_mode,
                                bootstrap_phase=bootstrap_phase,
                                elastic_support_scale=support_scale,
                                elastic_support_attitude_scale=(
                                    support_attitude_scale
                                ),
                            )
                    phase_samples_ms["critical_monitor"].append(
                        (time.perf_counter() - monitor_started) * 1000.0
                    )
                    phase_samples_ms["loop"].append(
                        (time.perf_counter() - loop_started) * 1000.0
                    )
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
                            "[pipeline-sim] completed "
                            f"{args_cli.unsupported_duration:.1f}s unsupported simulation-time hold"
                        )
                        break
                    if not args_cli.no_realtime_limit:
                        next_step_deadline += sim_step_s
                        delay = next_step_deadline - time.monotonic()
                        if delay > 0.0:
                            time.sleep(delay)
                        elif delay < -0.5:
                            # Do not attempt a large catch-up burst after a GUI
                            # pause or driver stall; resume pacing from now.
                            next_step_deadline = time.monotonic()
        if result == "STOPPED" and running:
            result = "GUI_CLOSED"

    except Exception as exc:
        result = "FAILED"
        unsafe_reason = str(exc)
        print(f"[baseline] FAILED: {exc}")
        raise
    finally:
        finalization_started = time.monotonic()
        video_finalization_started = finalization_started
        if video_thread is not None:
            try:
                if result == "COMPLETED" and capture_viewport_frame is not None:
                    final_frame = capture_viewport_frame()
                    video_queue.put(final_frame, timeout=5.0)
                    video_frames += 1
                video_queue.put(None, timeout=5.0)
                video_thread.join(timeout=30.0)
                if video_thread.is_alive():
                    raise RuntimeError("video encoder did not stop within 30 seconds")
                if video_encoder_errors:
                    raise RuntimeError(video_encoder_errors[0])
                print(
                    f"[pipeline-sim] video finalized: {video_path} "
                    f"({video_frames} frames at {args_cli.video_fps:g} fps)"
                )
            except Exception as exc:
                video_close_error = str(exc)
                print(f"[pipeline-sim] failed to finalize video: {exc}")
        elif video_writer is not None:
            try:
                video_writer.close()
            except Exception as exc:
                video_close_error = str(exc)
        video_finalization_s = time.monotonic() - video_finalization_started
        final_now = time.monotonic()
        if (
            args_cli.replay_trace
            and args_cli.replay_motion_id
            and result == "COMPLETED"
        ):
            exchange_root = (PROJECT_ROOT.parent / "motion_exchange").resolve()
            replay_artifact = (exchange_root / args_cli.replay_motion_id).resolve()
            if replay_artifact.parent != exchange_root or not replay_artifact.is_dir():
                raise RuntimeError(
                    f"unknown replay motion artifact: {args_cli.replay_motion_id}"
                )
            pipeline_package_root = PROJECT_ROOT.parent / "motion_pipeline"
            if str(pipeline_package_root) not in sys.path:
                sys.path.insert(0, str(pipeline_package_root))
            from motion_pipeline.latency import record_stage

            record_stage(
                replay_artifact / "timing.json",
                "offline_video_output",
                started_at=PROCESS_STARTED_UTC.isoformat(),
                duration_s=final_now - PROCESS_STARTED_MONOTONIC,
                excluded=bool(args_cli.replay_cold_bootstrap),
                details={
                    "non_blocking": True,
                    "cold_bootstrap": bool(args_cli.replay_cold_bootstrap),
                    "trace_path": str(args_cli.replay_trace),
                    "video_path": shared_video_path,
                    "video_frames": video_frames,
                    "video_fps": args_cli.video_fps,
                    "resolution": [args_cli.video_width, args_cli.video_height],
                    "quality_profile": args_cli.video_quality_profile,
                },
            )
        if playback_start_wall is not None and playback_end_wall is None:
            playback_end_wall = final_now
        reference_duration_s = (
            len(reference_joint_positions) / reference_hz
            if "reference_joint_positions" in locals()
            and reference_joint_positions
            and reference_hz > 0.0
            else None
        )
        reference_playback_wall_s = (
            reference_playback_end_wall - playback_start_wall
            if playback_start_wall is not None
            and reference_playback_end_wall is not None
            else None
        )
        playback_realtime_factor = (
            reference_duration_s / reference_playback_wall_s
            if reference_duration_s is not None
            and reference_playback_wall_s is not None
            and reference_playback_wall_s > 0.0
            else None
        )
        post_hold_wall_s = (
            max(0.0, playback_end_wall - reference_playback_end_wall)
            if playback_end_wall is not None
            and reference_playback_end_wall is not None
            else None
        )
        settling_wall_s = (
            release_wall - settle_start_wall
            if "release_wall" in locals()
            and release_wall is not None
            and settle_start_wall is not None
            else None
        )
        performance = {
            "isaac_runner_startup_s": (
                runner_ready_monotonic - PROCESS_STARTED_MONOTONIC
                if runner_ready_monotonic is not None
                else None
            ),
            "settling_wall_s": settling_wall_s,
            "settling_realtime_factor": (
                release_realtime_factor
                if "release_realtime_factor" in locals()
                else None
            ),
            "bootstrap_handoff": {
                "command_blend_duration_s": args_cli.command_blend_duration,
                "support_fade_min_handoff_progress": (
                    args_cli.support_fade_min_handoff_progress
                ),
                "bootstrap_waist_pitch_hold_gain_multiplier": (
                    args_cli.bootstrap_waist_pitch_hold_gain_multiplier
                ),
                "bootstrap_damping_multiplier": (
                    args_cli.bootstrap_damping_multiplier
                ),
                "support_attitude_fade_duration_s": (
                    args_cli.support_attitude_fade_duration
                ),
                "support_fade_duration_s": args_cli.support_fade_duration,
                "fade_velocity_violation_duration_s": (
                    args_cli.fade_velocity_violation_duration
                ),
                "supported_quiet_duration_s": args_cli.release_stable_duration,
                "unsupported_quiet_duration_s": (
                    args_cli.unsupported_stable_duration
                ),
                "max_settling_joint_velocity_rad_s": (
                    max_settling_joint_velocity
                    if "max_settling_joint_velocity" in locals()
                    else None
                ),
                "max_settling_joint_velocity_joint": (
                    max_settling_joint_velocity_joint
                    if "max_settling_joint_velocity_joint" in locals()
                    else None
                ),
                "max_settling_target_rate_rad_s": (
                    max_settling_target_rate
                    if "max_settling_target_rate" in locals()
                    else None
                ),
                "max_settling_torque_ratio": (
                    max_settling_torque_ratio
                    if "max_settling_torque_ratio" in locals()
                    else None
                ),
                "max_settling_torque_ratio_joint": (
                    max_settling_torque_ratio_joint
                    if "max_settling_torque_ratio_joint" in locals()
                    else None
                ),
            },
            "reference_duration_s": reference_duration_s,
            "reference_playback_wall_s": reference_playback_wall_s,
            "unsupported_playback_realtime_factor": playback_realtime_factor,
            "post_hold_wall_s": post_hold_wall_s,
            "phase_latency": {
                name: summarize_ms(values)
                for name, values in phase_samples_ms.items()
            },
            "dds": (
                g1_dds.performance_stats()
                if "g1_dds" in locals() and hasattr(g1_dds, "performance_stats")
                else None
            ),
            "lowcmd": (
                provider.performance_stats()
                if provider is not None and hasattr(provider, "performance_stats")
                else None
            ),
            "trace": trace_writer.stats() if trace_writer is not None else None,
            "video_finalization_s": video_finalization_s,
            "resources": {
                "process_cpu_time_s": time.process_time() - process_cpu_started,
                "process_wall_time_s": final_now - PROCESS_STARTED_MONOTONIC,
                "average_cpu_cores": (
                    (time.process_time() - process_cpu_started)
                    / max(final_now - PROCESS_STARTED_MONOTONIC, 1.0e-6)
                ),
                "cuda_device": str(env.device) if env is not None else None,
                "cuda_max_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(env.device))
                    if env is not None and torch.cuda.is_available()
                    else None
                ),
                "cuda_max_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(env.device))
                    if env is not None and torch.cuda.is_available()
                    else None
                ),
                "render_interval_physics_steps": (
                    int(env_cfg.sim.render_interval)
                    if "env_cfg" in locals()
                    else None
                ),
                "video_capture_enabled": bool(args_cli.video),
                "unused_contact_sensor_disabled": True,
                "qualified_zero_manager_step": True,
                "visual_state_output": args_cli.visual_state_output,
                "solver_position_iteration_count": (
                    ASSET_PROFILE.solver_position_iteration_count
                ),
                "solver_velocity_iteration_count": (
                    ASSET_PROFILE.solver_velocity_iteration_count
                ),
            },
        }
        report = {
            "schema_version": "1.0",
            "task": TASK_NAME,
            "asset_profile": ASSET_PROFILE.profile_id,
            "asset_profile_contract": ASSET_PROFILE.runtime_contract(),
            "asset_profile_qualified": ASSET_PROFILE.qualified,
            "diagnostic_only": DIAGNOSTIC_MODE,
            "elastic_target_height_m": (
                float(elastic_target_w[0, 2])
                if "elastic_target_w" in locals()
                else None
            ),
            "started_at": started_utc.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
            "reason": unsafe_reason,
            "bootstrap_support_released": not support_active,
            "bootstrap_support_mode": support_mode if "support_mode" in locals() else None,
            "bootstrap_phase": (
                bootstrap_phase if "bootstrap_phase" in locals() else None
            ),
            "elastic_support_scale": (
                support_scale if "support_scale" in locals() else None
            ),
            "elastic_support_attitude_scale": (
                support_attitude_scale
                if "support_attitude_scale" in locals()
                else None
            ),
            "command_seen": command_seen if "command_seen" in locals() else False,
            "request_id": (
                (active_request or {}).get("request_id")
                if "active_request" in locals()
                else None
            ),
            "motion_id": (
                (active_request or {}).get("motion_id")
                if "active_request" in locals()
                else None
            ),
            "shutdown_requested": (
                shutdown_requested if "shutdown_requested" in locals() else False
            ),
            "max_tracking_error_rad": max_tracking_error if "max_tracking_error" in locals() else None,
            "max_tracking_error_joint": max_tracking_joint if "max_tracking_joint" in locals() else None,
            "max_tracking_error_step": max_tracking_step if "max_tracking_step" in locals() else None,
            "max_command_position_error_rad": (
                max_command_position_error
                if "max_command_position_error" in locals()
                else None
            ),
            "max_command_position_error_joint": (
                max_command_position_joint
                if "max_command_position_joint" in locals()
                else None
            ),
            "max_command_position_error_step": (
                max_command_position_step
                if "max_command_position_step" in locals()
                else None
            ),
            "max_reference_root_height_error_m": (
                max_reference_root_height_error
                if "max_reference_root_height_error" in locals()
                else None
            ),
            "max_reference_root_height_error_frame": (
                max_reference_root_height_error_frame
                if "max_reference_root_height_error_frame" in locals()
                else None
            ),
            "playback_gate_step": (
                playback_gate_step if "playback_gate_step" in locals() else None
            ),
            "playback_start_step": (
                playback_start_step if "playback_start_step" in locals() else None
            ),
            "release_realtime_factor": (
                release_realtime_factor if "release_realtime_factor" in locals() else None
            ),
            "unsupported_playback_realtime_factor": playback_realtime_factor,
            "reference_clock": "lowstate_tick",
            "dds_domain": os.getenv("DDS_DOMAIN", "42"),
            "dds_interface": os.getenv("DDS_INTERFACE"),
            "lowstate_topic": os.getenv("SIM_LOWSTATE_TOPIC", "rt/socialnav_sim/g1/lowstate"),
            "lowcmd_topic": os.getenv("SIM_LOWCMD_TOPIC", "rt/socialnav_sim/g1/lowcmd"),
            "trace_path": str(trace_path),
            "video_path": str(video_path) if video_path is not None else None,
            "video_frames": video_frames,
            "video_fps": args_cli.video_fps if args_cli.video else None,
            "video_resolution": (
                [args_cli.video_width, args_cli.video_height]
                if args_cli.video
                else None
            ),
            "video_quality_profile": (
                args_cli.video_quality_profile if args_cli.video else None
            ),
            "video_camera_eye": (
                list(args_cli.video_camera_eye) if args_cli.video else None
            ),
            "video_camera_lookat": (
                list(args_cli.video_camera_lookat) if args_cli.video else None
            ),
            "video_error": video_close_error,
            "performance": performance,
            "samples": samples,
        }
        report_write_started = time.monotonic()
        log_path.write_text(json.dumps(report, indent=2) + "\n")
        performance["report_write_s"] = time.monotonic() - report_write_started
        performance["report_trace_finalization_s"] = (
            time.monotonic() - finalization_started
        )
        log_path.write_text(json.dumps(report, indent=2) + "\n")
        if not args_cli.replay_trace:
            try:
                write_runtime_status(
                    result,
                    request_id=(active_request or {}).get("request_id") if "active_request" in locals() else None,
                    motion_id=(active_request or {}).get("motion_id") if "active_request" in locals() else None,
                    reason=unsafe_reason,
                    report_path=shared_log_path,
                    trace_path=shared_trace_path,
                    release_realtime_factor=(
                        release_realtime_factor if "release_realtime_factor" in locals() else None
                    ),
                    max_tracking_error_rad=(
                        max_tracking_error if "max_tracking_error" in locals() else None
                    ),
                    max_command_position_error_rad=(
                        max_command_position_error
                        if "max_command_position_error" in locals()
                        else None
                    ),
                    max_reference_root_height_error_m=(
                        max_reference_root_height_error
                        if "max_reference_root_height_error" in locals()
                        else None
                    ),
                    video_path=shared_video_path,
                    video_frames=video_frames,
                    performance=performance,
                )
            except Exception as status_exc:
                print(f"[pipeline-sim] failed to write runtime status: {status_exc}")
        print(f"[pipeline-sim] result={result}; report={log_path}")
        if provider is not None:
            provider.cleanup()
        dds_manager.stop_all_communication()
        if env is not None:
            env.close()
        simulation_app.close()
    return 0 if result in {"COMPLETED", "STABLE", "SUPPORTED_STABLE", "STOPPED", "GUI_CLOSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
