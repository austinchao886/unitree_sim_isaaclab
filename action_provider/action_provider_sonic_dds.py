"""Physics-driven 29-DOF DDS action provider for GEAR-SONIC."""

import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from action_provider.action_base import ActionProvider
from dds.dds_master import dds_manager


# Unitree G1 motor order used by rt/lowcmd and rt/lowstate.
G1_MOTOR_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

SONIC_SIM_STARTUP_MARKER = 0x534F4E49

class SonicDDSActionProvider(ActionProvider):
    """Apply the complete SONIC low-level PD command as joint torques.

    This matches the official MuJoCo bridge:
    tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq).
    The task clamps the result through the articulation actuator effort limits.
    If DDS commands disappear, command zero torque rather than replaying stale
    targets indefinitely.
    """

    def __init__(self, env, args_cli):
        super().__init__("SonicDDSActionProvider")
        self.env = env
        self.robot_dds = dds_manager.get_object("g129")
        if self.robot_dds is None:
            raise RuntimeError("g129 DDS object is not registered")

        names = env.scene["robot"].data.joint_names
        profile = getattr(args_cli, "asset_profile_contract", None) or {}
        joint_map = profile.get("contract_to_asset_joint") or {
            name: name for name in G1_MOTOR_JOINTS
        }
        axis_signs = profile.get("joint_axis_sign") or {
            name: 1 for name in G1_MOTOR_JOINTS
        }
        effort_scales = profile.get("joint_effort_scale") or {
            name: 1.0 for name in G1_MOTOR_JOINTS
        }
        mapped_names = [joint_map[name] for name in G1_MOTOR_JOINTS]
        missing = [name for name in mapped_names if name not in names]
        if missing:
            raise ValueError(f"Isaac G1 is missing SONIC joints: {missing}")
        self._target_indices = torch.tensor(
            [names.index(name) for name in mapped_names],
            dtype=torch.long,
            device=env.device,
        )
        self._axis_signs = torch.tensor(
            [axis_signs[name] for name in G1_MOTOR_JOINTS],
            dtype=torch.float32,
            device=env.device,
        )
        self._effort_scales = torch.tensor(
            [effort_scales[name] for name in G1_MOTOR_JOINTS],
            dtype=torch.float32,
            device=env.device,
        )
        self._torque = torch.zeros_like(env.scene["robot"].data.default_joint_pos[0])
        self._default_joint_pos = env.scene["robot"].data.default_joint_pos[0].clone()
        self._default_target_pos = self._default_joint_pos.index_select(
            0, self._target_indices
        )
        # Equivalent to the official simulator's gantry/bootstrap phase: hold
        # the spawn pose until SONIC publishes its first complete LowCmd.
        # Match gear_sonic_deploy/policy_parameters.hpp exactly.  These gains
        # are evaluated at the 200 Hz physics-servo rate by the official
        # baseline task; the 50 Hz reference/policy rate is a separate layer.
        natural_freq = 10.0 * 2.0 * torch.pi
        damping_ratio = 2.0
        armature_5020 = 0.003609725
        armature_7520_14 = 0.010177520
        armature_7520_22 = 0.025101925
        armature_4010 = 0.00425

        def gains(armature, multiplier=1.0):
            kp = multiplier * armature * natural_freq**2
            kd = multiplier * 2.0 * damping_ratio * armature * natural_freq
            return kp, kd

        hold_kp = []
        hold_kd = []
        for name in G1_MOTOR_JOINTS:
            if "ankle" in name:
                kp, kd = gains(armature_5020, 2.0)
            elif "hip_pitch" in name or "hip_roll" in name or "knee" in name:
                kp, kd = gains(armature_7520_22)
            elif "hip_yaw" in name or name == "waist_yaw_joint":
                kp, kd = gains(armature_7520_14)
            elif name in {"waist_roll_joint", "waist_pitch_joint"}:
                kp, kd = gains(armature_5020, 2.0)
            elif "wrist_pitch" in name or "wrist_yaw" in name:
                kp, kd = gains(armature_4010)
            else:
                kp, kd = gains(armature_5020)
            hold_kp.append(kp)
            hold_kd.append(kd)
        # SONIC's physical deployment starts from a robot that is already held
        # upright by the platform controller.  An explicit-effort Isaac bridge
        # has no gravity feed-forward during that pre-CONTROL interval, so the
        # official PD gains alone let the torso statically sag (about 0.34 rad
        # at waist_pitch on the qualified asset).  Keep the same damping ratio
        # while strengthening only that gravity-loaded waist axis.  Raising
        # all 29 gains makes the low-inertia wrists numerically unstable.  The
        # authoritative LowCmd gains are used unchanged after handoff starts.
        self._bootstrap_waist_pitch_hold_gain_multiplier = max(
            1.0,
            float(
                getattr(
                    args_cli,
                    "bootstrap_waist_pitch_hold_gain_multiplier",
                    4.0,
                )
            ),
        )
        hold_gain_scale = torch.tensor(
            [
                self._bootstrap_waist_pitch_hold_gain_multiplier
                if name == "waist_pitch_joint"
                else 1.0
                for name in G1_MOTOR_JOINTS
            ],
            dtype=torch.float32,
            device=env.device,
        )
        self._hold_kp = torch.tensor(
            hold_kp, dtype=torch.float32, device=env.device
        ).mul_(hold_gain_scale)
        self._hold_kd = torch.tensor(
            hold_kd, dtype=torch.float32, device=env.device
        ).mul_(torch.sqrt(hold_gain_scale))
        # SharedMemoryManager re-opens named segments left by an earlier Isaac
        # process.  Never replay a LowCmd cached by a previous simulator run;
        # only commands received after this provider instance was created are
        # eligible.  SONIC publishes continuously, so a live controller will
        # provide a fresh sample on its next cycle.
        self._accept_commands_after = time.monotonic()
        self._last_fresh_cmd = 0.0
        self._last_q_des: torch.Tensor | None = None
        self._last_dq_des: torch.Tensor | None = None
        self._last_tau_ff: torch.Tensor | None = None
        self._last_kp: torch.Tensor | None = None
        self._last_kd: torch.Tensor | None = None
        self._actual_q: torch.Tensor | None = None
        self._actual_dq: torch.Tensor | None = None
        self._stale_timeout_s = float(getattr(args_cli, "sonic_command_timeout", 0.25))
        self._reported_stale = False
        self._reported_pre_session_cmd = False
        self._fresh_command_times: deque[float] = deque(maxlen=4096)
        self._lowcmd_age_samples_ms: deque[float] = deque(maxlen=20000)
        self._lowcmd_host = np.empty((5, 29), dtype=np.float32)
        self._lowcmd_host_tensor = torch.from_numpy(self._lowcmd_host)
        self._lowcmd_device = torch.empty(
            (5, 29), dtype=torch.float32, device=env.device
        )
        (
            self._q_des_device,
            self._dq_des_device,
            self._tau_ff_device,
            self._kp_device,
            self._kd_device,
        ) = self._lowcmd_device.unbind(0)
        # Preallocate the 200 Hz PD hot path. Reusing these tensors prevents
        # CUDA allocator churn from becoming physics-step tail latency.
        self._q_asset = torch.empty(29, dtype=torch.float32, device=env.device)
        self._dq_asset = torch.empty(29, dtype=torch.float32, device=env.device)
        self._actual_q = torch.empty(29, dtype=torch.float32, device=env.device)
        self._actual_dq = torch.empty(29, dtype=torch.float32, device=env.device)
        self._work_q = torch.empty(29, dtype=torch.float32, device=env.device)
        self._work_dq = torch.empty(29, dtype=torch.float32, device=env.device)
        self._motor_torque_contract = torch.empty(
            29, dtype=torch.float32, device=env.device
        )
        self._motor_torque_asset = torch.empty(
            29, dtype=torch.float32, device=env.device
        )
        self._applied_motor_torque_asset = torch.zeros(
            29, dtype=torch.float32, device=env.device
        )
        self._previous_output_motor_torque = torch.zeros(
            29, dtype=torch.float32, device=env.device
        )
        self._handoff_origin_motor_torque = torch.zeros(
            29, dtype=torch.float32, device=env.device
        )
        self._sim_step_s = float(env.physics_dt)
        self._handoff_duration_s = max(
            0.0, float(getattr(args_cli, "command_blend_duration", 0.2))
        )
        self._handoff_step = 0
        self._handoff_active = False
        self._handoff_armed = False
        # Cache SONIC's INIT LowCmd for observations/telemetry, but do not let
        # it replace the neutral bootstrap hold.  begin_control_handoff() is
        # called only after the supervisor has entered CONTROL, ensuring the
        # effort blend is spent on the first policy action rather than on the
        # ten-frame history warm-up.
        self._control_handoff_started = False
        self._handoff_progress = 1.0
        self._handoff_trigger_delta_rad = max(
            0.0,
            float(getattr(args_cli, "command_blend_trigger_delta", 0.01)),
        )
        self._bootstrap_target_slew_rate_rad_s = max(
            0.0,
            float(getattr(args_cli, "bootstrap_target_slew_rate", 0.0)),
        )
        self._bootstrap_target_shaping_active = False
        self._shaped_q_des = torch.empty(
            29, dtype=torch.float32, device=env.device
        )
        self._shaping_delta = torch.empty(
            29, dtype=torch.float32, device=env.device
        )
        self._bootstrap_damping_multiplier = max(
            1.0,
            float(getattr(args_cli, "bootstrap_damping_multiplier", 1.0)),
        )
        self._bootstrap_damping_active = False
        self._previous_q_des_host: np.ndarray | None = None
        self._latest_target_rate_rad_s = 0.0
        self._sonic_startup_marker_seen = False
        self._sonic_startup_ready = True

    def _update_sonic_startup_marker(self, reserve) -> None:
        if reserve is None or len(reserve) < 4:
            return
        if int(reserve[0]) != SONIC_SIM_STARTUP_MARKER:
            return
        self._sonic_startup_marker_seen = True
        self._sonic_startup_ready = int(reserve[1]) == 1

    def get_action(self, env) -> Optional[torch.Tensor]:
        self._previous_output_motor_torque.copy_(
            self._applied_motor_torque_asset
        )
        first_fresh_command = False
        cmd = self.robot_dds.get_robot_command()
        now = time.monotonic()
        if cmd and ("motor_cmd" in cmd or "raw_lowcmd" in cmd):
            received_at = float(cmd.get("received_monotonic", 0.0))
            if received_at <= self._accept_commands_after:
                if received_at and not self._reported_pre_session_cmd:
                    print(
                        "[SonicDDSActionProvider] ignoring LowCmd cached by a previous simulator session"
                    )
                    self._reported_pre_session_cmd = True
            elif received_at > self._last_fresh_cmd:
                first_fresh_command = self._last_fresh_cmd == 0.0
                raw_lowcmd = cmd.get("raw_lowcmd")
                if raw_lowcmd is not None:
                    if (
                        self.robot_dds.verify_fast_path_crc
                        and self.robot_dds.crc.Crc(raw_lowcmd) != raw_lowcmd.crc
                    ):
                        return self._torque.unsqueeze(0)
                    if len(raw_lowcmd.motor_cmd) < 29:
                        return self._torque.unsqueeze(0)
                    self._update_sonic_startup_marker(raw_lowcmd.reserve)
                    for index in range(29):
                        motor = raw_lowcmd.motor_cmd[index]
                        self._lowcmd_host[0, index] = motor.q
                        self._lowcmd_host[1, index] = motor.dq
                        self._lowcmd_host[2, index] = motor.tau
                        self._lowcmd_host[3, index] = motor.kp
                        self._lowcmd_host[4, index] = motor.kd
                else:
                    self._update_sonic_startup_marker(cmd.get("reserve"))
                    motor_cmd = cmd["motor_cmd"]
                    fields = (
                        motor_cmd.get("positions", []),
                        motor_cmd.get("velocities", []),
                        motor_cmd.get("torques", []),
                        motor_cmd.get("kp", []),
                        motor_cmd.get("kd", []),
                    )
                    if not all(len(field) >= 29 for field in fields):
                        return self._torque.unsqueeze(0)
                    self._lowcmd_host[:, :] = np.asarray(fields, dtype=np.float32)[:, :29]

                # Validate on the host before the one H2D transfer.  Calling
                # bool(torch.isfinite(...)) here forces an unnecessary CUDA
                # synchronization for every LowCmd update.
                if np.isfinite(self._lowcmd_host).all():
                    target_delta_rad = 0.0
                    if self._previous_q_des_host is not None:
                        target_delta_rad = float(
                            np.max(
                                np.abs(
                                    self._lowcmd_host[0]
                                    - self._previous_q_des_host
                                )
                            )
                        )
                        command_dt_s = received_at - self._last_fresh_cmd
                        if command_dt_s > 0.0:
                            self._latest_target_rate_rad_s = float(
                                target_delta_rad / command_dt_s
                            )
                    else:
                        self._latest_target_rate_rad_s = 0.0
                    if (
                        self._handoff_armed
                        and target_delta_rad >= self._handoff_trigger_delta_rad
                    ):
                        self._handoff_origin_motor_torque.copy_(
                            self._previous_output_motor_torque
                        )
                        self._handoff_step = 0
                        self._handoff_progress = (
                            0.0 if self._handoff_duration_s > 0.0 else 1.0
                        )
                        self._handoff_active = self._handoff_duration_s > 0.0
                        self._handoff_armed = False
                        print(
                            "[SonicDDSActionProvider] policy target transition "
                            f"detected ({target_delta_rad:.4f}rad); starting "
                            f"{self._handoff_duration_s:.3f}s effort handoff"
                        )
                    if self._previous_q_des_host is None:
                        self._previous_q_des_host = np.empty(29, dtype=np.float32)
                    np.copyto(self._previous_q_des_host, self._lowcmd_host[0])
                    self._lowcmd_device.copy_(self._lowcmd_host_tensor)
                    self._last_fresh_cmd = received_at
                    self._fresh_command_times.append(received_at)
                    self._last_q_des = self._q_des_device
                    self._last_dq_des = self._dq_des_device
                    self._last_tau_ff = self._tau_ff_device
                    self._last_kp = self._kp_device
                    self._last_kd = self._kd_device
                    self._reported_stale = False

        command_age_s = (
            now - self._last_fresh_cmd if self._last_fresh_cmd else float("inf")
        )
        if self._last_fresh_cmd:
            self._lowcmd_age_samples_ms.append(max(0.0, command_age_s) * 1000.0)
        command_stale = bool(
            self._last_fresh_cmd and command_age_s > self._stale_timeout_s
        )
        bootstrap_hold = bool(
            not self._last_fresh_cmd
            or first_fresh_command
            or self._handoff_armed
            or not self._control_handoff_started
        )
        robot = env.scene["robot"].data
        torch.index_select(
            robot.joint_pos[0], 0, self._target_indices, out=self._q_asset
        )
        torch.index_select(
            robot.joint_vel[0], 0, self._target_indices, out=self._dq_asset
        )
        torch.mul(self._axis_signs, self._q_asset, out=self._actual_q)
        torch.mul(self._axis_signs, self._dq_asset, out=self._actual_dq)
        # The 200 Hz safety monitor consumes the same current state.  Cache it
        # so the runner does not launch two more index_select kernels per tick.
        q = self._actual_q
        dq = self._actual_dq

        if command_stale:
            self._applied_motor_torque_asset.zero_()
            self._torque.zero_()
            if not self._reported_stale:
                print("[SonicDDSActionProvider] lowcmd timeout; commanding zero torque")
                self._reported_stale = True
        elif bootstrap_hold:
            # Bootstrap hold before the first live SONIC command.
            torch.sub(self._default_target_pos, self._q_asset, out=self._work_q)
            torch.mul(self._hold_kp, self._work_q, out=self._motor_torque_asset)
            torch.mul(self._hold_kd, self._dq_asset, out=self._work_dq)
            self._motor_torque_asset.sub_(self._work_dq)
            self._applied_motor_torque_asset.copy_(self._motor_torque_asset)
            self._torque.zero_()
            self._torque.index_copy_(
                0, self._target_indices, self._applied_motor_torque_asset
            )
        else:
            # Unitree LowCmd contains a PD target, not a torque sample.  Apply
            # the cached target against the latest simulated state on every
            # 200 Hz physics step even when the policy/DDS target is 50 Hz.
            effective_q_des = self._last_q_des
            if self._bootstrap_target_shaping_active:
                torch.sub(
                    self._last_q_des,
                    self._shaped_q_des,
                    out=self._shaping_delta,
                )
                max_delta = (
                    self._bootstrap_target_slew_rate_rad_s
                    * self._sim_step_s
                )
                self._shaping_delta.clamp_(min=-max_delta, max=max_delta)
                self._shaped_q_des.add_(self._shaping_delta)
                effective_q_des = self._shaped_q_des
            torch.sub(effective_q_des, q, out=self._work_q)
            torch.mul(self._last_kp, self._work_q, out=self._work_q)
            torch.sub(self._last_dq_des, dq, out=self._work_dq)
            torch.mul(self._last_kd, self._work_dq, out=self._work_dq)
            if self._bootstrap_damping_active:
                self._work_dq.mul_(self._bootstrap_damping_multiplier)
            torch.add(
                self._last_tau_ff,
                self._work_q,
                out=self._motor_torque_contract,
            )
            self._motor_torque_contract.add_(self._work_dq)
            torch.mul(
                self._axis_signs,
                self._motor_torque_contract,
                out=self._motor_torque_asset,
            )
            self._motor_torque_asset.mul_(self._effort_scales)
            if self._handoff_active:
                self._handoff_step += 1
                if self._handoff_duration_s <= 0.0:
                    self._handoff_progress = 1.0
                else:
                    self._handoff_progress = min(
                        1.0,
                        self._handoff_step
                        * self._sim_step_s
                        / self._handoff_duration_s,
                    )
                torch.lerp(
                    self._handoff_origin_motor_torque,
                    self._motor_torque_asset,
                    self._handoff_progress,
                    out=self._applied_motor_torque_asset,
                )
                if self._handoff_progress >= 1.0:
                    self._handoff_active = False
            else:
                self._applied_motor_torque_asset.copy_(self._motor_torque_asset)
            self._torque.zero_()
            self._torque.index_copy_(
                0, self._target_indices, self._applied_motor_torque_asset
            )
        return self._torque.unsqueeze(0)

    def begin_control_handoff(self) -> None:
        """Blend from the last applied effort into the live SONIC controller.

        The SETTLING handshake calls this exactly once.  Starting from the
        previous physics-step output prevents the first CONTROL target from
        injecting an effort discontinuity even when it was received in the
        same loop as the handshake.
        """

        self._control_handoff_started = True
        self._handoff_step = 0
        self._handoff_origin_motor_torque.copy_(
            self._applied_motor_torque_asset
        )
        self._handoff_progress = 0.0 if self._handoff_duration_s > 0.0 else 1.0
        self._handoff_active = False
        self._handoff_armed = self._handoff_duration_s > 0.0
        self._bootstrap_damping_active = True
        self._bootstrap_target_shaping_active = (
            self._bootstrap_target_slew_rate_rad_s > 0.0
        )
        if self._bootstrap_target_shaping_active:
            self._shaped_q_des.copy_(self._actual_q)
        if self._handoff_armed:
            print(
                "[SonicDDSActionProvider] effort handoff armed; waiting for "
                f"a {self._handoff_trigger_delta_rad:.4f}rad policy-target transition"
            )

    def end_control_handoff(self) -> None:
        """Restore authoritative SONIC damping after unsupported quiet gating."""

        self._bootstrap_damping_active = False
        self._handoff_armed = False
        self._bootstrap_target_shaping_active = False

    @property
    def has_fresh_command(self) -> bool:
        """Whether this simulator session has received a valid live LowCmd."""
        return self._last_fresh_cmd > self._accept_commands_after

    @property
    def command_age_s(self) -> float:
        """Wall-clock age of the last valid command, or infinity before one arrives."""
        if not self.has_fresh_command:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_fresh_cmd)

    @property
    def command_is_stale(self) -> bool:
        return self.has_fresh_command and self.command_age_s > self._stale_timeout_s

    @property
    def target_indices(self) -> torch.Tensor:
        return self._target_indices

    @property
    def axis_signs(self) -> torch.Tensor:
        return self._axis_signs

    def actual_joint_positions(self) -> torch.Tensor:
        """Current position transformed from the asset into Unitree semantics."""
        if self._actual_q is not None:
            return self._actual_q
        values = self.env.scene["robot"].data.joint_pos[0].index_select(
            0, self._target_indices
        )
        return self._axis_signs * values

    def actual_joint_velocities(self) -> torch.Tensor:
        """Current velocity transformed from the asset into Unitree semantics."""
        if self._actual_dq is not None:
            return self._actual_dq
        values = self.env.scene["robot"].data.joint_vel[0].index_select(
            0, self._target_indices
        )
        return self._axis_signs * values

    @property
    def desired_joint_positions(self) -> torch.Tensor | None:
        return self._last_q_des

    @property
    def applied_desired_joint_positions(self) -> torch.Tensor | None:
        if self._last_q_des is None:
            return None
        if self._bootstrap_target_shaping_active:
            return self._shaped_q_des
        return self._last_q_des

    @property
    def applied_motor_torques(self) -> torch.Tensor:
        """Actual blended effort in Unitree motor order (axis magnitude-safe)."""

        return self._applied_motor_torque_asset

    @property
    def handoff_progress(self) -> float:
        return self._handoff_progress

    @property
    def latest_target_rate_rad_s(self) -> float:
        return self._latest_target_rate_rad_s

    @property
    def bootstrap_damping_multiplier(self) -> float:
        return (
            self._bootstrap_damping_multiplier
            if self._bootstrap_damping_active
            else 1.0
        )

    @property
    def sonic_startup_ready(self) -> bool:
        """Whether SONIC reports that its simulation-only action slew caught up."""

        return (
            not self._sonic_startup_marker_seen
            or self._sonic_startup_ready
        )

    def performance_stats(self) -> dict[str, float | int | None]:
        """Return bounded LowCmd receive-rate and age metrics for reports."""

        timestamps = list(self._fresh_command_times)
        intervals_ms = [
            (current - previous) * 1000.0
            for previous, current in zip(timestamps, timestamps[1:])
            if current >= previous
        ]

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
            return ordered[index]

        span_s = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        return {
            "accepted_lowcmd_count": len(timestamps),
            "lowcmd_receive_rate_hz": (
                (len(timestamps) - 1) / span_s if span_s > 0.0 else None
            ),
            "lowcmd_interval_p50_ms": percentile(intervals_ms, 0.50),
            "lowcmd_interval_p95_ms": percentile(intervals_ms, 0.95),
            "lowcmd_age_p95_ms": percentile(
                list(self._lowcmd_age_samples_ms), 0.95
            ),
            "lowcmd_age_s": self.command_age_s if self.has_fresh_command else None,
            "control_handoff_duration_s": self._handoff_duration_s,
            "control_handoff_progress": self._handoff_progress,
            "control_handoff_armed": self._handoff_armed,
            "control_handoff_trigger_delta_rad": self._handoff_trigger_delta_rad,
            "bootstrap_waist_pitch_hold_gain_multiplier": (
                self._bootstrap_waist_pitch_hold_gain_multiplier
            ),
            "bootstrap_target_slew_rate_rad_s": (
                self._bootstrap_target_slew_rate_rad_s
            ),
            "latest_target_rate_rad_s": self._latest_target_rate_rad_s,
            "bootstrap_damping_multiplier": self.bootstrap_damping_multiplier,
            "sonic_startup_marker_seen": self._sonic_startup_marker_seen,
            "sonic_startup_ready": self.sonic_startup_ready,
        }

    def cleanup(self):
        # DDS lifecycle is owned by dds_manager.
        pass
