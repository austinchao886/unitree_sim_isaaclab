# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
G1 robot DDS communication class
Handle the state publishing and command receiving of the G1 robot
"""

import numpy as np
import os
import socket
import threading
import time
from collections import deque
from typing import Any, Dict, Optional
# from dds.dds_base import BaseDDSNode, node_manager
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_, LowCmd_
from motion_pipeline.joystick import (
    MAX_FRAME_BYTES,
    decode_line,
    encode_unitree_remote,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.utils.crc import CRC


class G1RobotDDS(DDSObject):
    """G1 robot DDS communication class - singleton pattern
    
    Features:
    - Publish simulated state on a simulation-only DDS topic
    - Receive simulated commands on a simulation-only DDS topic
    """
    
    def __init__(self,node_name:str="g1_robot"):
        """Initialize the G1 robot DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
            
        super().__init__()
        self.node_name = node_name
        self.crc = CRC()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.torso_imu_state = unitree_hg_msg_dds__IMUState_()
        self.lowstate_topic = os.getenv(
            "SIM_LOWSTATE_TOPIC", "rt/socialnav_sim/g1/lowstate"
        )
        self.lowcmd_topic = os.getenv(
            "SIM_LOWCMD_TOPIC", "rt/socialnav_sim/g1/lowcmd"
        )
        self.secondary_imu_topic = os.getenv(
            "SIM_SECONDARY_IMU_TOPIC", "rt/socialnav_sim/g1/secondary_imu"
        )
        self.joystick_host = os.getenv("SIM_JOYSTICK_TCP_HOST", "127.0.0.1")
        self.joystick_port = int(os.getenv("SIM_JOYSTICK_TCP_PORT", "16042"))
        self.joystick_stale_after_s = float(
            os.getenv("SIM_JOYSTICK_STALE_AFTER_S", "0.2")
        )
        self._joystick_lock = threading.Lock()
        self._latest_joystick = None
        self._latest_joystick_received_at = 0.0
        self._joystick_sequence = -1
        self._joystick_stale_reported = False
        self._joystick_thread = threading.Thread(
            target=self._joystick_server_loop,
            name="simulation-joystick-server",
            daemon=True,
        )
        self._joystick_thread.start()
        self._reported_command_write_failure = False
        self._in_process_fast_path = os.getenv(
            "SIM_DDS_IN_PROCESS_FAST_PATH", "1"
        ) not in {"0", "false", "False"}
        # Loopback simulator DDS is already transport-protected and the action
        # provider performs finite/limit checks.  The Python Unitree CRC walk
        # costs a material fraction of a 5 ms physics step; retain an opt-in
        # diagnostic check without changing physical-topic behavior (which is
        # categorically rejected below).
        self.verify_fast_path_crc = os.getenv(
            "SIM_LOWCMD_VERIFY_CRC", "0"
        ) not in {"0", "false", "False"}
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._latest_state = None
        self._latest_command = None
        self._state_publish_times = deque(maxlen=4096)
        self._command_receive_times = deque(maxlen=4096)
        self._state_publish_count = 0
        self._command_receive_count = 0
        self._visual_state_writer = None
        visual_state_path = os.getenv("SIM_VISUAL_STATE_PATH")
        if visual_state_path:
            try:
                from motion_pipeline.visual_state import VisualStateWriter

                self._visual_state_writer = VisualStateWriter(visual_state_path)
                print(
                    f"[{self.node_name}] Read-only visual state output: "
                    f"{visual_state_path}"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to initialize visual state output {visual_state_path}: {exc}"
                ) from exc
        # The DDS publisher runs on wall time while Isaac advances on physics
        # time.  A slow GUI/physics step must not be republished hundreds of
        # times with the same tick: SONIC treats every LowState sample as work,
        # and duplicate reliable samples can fill the reader queue and starve
        # the Isaac Python thread.  Publish each simulator state at most once.
        self._last_published_sim_step = None
        if self.lowcmd_topic == "rt/lowcmd":
            raise RuntimeError(
                "Refusing to use the physical G1 rt/lowcmd topic in simulator mode"
            )
        self._initialized = True
        
        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_robot_state",  # read the state of the G1 robot from Isaac Lab
            # A full 29-DOF LowCmd contains five arrays (q, dq, tau, kp,
            # kd).  Once SONIC leaves its all-zero initialization command,
            # JSON serialization is commonly 4-7 KiB.  The old 3072-byte
            # segment silently retained the short initialization command and
            # made the action provider report a false DDS timeout.  Use a new
            # segment name so a stale 3072-byte POSIX segment from an older
            # simulator process cannot be reopened with the wrong capacity.
            output_shm_name="dds_robot_cmd_v2",  # output the command to Isaac Lab
            input_size=3072,
            output_size=16384  # full precision 29-DOF LowCmd JSON + headroom
        )
        
        print(f"[{self.node_name}] G1 robot DDS node initialized")
    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the G1 robot"""
        try:
            self.publisher = ChannelPublisher(self.lowstate_topic, LowState_)
            self.publisher.Init()
            self.torso_imu_publisher = ChannelPublisher(
                self.secondary_imu_topic, IMUState_
            )
            self.torso_imu_publisher.Init()
            print(f"[{self.node_name}] State publisher initialized ({self.lowstate_topic})")
            print(
                f"[{self.node_name}] Torso IMU publisher initialized "
                f"({self.secondary_imu_topic})"
            )
            return True
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] State publisher initialization failed: {e}")    
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the G1 robot"""
        try:
            print(f"[{self.node_name}] Create ChannelSubscriber...")
            self.subscriber = ChannelSubscriber(self.lowcmd_topic, LowCmd_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 32)
            print(f"[{self.node_name}] Command subscriber initialized ({self.lowcmd_topic})")
            return True
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Command subscriber initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def dds_publisher(self) -> Any:
        """Convert Isaac Lab state to DDS message and publish."""
        try:
            if self._in_process_fast_path:
                with self._state_lock:
                    data = self._latest_state
            else:
                data = self.input_shm.read_data()
            if data is None:
                return

            sim_step = data.get("sim_step")
            if sim_step is not None:
                sim_step = int(sim_step) & 0xFFFFFFFF
                if sim_step == self._last_published_sim_step:
                    return

            motor_state = self.low_state.motor_state
            imu_state = self.low_state.imu_state
            num_motors =len(motor_state)

            positions = data.get("joint_positions")
            velocities = data.get("joint_velocities")
            torques = data.get("joint_torques")

            if positions is not None and velocities is not None and torques is not None:
                q_array = np.asarray(positions, dtype=np.float32)
                dq_array = np.asarray(velocities, dtype=np.float32)
                tau_array = np.asarray(torques, dtype=np.float32)
                for i in range(len(q_array)):
                    motor = motor_state[i]
                    motor.q = q_array[i]
                    motor.dq = dq_array[i]
                    motor.tau_est = tau_array[i]

            imu = data.get("imu_data")
            if imu is not None and len(imu) >= 13:
                imu_array = np.asarray(imu, dtype=np.float32)

                # get_robot_imu_data() already returns Unitree/SONIC ordering:
                # [position xyz, quaternion wxyz, accel xyz, gyro xyz].
                imu_state.quaternion[:] = imu_array[3:7]

                imu_state.accelerometer[:] = imu_array[7:10]

                imu_state.gyroscope[:] = imu_array[10:13]

            # The simulator remains the sole LowState publisher. A dedicated,
            # loopback-only application topic supplies only the 40-byte remote
            # field so a Mac-connected PS4 controller cannot impersonate robot
            # state or publish on the physical LowState topic.
            with self._joystick_lock:
                command = self._latest_joystick
                command_age = time.monotonic() - self._latest_joystick_received_at
            if command is None or command_age > self.joystick_stale_after_s:
                self.low_state.wireless_remote[:] = encode_unitree_remote(None)
                if command is not None and not self._joystick_stale_reported:
                    print(
                        f"[{self.node_name}] Joystick stale after "
                        f"{command_age:.3f}s; forcing deadman release"
                    )
                    self._joystick_stale_reported = True
            else:
                self.low_state.wireless_remote[:] = encode_unitree_remote(command)
                self._joystick_stale_reported = False

            # In simulator mode the LowState tick is the authoritative physics
            # step, not a publisher-thread counter.  The DDS publisher may run
            # many times while Isaac is still rendering one physics step; using
            # that wall-clock publish rate makes SONIC advance a 50 Hz reference
            # faster than simulation time.  Keep the legacy counter as a
            # fallback for callers that do not yet provide a simulation step.
            if sim_step is None:
                self.low_state.tick += 1
            else:
                self.low_state.tick = sim_step
            self.low_state.crc = self.crc.Crc(self.low_state)
            self.publisher.Write(self.low_state)
            published_at = time.monotonic()
            self._state_publish_times.append(published_at)
            self._state_publish_count += 1

            torso_imu = data.get("torso_imu_data")
            if torso_imu is not None and len(torso_imu) >= 13:
                torso_array = np.asarray(torso_imu, dtype=np.float32)
                self.torso_imu_state.quaternion[:] = torso_array[3:7]
                self.torso_imu_state.accelerometer[:] = torso_array[7:10]
                self.torso_imu_state.gyroscope[:] = torso_array[10:13]
                self.torso_imu_publisher.Write(self.torso_imu_state)

            if sim_step is not None:
                self._last_published_sim_step = sim_step

        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error processing publish data: {e}")

    def _joystick_server_loop(self) -> None:
        """Accept normalized PS4 frames over an SSH-forwarded loopback socket."""
        if self.joystick_host not in {"127.0.0.1", "::1", "localhost"}:
            print(
                f"[{self.node_name}] Refusing non-loopback simulation joystick "
                f"host: {self.joystick_host}"
            )
            return
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.joystick_host, self.joystick_port))
                server.listen(1)
                print(
                    f"[{self.node_name}] Simulation joystick listening on "
                    f"{self.joystick_host}:{self.joystick_port}"
                )
                while True:
                    connection, address = server.accept()
                    print(f"[{self.node_name}] Joystick client connected: {address}")
                    last_sequence = -1
                    try:
                        with connection, connection.makefile("rb") as stream:
                            for line in stream:
                                if len(line) > MAX_FRAME_BYTES:
                                    raise ValueError("joystick frame exceeds size limit")
                                command = decode_line(line)
                                if command.sequence <= last_sequence:
                                    continue
                                last_sequence = command.sequence
                                with self._joystick_lock:
                                    self._joystick_sequence = command.sequence
                                    self._latest_joystick = command
                                    self._latest_joystick_received_at = time.monotonic()
                    except (ConnectionError, OSError, ValueError) as exc:
                        print(f"[{self.node_name}] Joystick client ended: {exc}")
                    finally:
                        with self._joystick_lock:
                            self._latest_joystick = None
                            self._latest_joystick_received_at = 0.0
                        print(
                            f"[{self.node_name}] Joystick disconnected; "
                            "forcing deadman release"
                        )
        except OSError as exc:
            print(f"[{self.node_name}] Simulation joystick listener failed: {exc}")

    
    def dds_subscriber(self, msg: LowCmd_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data: convert the DDS command to the Isaac Lab format
        
        Return data format:
        {
            "mode_pr": int,
            "mode_machine": int,
            "motor_cmd": {
                "positions": [29 joint position commands],
                "velocities": [29 joint velocity commands],
                "torques": [29 joint torque commands],
                "kp": [29 position gains],
                "kd": [29 speed gains]
            }
        }
        """
        try:
            received_at = time.monotonic()
            # SONIC's servo republishes LowCmd at roughly 500 Hz while Isaac
            # physics runs at 200 Hz.  In-process mode only needs the newest
            # sample: retain the DDS object and defer CRC/list conversion to
            # the physics thread.  This keeps the callback short and avoids
            # 145 Python float allocations fighting PhysX for the GIL.
            if self._in_process_fast_path:
                with self._command_lock:
                    self._latest_command = {
                        "received_monotonic": received_at,
                        "raw_lowcmd": msg,
                    }
                self._command_receive_times.append(received_at)
                self._command_receive_count += 1
                return self._latest_command

            # verify the CRC
            if self.crc.Crc(msg) != msg.crc:
                print(f"g1_robot_dds [{self.node_name}] Warning: CRC verification failed!")
                return {}
            
            # extract the command data
            num_cmd_motors = len(msg.motor_cmd)
            cmd_data = {
                "received_monotonic": received_at,
                "mode_pr": int(msg.mode_pr),
                "mode_machine": int(msg.mode_machine),
                "reserve": [int(value) for value in msg.reserve],
                "motor_cmd": {
                    "positions": [float(msg.motor_cmd[i].q) for i in range(num_cmd_motors)],
                    "velocities": [float(msg.motor_cmd[i].dq) for i in range(num_cmd_motors)],
                    "torques": [float(msg.motor_cmd[i].tau) for i in range(num_cmd_motors)],
                    "kp": [float(msg.motor_cmd[i].kp) for i in range(num_cmd_motors)],
                    "kd": [float(msg.motor_cmd[i].kd) for i in range(num_cmd_motors)]
                }
            }
            with self._command_lock:
                self._latest_command = cmd_data
            received_at = cmd_data["received_monotonic"]
            self._command_receive_times.append(received_at)
            self._command_receive_count += 1
            if self._in_process_fast_path:
                self._reported_command_write_failure = False
            elif not self.output_shm.write_data(cmd_data):
                if not self._reported_command_write_failure:
                    print(
                        f"g1_robot_dds [{self.node_name}] Error: LowCmd did not fit "
                        "in command shared memory; retaining the previous command"
                    )
                    self._reported_command_write_failure = True
            else:
                self._reported_command_write_failure = False
            
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error processing subscribe data: {e}")
            return {}
    
    def get_robot_command(self) -> Optional[Dict[str, Any]]:
        """Get the robot control command
        
        Returns:
            Dict: the robot control command, return None if there is no new command
        """
        if self._in_process_fast_path:
            with self._command_lock:
                return self._latest_command
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_robot_state(
        self,
        joint_positions,
        joint_velocities,
        joint_torques,
        imu_data,
        torso_imu_data=None,
        sim_step=None,
    ):
        """Write the robot state to the shared memory
        
        Args:
            joint_positions: the joint position list or torch.Tensor
            joint_velocities: the joint velocity list or torch.Tensor
            joint_torques: the joint torque list or torch.Tensor
            imu_data: the IMU data list or torch.Tensor
            torso_imu_data: independent torso IMU data for rt/secondary_imu
            sim_step: Isaac physics-step counter used to synchronize SONIC
        """
        if self.input_shm is None:
            return
        try:
            def transport_value(value):
                if self._in_process_fast_path:
                    return value
                return value.tolist() if hasattr(value, "tolist") else value

            state_data = {
                "joint_positions": transport_value(joint_positions),
                "joint_velocities": transport_value(joint_velocities),
                "joint_torques": transport_value(joint_torques),
                "imu_data": transport_value(imu_data),
                "torso_imu_data": transport_value(torso_imu_data),
                "sim_step": None if sim_step is None else int(sim_step),
            }
            if self._visual_state_writer is not None:
                # imu_data starts with the actual root xyz + quaternion wxyz.
                # All values here are already part of the single LowState CPU
                # transfer, so the viewer adds no extra GPU synchronization.
                self._visual_state_writer.write(
                    simulation_step=0 if sim_step is None else int(sim_step),
                    root_pose_wxyz=imu_data[:7],
                    joint_positions=joint_positions,
                    joint_velocities=joint_velocities,
                )
            if self._in_process_fast_path:
                with self._state_lock:
                    self._latest_state = state_data
            else:
                self.input_shm.write_data(state_data)
        except Exception as e:
            print(f"g1_robot_dds [{self.node_name}] Error writing robot state: {e}")

    def stop_communication(self):
        super().stop_communication()
        if self._visual_state_writer is not None:
            self._visual_state_writer.close()
            self._visual_state_writer = None

    def performance_stats(self) -> Dict[str, Any]:
        """Return bounded in-process DDS transport timing without blocking control."""

        def rate(values):
            if len(values) < 2:
                return 0.0
            elapsed = values[-1] - values[0]
            return 0.0 if elapsed <= 0.0 else (len(values) - 1) / elapsed

        def interval_percentile(values, percentile):
            if len(values) < 2:
                return None
            intervals = sorted(
                (right - left) * 1000.0
                for left, right in zip(values, list(values)[1:])
            )
            index = min(
                len(intervals) - 1,
                max(0, round((percentile / 100.0) * (len(intervals) - 1))),
            )
            return intervals[index]

        state_times = list(self._state_publish_times)
        command_times = list(self._command_receive_times)
        return {
            "in_process_fast_path": self._in_process_fast_path,
            "fast_path_crc_verified": self.verify_fast_path_crc,
            "lowstate_publish_count": self._state_publish_count,
            "lowstate_publish_hz": rate(state_times),
            "lowstate_interval_p95_ms": interval_percentile(state_times, 95.0),
            "lowcmd_receive_count": self._command_receive_count,
            "lowcmd_receive_hz": rate(command_times),
            "lowcmd_interval_p95_ms": interval_percentile(command_times, 95.0),
            "lowcmd_age_ms": (
                None
                if not command_times
                else max(0.0, (time.monotonic() - command_times[-1]) * 1000.0)
            ),
        }
