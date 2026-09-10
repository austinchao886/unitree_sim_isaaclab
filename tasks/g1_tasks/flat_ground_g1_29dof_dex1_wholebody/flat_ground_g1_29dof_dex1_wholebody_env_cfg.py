# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

"""Minimal flat-ground environment for keyboard-controlled G1 locomotion."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets


# SONIC's policy was trained around this exact neutral body pose.  The custom
# USD keeps its four Dex1 hand joints, but its 29 body joints enter the bridge
# in the same state as the qualified official asset.
SONIC_NEUTRAL_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "left_hand_Joint1_1": 0.024,
    "left_hand_Joint2_1": 0.024,
    "right_hand_Joint1_1": 0.024,
    "right_hand_Joint2_1": 0.024,
}


# These are motor/servo limits and reflected armatures, not policy gains. Most
# values come from the pinned official G1_CYLINDER_MODEL_12_DEX_CFG snapshot.
# The custom USD's 5020-family joints use a measured armature of 0.005: the
# official 0.003609725 made the fixed-root wrist step 13.8% too fast, while the
# USD-authored 0.01 made it 19.4% too slow.  This interpolated value is verified
# below by the same LowCmd response test; SONIC's kp/kd remain unchanged.
SONIC_MOTOR_PROPERTIES = {
    "left_hip_pitch_joint": (139.0, 20.0, 0.025101925),
    "left_hip_roll_joint": (139.0, 20.0, 0.025101925),
    "left_hip_yaw_joint": (88.0, 32.0, 0.010177520),
    "left_knee_joint": (139.0, 20.0, 0.025101925),
    "left_ankle_pitch_joint": (50.0, 37.0, 0.007219450),
    "left_ankle_roll_joint": (50.0, 37.0, 0.007219450),
    "right_hip_pitch_joint": (139.0, 20.0, 0.025101925),
    "right_hip_roll_joint": (139.0, 20.0, 0.025101925),
    "right_hip_yaw_joint": (88.0, 32.0, 0.010177520),
    "right_knee_joint": (139.0, 20.0, 0.025101925),
    "right_ankle_pitch_joint": (50.0, 37.0, 0.007219450),
    "right_ankle_roll_joint": (50.0, 37.0, 0.007219450),
    "waist_yaw_joint": (88.0, 32.0, 0.010177520),
    "waist_roll_joint": (50.0, 37.0, 0.007219450),
    "waist_pitch_joint": (50.0, 37.0, 0.007219450),
    "left_shoulder_pitch_joint": (25.0, 37.0, 0.005000000),
    "left_shoulder_roll_joint": (25.0, 37.0, 0.005000000),
    "left_shoulder_yaw_joint": (25.0, 37.0, 0.005000000),
    "left_elbow_joint": (25.0, 37.0, 0.005000000),
    "left_wrist_roll_joint": (25.0, 37.0, 0.005000000),
    "left_wrist_pitch_joint": (5.0, 22.0, 0.004250000),
    "left_wrist_yaw_joint": (5.0, 22.0, 0.004250000),
    "right_shoulder_pitch_joint": (25.0, 37.0, 0.005000000),
    "right_shoulder_roll_joint": (25.0, 37.0, 0.005000000),
    "right_shoulder_yaw_joint": (25.0, 37.0, 0.005000000),
    "right_elbow_joint": (25.0, 37.0, 0.005000000),
    "right_wrist_roll_joint": (25.0, 37.0, 0.005000000),
    "right_wrist_pitch_joint": (5.0, 22.0, 0.004250000),
    "right_wrist_yaw_joint": (5.0, 22.0, 0.004250000),
}

SONIC_ACTUATOR_JOINTS = {
    "legs": tuple(
        name
        for name in SONIC_MOTOR_PROPERTIES
        if any(token in name for token in ("hip", "knee", "waist"))
    ),
    "feet": tuple(name for name in SONIC_MOTOR_PROPERTIES if "ankle" in name),
    "shoulders": tuple(
        name
        for name in SONIC_MOTOR_PROPERTIES
        if "shoulder_pitch" in name or "shoulder_roll" in name
    ),
    "arms": tuple(
        name
        for name in SONIC_MOTOR_PROPERTIES
        if "shoulder_yaw" in name or "elbow" in name
    ),
    "wrist": tuple(name for name in SONIC_MOTOR_PROPERTIES if "wrist" in name),
}


def _sonic_compatible_dex1_cfg() -> ArticulationCfg:
    robot = G1RobotPresets.g1_29dof_dex1_wholebody(
        init_pos=(0.0, 0.0, 0.76),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )
    return robot.replace(
        spawn=robot.spawn.replace(
            articulation_props=robot.spawn.articulation_props.replace(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            )
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=SONIC_NEUTRAL_JOINT_POS,
            joint_vel={".*": 0.0},
        )
    )


@configclass
class FlatGroundSceneCfg(InteractiveSceneCfg):
    """A ground plane, a light, and the policy-compatible G1 articulation."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    robot: ArticulationCfg = _sonic_compatible_dex1_cfg()

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    # SONIC publishes a complete low-level PD command. The provider converts it
    # to torque, so this action term must not apply a second position servo.
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        # ActionManager resolves this in articulation-native order. The provider
        # maps Unitree's 29 motor torques into that full vector by joint name.
        joint_names=[".*"],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    pass


@configclass
class EventCfg:
    # Apply ArticulationCfg.init_state to PhysX on reset.  Without this event,
    # the USD-authored zero pose receives a large neutral-pose torque impulse
    # on the first SONIC servo step.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class FlatGroundG129Dex1WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: FlatGroundSceneCfg = FlatGroundSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        # Disable PhysX position drives for the 29 body joints. SONIC's provider
        # supplies the full PD torque; retaining these drives would apply PD twice.
        for name, actuator in self.scene.robot.actuators.items():
            if name != "hands":
                actuator.stiffness = 0.0
                actuator.damping = 0.0
                joints = SONIC_ACTUATOR_JOINTS[name]
                actuator.effort_limit_sim = {
                    joint: SONIC_MOTOR_PROPERTIES[joint][0] for joint in joints
                }
                actuator.velocity_limit_sim = {
                    joint: SONIC_MOTOR_PROPERTIES[joint][1] for joint in joints
                }
                actuator.armature = {
                    joint: SONIC_MOTOR_PROPERTIES[joint][2] for joint in joints
                }
        # SONIC reference/policy output is 50 Hz, but the explicit LowCmd PD
        # torque must be recomputed at the 200 Hz physics rate.
        self.decimation = 1
        self.episode_length_s = 0.0
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "min"
