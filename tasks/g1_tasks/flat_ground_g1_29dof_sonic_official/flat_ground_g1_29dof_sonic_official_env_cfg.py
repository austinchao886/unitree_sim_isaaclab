"""Flat-ground Isaac environment using SONIC's authoritative G1 model config."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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


def _load_official_sonic_g1() -> ArticulationCfg:
    """Load the exact robot configuration shipped with the pinned SONIC checkout.

    Importing the source file directly avoids copying actuator constants into this
    adapter.  The robot is spawned from the USD produced from that exact URDF by
    Isaac Lab.  This keeps GUI startup independent of the monolithic URDF importer
    UI extension while preserving SONIC's articulation and actuator properties.
    """

    project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
    sonic_root = (
        project_root.parent
        / "motion_pipeline/vendor/GR00T-WholeBodyControl/gear_sonic"
    )
    config_path = sonic_root / "envs/manager_env/robots/g1.py"
    urdf_path = sonic_root / "data/assets/robot_description/urdf/g1/main.urdf"
    usd_path = (
        project_root.parent
        / "motion_pipeline/.build/sonic_official_g1_usd/main.usd"
    )
    if not config_path.is_file() or not urdf_path.is_file() or not usd_path.is_file():
        raise FileNotFoundError(
            "Pinned SONIC robot files are missing: "
            f"config={config_path}, urdf={urdf_path}, converted_usd={usd_path}"
        )

    spec = importlib.util.spec_from_file_location("pinned_sonic_g1_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SONIC robot config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = module.G1_CYLINDER_MODEL_12_DEX_CFG
    return official.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=official.spawn.activate_contact_sensors,
            rigid_props=official.spawn.rigid_props,
            articulation_props=official.spawn.articulation_props,
        ),
    )


SONIC_OFFICIAL_G1_CFG = _load_official_sonic_g1()


@configclass
class FlatGroundSceneCfg(InteractiveSceneCfg):
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

    robot: ArticulationCfg = SONIC_OFFICIAL_G1_CFG

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    # SONIC LowCmd PD is converted to torque in SonicDDSActionProvider.  This
    # action term must therefore apply effort directly without another servo.
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
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
    # Writing only ArticulationCfg.init_state is not sufficient after the
    # simulation has started.  Apply it to PhysX on every env.reset(); without
    # this event the USD-authored zero pose receives a full neutral-pose torque
    # impulse on the first control step.
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class FlatGroundG129SonicOfficialEnvCfg(ManagerBasedRLEnvCfg):
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
        # Preserve official effort/velocity/armature values, but disable the
        # imported implicit drives because the DDS provider applies LowCmd PD.
        for actuator in self.scene.robot.actuators.values():
            actuator.stiffness = 0.0
            actuator.damping = 0.0

        # The policy/reference layer is 50 Hz, but the official implicit PD is
        # evaluated at every 200 Hz physics step.  Because this bridge computes
        # the full LowCmd PD torque explicitly, it must also update at 200 Hz.
        self.decimation = 1
        self.episode_length_s = 0.0
        self.sim.dt = 0.005
        # Keep GUI rendering at 50 Hz while running four torque updates per
        # rendered frame.
        self.sim.render_interval = 4
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "min"
