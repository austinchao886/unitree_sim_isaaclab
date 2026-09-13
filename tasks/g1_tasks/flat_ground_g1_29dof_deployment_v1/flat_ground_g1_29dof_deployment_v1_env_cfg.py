"""SONIC compatibility environment for the deployment-v1 G1 USD.

The task deliberately inherits the qualified bridge semantics while keeping a
separate asset path and Gym id.  It may be used for read-only inspection and
fixed-root probes while its asset profile is pending calibration.
"""

import os
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tasks.common_config import G1RobotPresets
from tasks.g1_tasks.flat_ground_g1_29dof_dex1_wholebody.flat_ground_g1_29dof_dex1_wholebody_env_cfg import (
    FlatGroundG129Dex1WholebodyEnvCfg,
    FlatGroundSceneCfg,
    SONIC_NEUTRAL_JOINT_POS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_USD = (
    PROJECT_ROOT
    / "assets/robots/g1-29dof_wholebody_deployment-v1"
    / "g1_29dof_with_inspire_rev_1_0.usd"
)

INSPIRE_HAND_JOINTS = (
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
)


def _solver_iterations() -> tuple[int, int]:
    """Resolve bounded A/B knobs; production defaults remain explicit."""

    position = int(os.getenv("G1_DEPLOYMENT_SOLVER_POSITION_ITERS", "8"))
    velocity = int(os.getenv("G1_DEPLOYMENT_SOLVER_VELOCITY_ITERS", "1"))
    if not 4 <= position <= 8:
        raise ValueError("G1_DEPLOYMENT_SOLVER_POSITION_ITERS must be within [4, 8]")
    if not 1 <= velocity <= 4:
        raise ValueError("G1_DEPLOYMENT_SOLVER_VELOCITY_ITERS must be within [1, 4]")
    return position, velocity

def _sonic_compatible_deployment_cfg() -> ArticulationCfg:
    if not DEPLOYMENT_USD.is_file():
        raise FileNotFoundError(f"deployment-v1 USD is missing: {DEPLOYMENT_USD}")

    solver_position_iterations, solver_velocity_iterations = _solver_iterations()
    robot = G1RobotPresets.g1_29dof_inspire_wholebody(
        init_pos=(0.0, 0.0, 0.76),
        init_rot=(1.0, 0.0, 0.0, 0.0),
    )
    body_neutral = {
        name: value
        for name, value in SONIC_NEUTRAL_JOINT_POS.items()
        if name not in {
            "left_hand_Joint1_1",
            "left_hand_Joint2_1",
            "right_hand_Joint1_1",
            "right_hand_Joint2_1",
        }
    }
    neutral = {**body_neutral, **{name: 0.0 for name in INSPIRE_HAND_JOINTS}}
    return robot.replace(
        spawn=robot.spawn.replace(
            usd_path=str(DEPLOYMENT_USD),
            articulation_props=robot.spawn.articulation_props.replace(
                # Preserve the deployment asset's authored setting.  Enabling
                # self-collision on the 24 high-stiffness Inspire finger links
                # injects a persistent disturbance into the wrist chain during
                # neutral bootstrap, before SONIC has taken control.
                enabled_self_collisions=False,
                solver_position_iteration_count=solver_position_iterations,
                solver_velocity_iteration_count=solver_velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=neutral,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class DeploymentV1SceneCfg(FlatGroundSceneCfg):
    """Qualified flat-ground scene with the deployment-v1 articulation."""

    robot: ArticulationCfg = _sonic_compatible_deployment_cfg()


@configclass
class FlatGroundG129DeploymentV1EnvCfg(FlatGroundG129Dex1WholebodyEnvCfg):
    """Pending-calibration deployment asset using the SONIC LowCmd contract."""

    scene: DeploymentV1SceneCfg = DeploymentV1SceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )
