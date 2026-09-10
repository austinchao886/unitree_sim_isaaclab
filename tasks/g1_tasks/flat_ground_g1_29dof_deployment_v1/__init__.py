"""Registration for the isolated deployment-v1 G1 compatibility task."""

import gymnasium as gym

from . import flat_ground_g1_29dof_deployment_v1_env_cfg


gym.register(
    id="Isaac-Flat-G129-Deployment-V1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            flat_ground_g1_29dof_deployment_v1_env_cfg.
            FlatGroundG129DeploymentV1EnvCfg
        ),
    },
    disable_env_checker=True,
)
