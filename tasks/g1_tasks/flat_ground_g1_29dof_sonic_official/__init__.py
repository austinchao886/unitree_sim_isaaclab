"""Register the isolated official SONIC G1 baseline environment."""

import gymnasium as gym

from . import flat_ground_g1_29dof_sonic_official_env_cfg


gym.register(
    id="Isaac-Flat-G129-SONIC-Official",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            flat_ground_g1_29dof_sonic_official_env_cfg.FlatGroundG129SonicOfficialEnvCfg
        ),
    },
    disable_env_checker=True,
)
