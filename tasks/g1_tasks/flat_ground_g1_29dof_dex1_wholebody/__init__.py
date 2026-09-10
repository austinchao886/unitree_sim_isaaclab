# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import flat_ground_g1_29dof_dex1_wholebody_env_cfg


gym.register(
    id="Isaac-Flat-G129-Dex1-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            flat_ground_g1_29dof_dex1_wholebody_env_cfg.FlatGroundG129Dex1WholebodyEnvCfg
        ),
    },
    disable_env_checker=True,
)
