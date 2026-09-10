#!/usr/bin/env python3
"""Run the Unitree G1 locomotion policy on a minimal flat-ground scene."""

import argparse
import os
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import tasks  # noqa: F401  Registers the Unitree environments.
from action_provider.action_provider_wh_dds import DDSRLActionProvider
from dds.commands_dds import RunCommandDDS
from dds.dds_master import dds_manager
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


TASK_NAME = "Isaac-Flat-G129-Dex1-Wholebody"


def main():
    env = None
    provider = None
    running = True

    def request_stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped
        env.sim.reset()
        env.reset()

        run_command_dds = RunCommandDDS()
        dds_manager.register_object("run_command", run_command_dds)
        dds_manager.start_subscribing(["run_command"])

        ## Locomotion Policy
        provider_args = argparse.Namespace(
            robot_type="g129",
            enable_dex1_dds=False,
            enable_dex3_dds=False,
            enable_inspire_dds=False,
            enable_wholebody_dds=True,
            model_path="assets/model/policy.onnx",
        )
        provider = DDSRLActionProvider(env, provider_args)

        print("\nFlat-ground G1 locomotion is ready.")
        print("In another terminal, run: python send_commands_keyboard.py")
        print("Press Ctrl+C here to stop the simulation.\n")

        while running and simulation_app.is_running():
            provider.get_action(env)
    finally:
        if provider is not None:
            provider.cleanup()
        dds_manager.stop_all_communication()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
