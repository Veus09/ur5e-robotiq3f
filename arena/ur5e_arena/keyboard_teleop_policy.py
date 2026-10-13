"""Keyboard teleoperation of the ``ur5e_3f_ik`` embodiment (relative differential IK).

Needs the GUI (``--viz kit``); click the viewport so it receives key presses.

    W / S   TCP +x / -x  (robot base frame)        Z / X   roll  + / -
    A / D   TCP +y / -y                            T / G   pitch + / -
    Q / E   TCP +z / -z                            C / V   yaw   + / -
    K       toggle gripper open / close            L       reset command

    python run_arena.py --viz kit \\
        --policy_type ur5e_arena.keyboard_teleop_policy.Ur5eKeyboardTeleopPolicy --num_steps 100000 \\
        pick_and_place_maple_table --embodiment ur5e_3f_ik \\
        --pick_up_object banana_ycb_robolab --destination_location plate_large_vomp_robolab
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

import ur5e_arena.embodiment  # noqa: F401  (registers the UR5e embodiments)


@dataclass
class Ur5eKeyboardTeleopPolicyCfg(PolicyCfg):
    """Configure keyboard teleoperation."""

    teleop_pos_sensitivity: float = 0.03
    """Commanded TCP translation per step while a key is held (before the action scale of 0.5), meters."""

    teleop_rot_sensitivity: float = 0.08
    """Commanded TCP rotation per step while a key is held (before the action scale of 0.5), radians."""


@register_policy
class Ur5eKeyboardTeleopPolicy(PolicyBase[Ur5eKeyboardTeleopPolicyCfg]):
    """Map Isaac Lab's SE(3) keyboard device onto the relative-IK action [dpos, drot, gripper]."""

    name = "ur5e_keyboard_teleop"

    def __init__(self, config: Ur5eKeyboardTeleopPolicyCfg):
        super().__init__(config)
        self._keyboard = None

    def _create_keyboard(self, device: str):
        from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg

        keyboard = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=self.config.teleop_pos_sensitivity,
                rot_sensitivity=self.config.teleop_rot_sensitivity,
                sim_device=device,
            )
        )
        print(keyboard)
        return keyboard

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._keyboard is not None:
            self._keyboard.reset()

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        base_env = env.unwrapped
        action_dim = env.action_space.shape[-1]
        assert action_dim == 7, f"Keyboard teleop expects the ur5e_3f_ik embodiment (7-D actions), got {action_dim}"
        if self._keyboard is None:
            import omni

            assert hasattr(omni, "appwindow"), "Keyboard teleop needs the GUI: add --viz kit to the command"
            self._keyboard = self._create_keyboard(base_env.device)
        command = self._keyboard.advance().to(base_env.device)
        return command.unsqueeze(0).repeat(base_env.num_envs, 1)
