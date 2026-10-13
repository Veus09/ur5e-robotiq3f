"""Scripted pick-and-place policy for the ``ur5e_3f_ik_abs`` embodiment.

Reads the pick object and destination poses from the simulator every step, plans TCP waypoints
above / at the object and the destination, and sends absolute IK targets (robot base frame).
Works with any Arena pick-and-place workspace, e.g. ``pick_and_place_maple_table``:

    python run_arena.py --viz kit \\
        --policy_type ur5e_arena.scripted_pick_place_policy.Ur5eScriptedPickPlacePolicy \\
        --num_episodes 3 \\
        pick_and_place_maple_table --embodiment ur5e_3f_ik_abs \\
        --pick_up_object apple_01_objaverse_robolab --destination_location plate_large_vomp_robolab

The object names are read from ``--pick_up_object`` / ``--destination_location`` on the command line,
or from ``--ik_pick_asset`` / ``--ik_place_asset`` when an environment names them differently.
"""

from __future__ import annotations

import argparse
import sys
import torch
from dataclasses import dataclass

import gymnasium as gym
import isaaclab.utils.math as math_utils
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

import ur5e_arena.embodiment  # noqa: F401  (registers the UR5e embodiments)

PHASES = ("pre_grasp", "approach", "grasp", "lift", "transport", "lower", "release", "retract", "done")
_PRE_GRASP, _APPROACH, _GRASP, _LIFT, _TRANSPORT, _LOWER, _RELEASE, _RETRACT, _DONE = range(len(PHASES))
_GRIPPER_PHASES = {_GRASP: 25, _RELEASE: 15}
"""Phases that only actuate the gripper, with their duration in policy steps."""


@dataclass
class Ur5eScriptedPickPlacePolicyCfg(PolicyCfg):
    """Configure the scripted UR5e pick-and-place policy."""

    ik_pick_asset: str | None = None
    """Scene name of the object to pick (default: value of --pick_up_object)."""

    ik_place_asset: str | None = None
    """Scene name of the destination (default: value of --destination_location)."""

    ik_hover_height_m: float = 0.15
    """Height of the pre-grasp / lift / retract waypoints above the grasp or place point."""

    ik_grasp_z_offset_m: float = 0.0
    """Raise the grasp point above the object's mid-height by this much."""

    ik_max_step_m: float = 0.02
    """Largest TCP translation commanded per policy step (smooths the IK targets)."""

    ik_position_tolerance_m: float = 0.012
    """TCP distance to a waypoint that counts as reached."""

    ik_phase_timeout_steps: int = 120
    """Move to the next phase after this many steps even if the waypoint is not reached."""

    ik_verbose: bool = True
    """Print phase changes and the scene layout at the first step."""


def _env_prim_path(prim_path: str, env_id: int) -> str:
    """Resolve an Isaac Lab per-env prim path expression for one environment."""
    for pattern in ("{ENV_REGEX_NS}", "/World/envs/env_[^/]+", "/World/envs/env_.*"):
        prim_path = prim_path.replace(pattern, f"/World/envs/env_{env_id}")
    return prim_path


def _names_from_argv() -> tuple[str | None, str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pick_up_object", default=None)
    parser.add_argument("--destination_location", default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.pick_up_object, args.destination_location


@register_policy
class Ur5eScriptedPickPlacePolicy(PolicyBase[Ur5eScriptedPickPlacePolicyCfg]):
    """Waypoint state machine on top of absolute differential IK."""

    name = "ur5e_scripted_pick_place"

    def __init__(self, config: Ur5eScriptedPickPlacePolicyCfg):
        super().__init__(config)
        argv_pick, argv_place = _names_from_argv()
        self.pick_name = config.ik_pick_asset or argv_pick
        self.place_name = config.ik_place_asset or argv_place
        self._phase: torch.Tensor | None = None
        self._phase_steps: torch.Tensor | None = None
        self._cmd_pos_b: torch.Tensor | None = None
        self._frozen_grasp_w: torch.Tensor | None = None
        self._frozen_close_axis_w: torch.Tensor | None = None
        self._local_bounds: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._printed_layout = False

    # ------------------------------------------------------------------------------
    # PolicyBase API
    # ------------------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._phase is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self._phase.numel(), device=self._phase.device)
        self._phase[env_ids] = _PRE_GRASP
        self._phase_steps[env_ids] = 0
        self._cmd_pos_b[env_ids] = torch.nan

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        base_env = env.unwrapped
        scene = base_env.scene
        self._check_setup(scene)
        num_envs, device = scene.num_envs, base_env.device
        if self._phase is None:
            self._phase = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._phase_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._cmd_pos_b = torch.full((num_envs, 3), torch.nan, device=device)

        robot = scene["robot"]
        root_pos_w, root_quat_w = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        tcp_pos_w = scene["ee_frame"].data.target_pos_w.torch[:, 0]
        tcp_pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, tcp_pos_w)

        pick_min_w, pick_max_w, pick_yaw_axis_w = self._world_bounds(scene, self.pick_name)
        place_min_w, place_max_w, _ = self._world_bounds(scene, self.place_name)
        if self.config.ik_verbose and not self._printed_layout:
            self._print_layout(root_pos_w, tcp_pos_w, pick_min_w, pick_max_w, place_min_w, place_max_w)

        # Grasp point: object centre in xy, TCP at mid-height (+ optional offset, never closer than 2 cm to the base).
        height = pick_max_w[:, 2] - pick_min_w[:, 2]
        grasp_w = 0.5 * (pick_min_w + pick_max_w)
        grasp_w[:, 2] = pick_min_w[:, 2] + torch.clamp(0.5 * height + self.config.ik_grasp_z_offset_m, min=0.02)
        # Release point: on top of the destination, object base slightly above its upper surface.
        place_w = 0.5 * (place_min_w + place_max_w)
        place_w[:, 2] = place_max_w[:, 2] + (grasp_w[:, 2] - pick_min_w[:, 2]) + 0.02
        hover = torch.tensor([0.0, 0.0, self.config.ik_hover_height_m], device=device)

        waypoints_w = torch.stack(
            [
                grasp_w + hover,  # pre_grasp
                grasp_w,  # approach
                grasp_w,  # grasp
                grasp_w + hover,  # lift
                place_w + hover,  # transport
                place_w,  # lower
                place_w,  # release
                place_w + hover,  # retract
                place_w + hover,  # done
            ],
            dim=1,
        )
        # Freeze the pick waypoints and wrist yaw once grasped, so they do not chase the object in hand.
        if self._frozen_grasp_w is None:
            self._frozen_grasp_w = grasp_w.clone()
            self._frozen_close_axis_w = pick_yaw_axis_w.clone()
        holding = self._phase >= _GRASP
        self._frozen_grasp_w[~holding] = grasp_w[~holding]
        self._frozen_close_axis_w[~holding] = pick_yaw_axis_w[~holding]
        waypoints_w[:, _GRASP] = torch.where(holding[:, None], self._frozen_grasp_w, waypoints_w[:, _GRASP])
        waypoints_w[:, _LIFT] = torch.where(holding[:, None], self._frozen_grasp_w + hover, waypoints_w[:, _LIFT])

        env_idx = torch.arange(num_envs, device=device)
        goal_w = waypoints_w[env_idx, self._phase]
        goal_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, goal_w)

        # Rate-limit the commanded TCP position.
        uninitialised = torch.isnan(self._cmd_pos_b).any(dim=-1)
        self._cmd_pos_b[uninitialised] = tcp_pos_b[uninitialised]
        delta = goal_b - self._cmd_pos_b
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        self._cmd_pos_b += delta * torch.clamp(self.config.ik_max_step_m / (dist + 1e-9), max=1.0)

        # Tool pointing down; fingers close across the object's narrow horizontal axis.
        quat_b = self._tool_down_quat_b(root_quat_w, self._frozen_close_axis_w)

        gripper = torch.ones(num_envs, 1, device=device)
        gripper[(self._phase >= _GRASP) & (self._phase < _RELEASE)] = -1.0

        self._advance_phases(torch.linalg.norm(tcp_pos_b - goal_b, dim=-1), pick_min_w[:, 2])
        return torch.cat([self._cmd_pos_b, quat_b, gripper], dim=-1)

    # ------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------

    def _check_setup(self, scene) -> None:
        assert "ee_frame" in scene.keys(), "Use a UR5e embodiment (ee_frame missing from the scene)"
        for label, name in (("pick", self.pick_name), ("place", self.place_name)):
            assert name is not None, (
                f"Unknown {label} object: pass --pick_up_object/--destination_location or"
                " --ik_pick_asset/--ik_place_asset"
            )
            assert name in scene.keys(), f"'{name}' is not in the scene. Scene entries: {list(scene.keys())}"

    def _advance_phases(self, error_m: torch.Tensor, pick_bottom_z: torch.Tensor) -> None:
        self._phase_steps += 1
        reached = (error_m < self.config.ik_position_tolerance_m) & (self._phase_steps > 5)
        for phase, duration in _GRIPPER_PHASES.items():
            in_phase = self._phase == phase
            reached = torch.where(in_phase, self._phase_steps >= duration, reached)
        advance = (reached | (self._phase_steps >= self.config.ik_phase_timeout_steps)) & (self._phase < _DONE)
        if advance.any():
            self._phase[advance] += 1
            self._phase_steps[advance] = 0
            if self.config.ik_verbose and bool(advance[0]):
                print(
                    f"[ur5e_scripted] env 0 -> {PHASES[int(self._phase[0])]} (tcp err {float(error_m[0]):.3f} m,"
                    f" {self.pick_name} bottom z {float(pick_bottom_z[0]):+.3f} m)"
                )

    def _local_bound(self, scene, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Object bounds in its own root frame, read once from USD."""
        if name not in self._local_bounds:
            from pxr import Usd, UsdGeom, UsdPhysics

            import isaaclab.sim as sim_utils

            stage = sim_utils.get_current_stage()
            prim_path = _env_prim_path(scene[name].cfg.prim_path, 0)
            prim = stage.GetPrimAtPath(prim_path)
            assert prim.IsValid(), f"Prim not found for '{name}': {prim_path}"
            # The simulator reports the pose of the rigid body, which may be a child of the asset prim.
            body = next((p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)), prim)
            time = Usd.TimeCode.Default()
            cache = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            world_box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            # Both come from the same (spawn-time) USD state, so their relation is valid whatever the live pose.
            world_to_body = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time).RemoveScaleShear().GetInverse()
            corners_body = [world_to_body.Transform(world_box.GetCorner(i)) for i in range(8)]
            device = scene["robot"].data.root_pos_w.torch.device
            corners = torch.tensor([list(c) for c in corners_body], dtype=torch.float32, device=device)
            self._local_bounds[name] = (corners.min(dim=0).values, corners.max(dim=0).values)
        return self._local_bounds[name]

    def _static_world_bounds(self, scene, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World AABB of a static scene entry (e.g. a cabinet in the background) for every env."""
        if name not in self._local_bounds:
            from pxr import Usd, UsdGeom

            import isaaclab.sim as sim_utils

            stage = sim_utils.get_current_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            device = scene["robot"].data.root_pos_w.torch.device
            lows, highs = [], []
            for env_id in range(scene.num_envs):
                prim_path = _env_prim_path(scene[name].prim_path, env_id)
                prim = stage.GetPrimAtPath(prim_path)
                assert prim.IsValid(), f"Prim not found for '{name}': {prim_path}"
                box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                lows.append(list(box.GetMin()))
                highs.append(list(box.GetMax()))
            self._local_bounds[name] = (torch.tensor(lows, device=device), torch.tensor(highs, device=device))
        lo, hi = self._local_bounds[name]
        axis_w = torch.zeros_like(lo)
        axis_w[:, 0] = 1.0
        return lo, hi, axis_w

    def _world_bounds(self, scene, name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World AABB of an object from its live pose, plus its narrow horizontal axis in world."""
        asset = scene[name]
        if not hasattr(asset, "data"):
            return self._static_world_bounds(scene, name)
        lo, hi = self._local_bound(scene, name)
        pos_w, quat_w = asset.data.root_pos_w.torch, asset.data.root_quat_w.torch
        corners = torch.stack(
            [torch.stack([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )  # (8, 3)
        n = pos_w.shape[0]
        corners_w = math_utils.quat_apply(
            quat_w[:, None].expand(n, 8, 4).reshape(-1, 4), corners[None].expand(n, 8, 3).reshape(-1, 3)
        ).reshape(n, 8, 3) + pos_w[:, None]
        size = hi - lo
        local_axis = torch.zeros(3, device=pos_w.device)
        local_axis[0 if size[0] <= size[1] else 1] = 1.0
        axis_w = math_utils.quat_apply(quat_w, local_axis.expand(n, 3))
        axis_w[:, 2] = 0.0
        axis_w = axis_w / torch.clamp(torch.linalg.norm(axis_w, dim=-1, keepdim=True), min=1e-6)
        return corners_w.min(dim=1).values, corners_w.max(dim=1).values, axis_w

    def _tool_down_quat_b(self, root_quat_w: torch.Tensor, close_axis_w: torch.Tensor) -> torch.Tensor:
        """Orientation (base frame) with flange z pointing down and flange x along ``close_axis_w``."""
        z_w = torch.zeros_like(close_axis_w)
        z_w[:, 2] = -1.0
        x_w = close_axis_w
        y_w = torch.linalg.cross(z_w, x_w)
        rot_w = torch.stack([x_w, y_w, z_w], dim=-1)  # columns are the tool axes
        quat_w = math_utils.quat_from_matrix(rot_w)
        quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat_w), quat_w)
        return math_utils.quat_unique(quat_b)

    def _print_layout(self, root_pos_w, tcp_pos_w, pick_min, pick_max, place_min, place_max) -> None:
        self._printed_layout = True

        def fmt(t):
            return "(" + ", ".join(f"{v:+.3f}" for v in t[0].tolist()) + ")"

        print("[ur5e_scripted] ---- scene layout (env 0, world frame) ----")
        print(f"[ur5e_scripted] robot base      {fmt(root_pos_w)}")
        print(f"[ur5e_scripted] TCP             {fmt(tcp_pos_w)}")
        print(f"[ur5e_scripted] {self.pick_name:<16s}min {fmt(pick_min)} max {fmt(pick_max)}")
        print(f"[ur5e_scripted] {self.place_name:<16s}min {fmt(place_min)} max {fmt(place_max)}")
