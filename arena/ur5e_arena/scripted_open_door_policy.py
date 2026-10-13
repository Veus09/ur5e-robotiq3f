"""Scripted open-door policy for the ``ur5e_3f_ik_abs`` embodiment (IK on a hinged fridge/freezer door).

The policy reads the door geometry from the stage (hinge pivot, hinge axis, handle bar), then drives the TCP
with absolute IK targets (robot base frame):

    pre_grasp  tool horizontal, TCP in front of the handle
    approach   slide the open gripper over the vertical handle bar
    grasp      close the Robotiq 3F on the bar
    pull       follow the handle's circular arc around the hinge; the tool yaw turns with the door
    release    open the gripper
    retract    back off along the approach direction

    python run_arena.py --viz kit \\
        --policy_type ur5e_arena.scripted_open_door_policy.Ur5eScriptedOpenDoorPolicy --num_episodes 3 \\
        --env_spec ~/ur5eredodo/arena/envs/open_freezer_ur5e.yaml
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass

import gymnasium as gym
import isaaclab.utils.math as math_utils
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

import ur5e_arena.embodiment  # noqa: F401  (registers the UR5e embodiments)

PHASES = ("pre_grasp", "approach", "grasp", "pull", "release", "retract", "done")
_PRE_GRASP, _APPROACH, _GRASP, _PULL, _RELEASE, _RETRACT, _DONE = range(len(PHASES))
_GRIPPER_PHASES = {_GRASP: 30, _RELEASE: 20}
"""Phases that only actuate the gripper, with their duration in policy steps."""


@dataclass
class Ur5eScriptedOpenDoorPolicyCfg(PolicyCfg):
    """Configure the scripted UR5e open-door policy."""

    ik_door_asset: str = "fridge"
    """Scene name of the articulation that owns the door."""

    ik_door_joint: str = "freezer_door_joint"
    """Revolute joint of the door to open."""

    ik_target_angle_rad: float = 1.0
    """Door opening angle the pull phase aims for, capped at 90 % of the joint range."""

    ik_grasp_height_m: float | None = None
    """World z of the grasp point on the handle (default: handle mid-height, clamped to the arm's comfortable band)."""

    ik_standoff_m: float = 0.15
    """Distance in front of the handle for the pre-grasp and retract waypoints."""

    ik_depth_offset_m: float = 0.0
    """Shift the TCP along the approach direction relative to the handle axis (positive = deeper)."""

    ik_max_step_m: float = 0.015
    """Largest TCP translation commanded per policy step."""

    ik_max_rot_step_rad: float = 0.05
    """Largest TCP rotation commanded per policy step."""

    ik_pull_rate_rad: float = 0.006
    """Door-angle increase commanded per policy step while pulling."""

    ik_position_tolerance_m: float = 0.012
    """TCP distance to a waypoint that counts as reached."""

    ik_phase_timeout_steps: int = 200
    """Move to the next phase after this many steps even if the waypoint is not reached."""

    ik_verbose: bool = True
    """Print phase changes and the door geometry at the first step."""


def _env_prim_path(prim_path: str, env_id: int) -> str:
    for pattern in ("{ENV_REGEX_NS}", "/World/envs/env_[^/]+", "/World/envs/env_.*"):
        prim_path = prim_path.replace(pattern, f"/World/envs/env_{env_id}")
    return prim_path


def _rot_z(angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices about world z for a batch of angles, shape ``(N, 3, 3)``."""
    c, s = torch.cos(angle), torch.sin(angle)
    zeros, ones = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, zeros, s, c, zeros, zeros, zeros, ones], dim=-1).reshape(-1, 3, 3)


@register_policy
class Ur5eScriptedOpenDoorPolicy(PolicyBase[Ur5eScriptedOpenDoorPolicyCfg]):
    """Waypoint state machine that grabs a vertical door handle and swings the door open."""

    name = "ur5e_scripted_open_door"

    def __init__(self, config: Ur5eScriptedOpenDoorPolicyCfg):
        super().__init__(config)
        self._geometry: dict | None = None
        self._phase: torch.Tensor | None = None
        self._phase_steps: torch.Tensor | None = None
        self._cmd_pos_w: torch.Tensor | None = None
        self._cmd_quat_w: torch.Tensor | None = None
        self._cmd_angle: torch.Tensor | None = None

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
        self._cmd_pos_w[env_ids] = torch.nan
        self._cmd_quat_w[env_ids] = torch.nan
        self._cmd_angle[env_ids] = 0.0

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        base_env = env.unwrapped
        scene = base_env.scene
        assert "ee_frame" in scene.keys(), "Use a UR5e embodiment (ee_frame missing from the scene)"
        num_envs, device = scene.num_envs, base_env.device
        if self._geometry is None:
            self._geometry = self._read_geometry(scene)
            self._phase = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._phase_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._cmd_pos_w = torch.full((num_envs, 3), torch.nan, device=device)
            self._cmd_quat_w = torch.full((num_envs, 4), torch.nan, device=device)
            self._cmd_angle = torch.zeros(num_envs, device=device)
        geo = self._geometry
        cfg = self.config

        robot = scene["robot"]
        root_pos_w, root_quat_w = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        tcp_pos_w = scene["ee_frame"].data.target_pos_w.torch[:, 0]
        tcp_quat_w = scene["ee_frame"].data.target_quat_w.torch[:, 0]
        # Opening angle: the joint value signed so that opening is positive (some doors open toward negative angles).
        door_angle = geo["open_sign"] * scene[cfg.ik_door_asset].data.joint_pos.torch[:, geo["joint_id"]]
        target_angle = min(cfg.ik_target_angle_rad, 0.9 * geo["range_rad"])

        # Hinge-frame geometry of this env (the stage holds env 0; other envs are translated copies).
        origin = scene.env_origins - scene.env_origins[0]
        hinge_w = geo["hinge_w"] + origin
        handle_w = geo["handle_w"] + origin
        approach_w = geo["approach_w"].expand(num_envs, 3)

        # Door angle the TCP should track: ramp from the current angle while pulling, hold otherwise.
        pulling = self._phase == _PULL
        self._cmd_angle = torch.where(
            pulling,
            torch.minimum(self._cmd_angle + cfg.ik_pull_rate_rad, torch.full_like(door_angle, target_angle)),
            torch.where(self._phase < _PULL, torch.zeros_like(door_angle), self._cmd_angle),
        )
        signed = geo["hinge_sign"] * geo["open_sign"] * self._cmd_angle
        rot = _rot_z(signed)
        arc_handle_w = hinge_w + torch.einsum("nij,nj->ni", rot, handle_w - hinge_w)
        arc_approach_w = torch.einsum("nij,nj->ni", rot, approach_w)
        standoff = cfg.ik_standoff_m * arc_approach_w
        depth = cfg.ik_depth_offset_m * arc_approach_w

        waypoints_w = torch.stack(
            [
                handle_w - standoff,  # pre_grasp
                handle_w + depth,  # approach
                handle_w + depth,  # grasp
                arc_handle_w + depth,  # pull
                arc_handle_w + depth,  # release
                arc_handle_w - standoff,  # retract
                arc_handle_w - standoff,  # done
            ],
            dim=1,
        )
        env_idx = torch.arange(num_envs, device=device)
        goal_pos_w = waypoints_w[env_idx, self._phase]

        # Tool horizontal along the door-normal, fingers closing across the radial direction, turning with the door.
        goal_quat_w = self._tool_quat_w(arc_approach_w)

        self._rate_limit(tcp_pos_w, tcp_quat_w, goal_pos_w, goal_quat_w)

        cmd_pos_b, cmd_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, self._cmd_pos_w, self._cmd_quat_w
        )
        cmd_quat_b = math_utils.quat_unique(cmd_quat_b)

        gripper = torch.ones(num_envs, 1, device=device)
        gripper[(self._phase >= _GRASP) & (self._phase < _RELEASE)] = -1.0

        pos_err = torch.linalg.norm(tcp_pos_w - goal_pos_w, dim=-1)
        rot_err = math_utils.quat_error_magnitude(tcp_quat_w, goal_quat_w)
        if cfg.ik_verbose and bool(pulling[0]) and self._phase_steps[0] % 20 == 0:
            print(
                f"[ur5e_door] pull: door {math.degrees(float(door_angle[0])):.1f} deg, cmd {math.degrees(float(self._cmd_angle[0])):.1f} deg,"
                f" tcp err {float(pos_err[0]):.3f} m"
            )
        self._advance_phases(pos_err, rot_err, door_angle, target_angle)
        return torch.cat([cmd_pos_b, cmd_quat_b, gripper], dim=-1)

    # ------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------

    def _rate_limit(self, tcp_pos_w, tcp_quat_w, goal_pos_w, goal_quat_w) -> None:
        """Move the commanded pose toward the goal by at most one step (initialised from the live TCP pose)."""
        fresh = torch.isnan(self._cmd_pos_w).any(dim=-1)
        self._cmd_pos_w[fresh] = tcp_pos_w[fresh]
        self._cmd_quat_w[fresh] = tcp_quat_w[fresh]

        delta = goal_pos_w - self._cmd_pos_w
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        self._cmd_pos_w = self._cmd_pos_w + delta * torch.clamp(self.config.ik_max_step_m / (dist + 1e-9), max=1.0)

        angle = math_utils.quat_error_magnitude(self._cmd_quat_w, goal_quat_w)
        frac = torch.clamp(self.config.ik_max_rot_step_rad / (angle + 1e-9), max=1.0)[:, None]
        # Normalised lerp on the shorter arc (a batched slerp is not in Isaac Lab); exact enough for small steps.
        sign = torch.sign(torch.sum(self._cmd_quat_w * goal_quat_w, dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        blended = (1.0 - frac) * self._cmd_quat_w + frac * sign * goal_quat_w
        self._cmd_quat_w = blended / torch.linalg.norm(blended, dim=-1, keepdim=True)

    def _tool_quat_w(self, approach_w: torch.Tensor) -> torch.Tensor:
        """Orientation with flange z along ``approach_w`` and flange y pointing up (flange x closes horizontally)."""
        up = torch.zeros_like(approach_w)
        up[:, 2] = 1.0
        z_axis = approach_w
        x_axis = torch.linalg.cross(up, z_axis)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        return math_utils.quat_unique(math_utils.quat_from_matrix(torch.stack([x_axis, y_axis, z_axis], dim=-1)))

    def _advance_phases(
        self, pos_err: torch.Tensor, rot_err: torch.Tensor, door_angle: torch.Tensor, target_angle: float
    ) -> None:
        self._phase_steps += 1
        reached = (pos_err < self.config.ik_position_tolerance_m) & (rot_err < 0.08) & (self._phase_steps > 5)
        # The pull ends when the door reached the target angle.
        reached = torch.where(
            self._phase == _PULL, door_angle >= target_angle - 0.02, reached
        )
        for phase, duration in _GRIPPER_PHASES.items():
            reached = torch.where(self._phase == phase, self._phase_steps >= duration, reached)
        timeout = self._phase_steps >= self.config.ik_phase_timeout_steps
        advance = (reached | (timeout & (self._phase != _PULL))) & (self._phase < _DONE)
        if advance.any():
            self._phase[advance] += 1
            self._phase_steps[advance] = 0
            if self.config.ik_verbose and bool(advance[0]):
                print(
                    f"[ur5e_door] env 0 -> {PHASES[int(self._phase[0])]} (tcp err {float(pos_err[0]):.3f} m,"
                    f" door {math.degrees(float(door_angle[0])):.1f} deg)"
                )

    def _read_geometry(self, scene) -> dict:
        """Hinge pivot/axis, handle centre and approach direction (world frame, env 0), read from the stage."""
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        import isaaclab.sim as sim_utils

        cfg = self.config
        device = scene["robot"].data.root_pos_w.torch.device
        asset = scene[cfg.ik_door_asset]
        assert cfg.ik_door_joint in asset.joint_names, f"'{cfg.ik_door_joint}' not in {asset.joint_names}"
        joint_id = asset.joint_names.index(cfg.ik_door_joint)

        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(_env_prim_path(asset.cfg.prim_path, 0))
        assert root.IsValid(), f"Prim not found for '{cfg.ik_door_asset}'"
        joint_prim = next(p for p in Usd.PrimRange(root) if p.GetName() == cfg.ik_door_joint)
        joint = UsdPhysics.RevoluteJoint(joint_prim)
        body0 = stage.GetPrimAtPath(str(joint.GetBody0Rel().GetTargets()[0]))
        door = stage.GetPrimAtPath(str(joint.GetBody1Rel().GetTargets()[0]))

        time = Usd.TimeCode.Default()
        body0_to_world = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(time).RemoveScaleShear()
        hinge = body0_to_world.Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        axis_local = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[joint.GetAxisAttr().Get()]
        local_rot0 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
        axis_w = np.array(body0_to_world.TransformDir(local_rot0.TransformDir(Gf.Vec3d(*axis_local))))
        assert abs(axis_w[2]) > 0.9, f"Door hinge axis is not vertical: {axis_w}"
        hinge_sign = 1.0 if axis_w[2] > 0 else -1.0

        # World points of every mesh on the door body.
        pts = []
        for prim in Usd.PrimRange(door):
            if prim.IsA(UsdGeom.Mesh):
                local = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time))
                pts.append((np.c_[local, np.ones(len(local))] @ m)[:, :3])
        pts = np.concatenate(pts)
        centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        extent = pts.max(axis=0) - pts.min(axis=0)

        # Door normal: the thinner horizontal axis of the door, pointing away from the robot side.
        robot_xy = scene["robot"].data.root_pos_w.torch[0, :2].cpu().numpy()
        thin = 0 if extent[0] < extent[1] else 1
        outward = np.zeros(3)
        outward[thin] = np.sign(robot_xy[thin] - centre[thin]) or 1.0
        # Handle = points sticking out furthest toward the robot: the front slab of the door mesh.
        depth_out = pts @ outward
        slab = pts[depth_out > depth_out.max() - 0.025]
        slab_lo, slab_hi = slab.min(axis=0), slab.max(axis=0)
        handle = 0.5 * (slab_lo + slab_hi)
        # The bar's axis sits half a bar width behind its front tip (bar width = slab extent across the door).
        bar_width = float((slab_hi - slab_lo)[1 - thin])
        tip_coord = depth_out.max() * outward[thin]
        handle[thin] = tip_coord - outward[thin] * 0.5 * bar_width
        if cfg.ik_grasp_height_m is not None:
            handle[2] = cfg.ik_grasp_height_m

        def vec(v):
            return torch.tensor(np.asarray(v, dtype=np.float32), device=device)

        lower, upper = asset.data.joint_pos_limits.torch[0, joint_id].tolist()
        open_sign = 1.0 if upper >= -lower else -1.0
        print(f"[ur5e_door] joint '{cfg.ik_door_joint}' limits [{lower:+.2f}, {upper:+.2f}] rad -> opens toward {open_sign:+.0f}")

        geometry = {
            "joint_id": joint_id,
            "hinge_w": vec(hinge)[None],
            "hinge_sign": hinge_sign,
            "open_sign": open_sign,
            "range_rad": max(upper, -lower),
            "handle_w": vec(handle)[None],
            "approach_w": vec(-outward)[None],
        }
        print(
            f"[ur5e_door] hinge {np.round(np.array(hinge), 3)} axis_z {axis_w[2]:+.0f}, handle {np.round(handle, 3)}"
            f" (slab {np.round(slab_hi - slab_lo, 3)}), approach {-outward}"
        )
        return geometry
