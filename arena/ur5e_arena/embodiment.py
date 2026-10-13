"""UR5e + Robotiq 3F embodiments for Isaac Lab-Arena.

Registers three embodiments that drop into any Arena environment through ``--embodiment``:

    ur5e_3f_ik          differential IK, relative (delta) end-effector pose actions   [7 = 6 pose + 1 gripper]
    ur5e_3f_ik_abs      differential IK, absolute end-effector pose actions          [8 = 3 pos + 4 quat xyzw + 1 gripper]
    ur5e_3f_joint_pos   absolute joint-position actions                              [7 = 6 joints + 1 gripper]

Gripper action: value < 0 closes the Robotiq 3F, value >= 0 opens it.

This module must be imported after the SimulationApp starts (it imports Isaac Lab / pxr).
"""

from __future__ import annotations

import functools
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.contrib.stack.mdp.observations import ee_frame_pos, ee_frame_quat
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.franka.franka import franka_stack_events
from isaaclab_arena.embodiments.gripper import Gripper
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv

# =====================================================================================
# Robot asset
# =====================================================================================

UR5E_ROBOT_DIR = Path(os.environ.get("UR5E_ROBOT_DIR", Path(__file__).resolve().parents[2] / "ur"))
"""Folder holding ur.usda (override with the UR5E_ROBOT_DIR environment variable)."""

UR5E_USD_PATH = UR5E_ROBOT_DIR / "ur.usda"

# Link prims inside ur.usda (default prim /ur, spawned by Arena at {ENV_REGEX_NS}/Robot).
_BASE_LINK = "Geometry/world/base_link"
_WRIST_3_LINK = (
    f"{_BASE_LINK}/base_link_inertia/shoulder_link/upper_arm_link/forearm_link/wrist_1_link/wrist_2_link/wrist_3_link"
)
_PALM_LINK = "robotiq_3f_gripper_articulated/Geometry/palm"

TOOL_BODY_NAME = "wrist_3_link"
"""Body the IK controller drives (flange). The TCP sits ``TCP_OFFSET_M`` along its +z axis."""

TCP_OFFSET_M = 0.19
"""Flange -> grasp point between the Robotiq 3F fingertips (fingertips are ~0.21 m out when open)."""

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ARM_HOME = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": -1.57,
    "wrist_3_joint": 0.0,
}
"""Arm up, tool pointing down (same home as scripts/pick_box_table.py)."""

# Robotiq 3F, "basic" mode with parallel fingertips: joint_2 = 0, joint_3 = -joint_1.
# finger_2_joint_3 has a flipped limit ([3, 70] deg) so it uses the opposite sign.
GRIPPER_OPEN = {
    "palm_finger_1_joint": 0.0173,
    "palm_finger_2_joint": 0.0173,
    "finger_1_joint_1": 0.0496,
    "finger_2_joint_1": 0.0496,
    "finger_middle_joint_1": 0.0496,
    "finger_1_joint_2": 0.0,
    "finger_2_joint_2": 0.0,
    "finger_middle_joint_2": 0.0,
    "finger_1_joint_3": -0.0524,
    "finger_2_joint_3": 0.0524,
    "finger_middle_joint_3": -0.0524,
}
_CLOSE_RAD = 1.0
"""Past the contact angle for small objects so the PD drive keeps squeezing."""
GRIPPER_CLOSED = {
    **GRIPPER_OPEN,
    "finger_1_joint_1": _CLOSE_RAD,
    "finger_2_joint_1": _CLOSE_RAD,
    "finger_middle_joint_1": _CLOSE_RAD,
    "finger_1_joint_3": -_CLOSE_RAD,
    "finger_2_joint_3": _CLOSE_RAD,
    "finger_middle_joint_3": -_CLOSE_RAD,
}
GRIPPER_JOINT_NAMES = tuple(GRIPPER_OPEN)

DEFAULT_PEDESTAL_HEIGHT_M = 0.02
"""Visual/collision pedestal under the base. Arena tabletop scenes put the table top near z=0."""


# =====================================================================================
# Robot + pedestal USD
# =====================================================================================


def compose_ur5e_usd(pedestal_height_m: float = DEFAULT_PEDESTAL_HEIGHT_M) -> str:
    """Return a cached USD that references ur.usda and adds a box pedestal under ``base_link``.

    Args:
        pedestal_height_m: Pedestal height below the robot base, in meters. 0 disables the pedestal.
    """
    return _compose_ur5e_usd_cached(str(UR5E_USD_PATH), round(float(pedestal_height_m), 4))


@functools.cache
def _compose_ur5e_usd_cached(robot_usd_path: str, pedestal_height_m: float) -> str:
    assert Path(robot_usd_path).is_file(), f"UR5e USD not found: {robot_usd_path}"
    out_dir = get_arena_asset_cache_dir().parent / "usd" / "ur5e_3f"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ur5e_robotiq3f_pedestal_{pedestal_height_m}.usda"

    out_path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = stage.DefinePrim("/ur", "Xform")
    root.GetReferences().AddReference(robot_usd_path, "/ur")
    stage.SetDefaultPrim(root)

    if pedestal_height_m > 0.0:
        # Child of base_link, so it moves with the fixed base and joins the base collider.
        pedestal = UsdGeom.Cube.Define(stage, f"/ur/{_BASE_LINK}/pedestal")
        pedestal.CreateSizeAttr(1.0)
        pedestal.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -pedestal_height_m / 2.0 - 0.001))
        pedestal.AddScaleOp().Set(Gf.Vec3f(0.22, 0.22, pedestal_height_m))
        pedestal.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.25, 0.27)])
        UsdPhysics.CollisionAPI.Apply(pedestal.GetPrim())

    assert stage.GetRootLayer().Save(), f"Failed to save {out_path}"
    return str(out_path)


# =====================================================================================
# Gripper interface (used by Arena predicates that ask about the gripper)
# =====================================================================================

_FINGER_LINK_LENGTH_M = 0.057
"""Approximate proximal-phalanx length, used to turn knuckle angles into meters."""
_OPEN_WIDTH_M = 0.155
"""Approximate fingertip gap with the gripper fully open."""


@dataclass(frozen=True, kw_only=True)
class Robotiq3FGripper(Gripper):
    """Robotiq 3F gripper approximated by its middle-finger knuckle angle."""

    knuckle_joint_name: str = "finger_middle_joint_1"
    """Joint used to measure the opening."""

    frame_transformer_name: str = "ee_frame"
    """Scene key of the end-effector frame transformer."""

    target_frame_name: str = "end_effector"
    """Frame-transformer target at the grasp point."""

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)

    def get_opening_width_m(self, world: ArenaWorld) -> torch.Tensor:
        angle = world.get_joint_position("robot", self.knuckle_joint_name)
        closed_by = 2.0 * _FINGER_LINK_LENGTH_M * torch.sin(angle - GRIPPER_OPEN[self.knuckle_joint_name])
        return torch.clamp(_OPEN_WIDTH_M - closed_by, min=0.0, max=_OPEN_WIDTH_M)

    def get_closing_error_m(self, env: IsaacLabArenaManagerBasedRLEnv) -> torch.Tensor:
        world = env.arena_world
        measured = world.get_joint_position("robot", self.knuckle_joint_name)
        commanded = world.get_joint_position_target("robot", self.knuckle_joint_name)
        # The knuckle angle grows as the finger closes, so "more open than commanded" is commanded > measured.
        return (commanded - measured) * _FINGER_LINK_LENGTH_M


# =====================================================================================
# Scene / observations / events
# =====================================================================================


def _ur5e_robot_cfg(pedestal_height_m: float) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=compose_ur5e_usd(pedestal_height_m),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={**ARM_HOME, **GRIPPER_OPEN},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            # Gravity is disabled on the robot, so stiff PD gains track IK targets closely.
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint"],
                effort_limit_sim=330.0,
                velocity_limit_sim=3.14,
                stiffness=2000.0,
                damping=120.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
                stiffness=1500.0,
                damping=100.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*_joint"],
                # Real UR5e wrists are 28 Nm; that saturates in sim (see pick_box_table.py), so allow more.
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
                stiffness=800.0,
                damping=60.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_.*_joint_.*", "palm_finger_.*_joint"],
                effort_limit_sim=5.0,
                velocity_limit_sim=2.0,
                stiffness=5.0,
                damping=0.5,
            ),
        },
    )


@configclass
class Ur5e3FSceneCfg:
    """Scene entries added by the UR5e + Robotiq 3F embodiment."""

    robot: ArticulationCfg | None = None

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{_BASE_LINK}",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{_WRIST_3_LINK}",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, TCP_OFFSET_M)),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


def gripper_knuckle_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the three Robotiq 3F knuckle (joint_1) angles in radians, shape ``(num_envs, 3)``."""
    robot = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"], preserve_order=True)
    return robot.data.joint_pos.torch[:, joint_ids]


@configclass
class Ur5e3FObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_knuckle_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ur5e3FEventCfg:
    """Small Gaussian noise on the arm joints at reset (gripper stays exactly open)."""

    randomize_arm_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES)),
        },
    )


# =====================================================================================
# Actions
# =====================================================================================

_GRIPPER_ACTION = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=list(GRIPPER_JOINT_NAMES),
    open_command_expr=GRIPPER_OPEN,
    close_command_expr=GRIPPER_CLOSED,
)

_TCP_OFFSET = DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, TCP_OFFSET_M))


@configclass
class Ur5e3FIKActionCfg:
    """Relative IK: action[:6] is a (dx, dy, dz, rx, ry, rz) TCP delta in the robot base frame."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=TOOL_BODY_NAME,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=_TCP_OFFSET,
    )
    gripper_action: ActionTermCfg = _GRIPPER_ACTION.copy()


@configclass
class Ur5e3FIKAbsActionCfg:
    """Absolute IK: action[:7] is the TCP target (x, y, z, qx, qy, qz, qw) in the robot base frame."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=TOOL_BODY_NAME,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=_TCP_OFFSET,
    )
    gripper_action: ActionTermCfg = _GRIPPER_ACTION.copy()


@configclass
class Ur5e3FJointPosActionCfg:
    """Absolute joint targets for the 6 arm joints (radians)."""

    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=1.0,
        use_default_offset=False,
    )
    gripper_action: ActionTermCfg = _GRIPPER_ACTION.copy()


# =====================================================================================
# Embodiments
# =====================================================================================


class Ur5e3FEmbodimentBase(EmbodimentBase):
    """UR5e arm + Robotiq 3F gripper on a pedestal. Subclasses choose the action space.

    ``initial_pose`` places the robot base in the environment frame (Arena tabletop scenes have the
    table top near z=0 with the robot at the origin facing +x).
    """

    default_arm_mode = ArmMode.SINGLE_ARM
    _default_initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        initial_joint_pose: list[float] | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        pedestal_height_m: float = DEFAULT_PEDESTAL_HEIGHT_M,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.gripper = Robotiq3FGripper()
        self.scene_config = Ur5e3FSceneCfg()
        self.scene_config.robot = _ur5e_robot_cfg(pedestal_height_m)
        self.observation_config = Ur5e3FObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = Ur5e3FEventCfg()
        self.reward_config = None
        self.mimic_env = None
        self.camera_config = None
        if initial_joint_pose is not None:
            self.set_initial_joint_pose(initial_joint_pose)

    def set_initial_joint_pose(self, initial_joint_pose: list[float]) -> None:
        """Set the 6 arm joint angles (radians) used at spawn and reset."""
        assert len(initial_joint_pose) == len(ARM_JOINT_NAMES), f"expected 6 arm joints, got {initial_joint_pose}"
        self.set_joint_initial_pos(dict(zip(ARM_JOINT_NAMES, initial_joint_pose)))

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return TOOL_BODY_NAME


@register_asset
class Ur5e3FIKEmbodiment(Ur5e3FEmbodimentBase):
    """UR5e + Robotiq 3F with relative differential-IK control."""

    name = "ur5e_3f_ik"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_config = Ur5e3FIKActionCfg()


@register_asset
class Ur5e3FIKAbsEmbodiment(Ur5e3FEmbodimentBase):
    """UR5e + Robotiq 3F with absolute differential-IK control (TCP pose targets)."""

    name = "ur5e_3f_ik_abs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_config = Ur5e3FIKAbsActionCfg()


@register_asset
class Ur5e3FJointPosEmbodiment(Ur5e3FEmbodimentBase):
    """UR5e + Robotiq 3F with absolute joint-position control."""

    name = "ur5e_3f_joint_pos"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_config = Ur5e3FJointPosActionCfg()

