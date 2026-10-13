"""Gắp một khối hộp trên bàn bằng UR5e + Robotiq 3F (cuMotion RMPflow) trong Isaac Sim 6.0.

Cách chạy (dùng python.sh của Isaac Sim, KHÔNG dùng python hệ thống):

    cd /path/to/isaac-sim
    ./python.sh /home/vuonggia/ur5eredodo/scripts/pick_box_table.py
    ./python.sh /home/vuonggia/ur5eredodo/scripts/pick_box_table.py --gripper-test   # chỉ thử đóng/mở gripper

Script chia thành 6 PART:
    PART 1  Tham số dòng lệnh + khởi động SimulationApp
    PART 2  Hằng số: đường dẫn, kích thước bàn/hộp, tư thế home, cấu hình gripper 3F
    PART 3  Dựng scene: sàn, bệ robot, bàn, hộp, robot
    PART 4  Khởi tạo cuMotion: world (vật cản) + robot (URDF/XRDF) + RMPflow (IK/điều khiển)
    PART 5  State machine gắp-thả 8 pha
    PART 6  Vòng lặp mô phỏng (main)
"""

# =====================================================================================
# PART 1. THAM SỐ DÒNG LỆNH + KHỞI ĐỘNG SIMULATIONAPP
# -------------------------------------------------------------------------------------
# SimulationApp PHẢI được tạo trước khi import bất kỳ module isaacsim.* / omni.* nào,
# vì các module đó chỉ tồn tại sau khi Kit (nhân của Isaac Sim) đã chạy.
# =====================================================================================
import argparse
from pathlib import Path

from isaacsim import SimulationApp

_REPO_DIR = Path(__file__).resolve().parent.parent  # .../ur5eredodo

parser = argparse.ArgumentParser(description="UR5e + Robotiq 3F gắp hộp trên bàn")
parser.add_argument("--headless", action="store_true", help="Chạy không có cửa sổ")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--robot-dir", type=Path, default=_REPO_DIR / "ur", help="Thư mục chứa ur.usda/robot.urdf/robot.xrdf")
parser.add_argument("--gripper-test", action="store_true", help="Chỉ đóng/mở gripper liên tục, không chạy cánh tay")
parser.add_argument("--test", action="store_true", help="Thoát sau khi chạy xong 1 lượt (dùng cho CI)")
args, _ = parser.parse_known_args()

for f in ("ur.usda", "robot.urdf", "robot.xrdf", "rmp_flow.yaml"):
    if not (args.robot_dir / f).is_file():
        parser.error(f"Không tìm thấy {args.robot_dir / f}")

simulation_app = SimulationApp({"headless": args.headless})

# Sau dòng này mới import được API của Isaac Sim.
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.app
import warp as wp
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, DomeLight, GroundPlane, Mesh
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, RmpFlowController, load_cumotion_robot

# =====================================================================================
# PART 2. HẰNG SỐ CẤU HÌNH
# -------------------------------------------------------------------------------------
# Toàn bộ số liệu hình học được tính từ robot.urdf (forward kinematics):
#   * wrist_3_link -> palm: 0.0434 m dọc trục z của wrist_3 (trục z = trục mặt bích, hướng ra ngoài).
#   * Đầu ngón (đốt 3) khi MỞ nằm cách mặt bích ~0.209 m, khi kẹp hộp 6 cm thì ~0.219 m.
#   * Ngón 1, 2 nằm ở phía -x của wrist_3, ngón giữa ở phía +x  => gripper kẹp theo trục x của wrist_3.
#   * Khi j1 ≈ 0.6 rad (j2 = 0, j3 = -j1) hai mặt đốt 3 cách nhau ~0.059 m  => vừa hộp 6 cm.
# =====================================================================================
ROBOT_PRIM = "/World/ur5e"
BOX_PRIM = "/World/box"
TABLE_PRIM = "/World/table"
PEDESTAL_PRIM = "/World/pedestal"

# Toạ độ thế giới: gốc robot đặt tại (0,0,0), mặt bàn cũng ở z = 0 (robot đứng trên bệ cao bằng bàn).
TABLE_TOP_Z = 0.0
TABLE_HEIGHT = 0.75
TABLE_CENTER_XY = (0.575, 0.10)   # bàn nằm phía +x của robot (UR5e trong URDF này vươn theo +x)
TABLE_SIZE_XY = (0.55, 0.90)

BOX_SIZE = 0.06                    # cạnh hộp (m)
BOX_MASS = 0.10                    # kg - nhẹ để gripper giữ chắc
BOX_FRICTION = 1.0
BOX_START_XY = (0.55, 0.00)        # vị trí hộp ban đầu
BOX_PLACE_XY = (0.45, 0.35)        # vị trí đặt hộp

# Tool frame mà cuMotion điều khiển (khai báo trong robot.xrdf: tool_frames: ["wrist_3_link"]).
TOOL_FRAME = "wrist_3_link"
# Khoảng cách dọc trục công cụ từ mặt bích wrist_3 tới đầu ngón (lấy giá trị lúc đã kẹp: 0.219 m).
FINGERTIP_OFFSET = 0.22
# Khi gắp, đầu ngón cách mặt bàn bao nhiêu (đủ để không cọ bàn nhưng vẫn ôm gần hết chiều cao hộp).
GRASP_CLEARANCE = 0.015
# Độ cao nâng lên trên vị trí gắp/đặt (pre-grasp, lift, retract).
LIFT_HEIGHT = 0.20

# Hướng công cụ: xoay 180° quanh trục y  => trục z của wrist_3 chỉ thẳng xuống (-Z thế giới).
# euler_angles_to_quaternion trả quaternion dạng (w, x, y, z).
DOWN_ORIENTATION = transform_utils.euler_angles_to_quaternion(np.array([0.0, np.pi, 0.0])).numpy()

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Tư thế home: cánh tay dựng lên, gripper chỉ xuống, mặt bích ở ~(0.49, 0.13, 0.49).
# Không dùng default_joint_positions của XRDF (toàn 0) vì khi đó cánh tay nằm ngang sát mặt bàn.
ARM_HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# --- Robotiq 3F -------------------------------------------------------------------
# Mỗi ngón có 3 khớp: joint_1 (gốc ngón), joint_2 (đốt giữa), joint_3 (đầu ngón).
# Ở chế độ "basic" (2 ngón bên song song với ngón giữa), palm_finger_1/2 giữ nguyên.
# Kiểu kẹp "song song": joint_2 = 0, joint_3 = -joint_1  => đốt đầu ngón luôn thẳng đứng,
# ép phẳng vào 2 mặt hộp. Dấu của finger_2_joint_3 bị đảo trong ur.usda (giới hạn [3°, 70°])
# nên dùng giá trị dương cho ngón này. Nếu --gripper-test thấy ngón 2 cong ra ngoài, đổi dấu.
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
# Chạm hộp 6 cm ở j1 ≈ 0.6 rad. Đặt target 1.0 rad (vượt qua điểm chạm) để drive PD của khớp
# luôn tạo lực ép lên hộp — đây là cách "kẹp có lực" khi chỉ điều khiển vị trí.
_CLOSE = 1.0
GRIPPER_CLOSED = {
    "palm_finger_1_joint": 0.0173,
    "palm_finger_2_joint": 0.0173,
    "finger_1_joint_1": _CLOSE,
    "finger_2_joint_1": _CLOSE,
    "finger_middle_joint_1": _CLOSE,
    "finger_1_joint_2": 0.0,
    "finger_2_joint_2": 0.0,
    "finger_middle_joint_2": 0.0,
    "finger_1_joint_3": -_CLOSE,
    "finger_2_joint_3": _CLOSE,
    "finger_middle_joint_3": -_CLOSE,
}
# Nếu USD không gán stiffness cho khớp ngón (= 0) thì position target vô tác dụng; khi đó gán giá trị này.
GRIPPER_FALLBACK_STIFFNESS = 100.0
GRIPPER_FALLBACK_DAMPING = 5.0

# Giới hạn mô-men 3 khớp cổ tay trong ur.usda là 28 Nm (đúng thông số UR5e thật), nhưng trong mô phỏng
# drive PD có stiffness rất lớn nên bị bão hoà: wrist_1 lệch tới 0.7 rad so với lệnh, quỹ đạo đi sai và
# ngón tay đẩy văng hộp. Nâng giới hạn lên để cổ tay bám lệnh (chỉ áp dụng trong lúc chạy, không sửa USD).
WRIST_MAX_EFFORT = 150.0
# Vận tốc khớp tối đa mà RMPflow được phép ra lệnh (mặc định 2.15 rad/s trong rmp_flow.yaml).
# Chậm hơn => drive bám tốt hơn, hạ xuống hộp chính xác hơn.
RMP_MAX_JOINT_VELOCITY = 0.6

PHYSICS_DT = 1.0 / 60.0


# =====================================================================================
# PART 3. DỰNG SCENE
# =====================================================================================
async def build_scene() -> None:
    """Tạo sàn, bệ robot, bàn, hộp và nạp robot UR5e + 3F vào stage."""
    # Sàn đặt dưới chân bàn; đèn để nhìn thấy scene.
    GroundPlane("/World/ground", positions=[[0.0, 0.0, TABLE_TOP_Z - TABLE_HEIGHT]])
    DomeLight("/World/light").set_intensities(1000)

    # Bàn và bệ là Cube CHỈ có collision (không có RigidBody) => vật thể tĩnh, không bị rơi.
    # Cube(sizes=1.0) là khối 1 m, kích thước thật được quyết định bằng scales.
    table = Cube(
        TABLE_PRIM,
        sizes=1.0,
        positions=[[TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z - TABLE_HEIGHT / 2]],
        scales=[[TABLE_SIZE_XY[0], TABLE_SIZE_XY[1], TABLE_HEIGHT]],
        colors="saddlebrown",
    )
    GeomPrim(table.paths, apply_collision_apis=True)

    pedestal = Cube(
        PEDESTAL_PRIM,
        sizes=1.0,
        positions=[[0.0, 0.0, TABLE_TOP_Z - TABLE_HEIGHT / 2]],
        scales=[[0.25, 0.25, TABLE_HEIGHT]],
        colors="dimgray",
    )
    GeomPrim(pedestal.paths, apply_collision_apis=True)

    # Hộp cần gắp: có RigidBody (chịu trọng lực, bị đẩy/kẹp) + collision + vật liệu ma sát cao.
    box = Cube(
        BOX_PRIM,
        sizes=1.0,
        positions=[[BOX_START_XY[0], BOX_START_XY[1], TABLE_TOP_Z + BOX_SIZE / 2]],
        scales=[[BOX_SIZE, BOX_SIZE, BOX_SIZE]],
        colors="orange",
    )
    RigidPrim(box.paths, masses=[BOX_MASS])
    box_geom = GeomPrim(box.paths, apply_collision_apis=True)
    box_geom.apply_physics_materials(
        RigidBodyMaterial("/World/materials/box_rubber", static_frictions=BOX_FRICTION, dynamic_frictions=BOX_FRICTION)
    )

    # Nạp robot. ur.usda đã có base_joint cố định robot vào world nên robot không rơi.
    stage_utils.add_reference_to_stage(usd_path=str(args.robot_dir / "ur.usda"), path=ROBOT_PRIM)

    await omni.kit.app.get_app().next_update_async()
    set_camera_view(eye=[1.6, 1.2, 0.9], target=[0.45, 0.15, 0.1], camera_prim_path="/OmniverseKit_Persp")


# =====================================================================================
# PART 4 + 5. CUMOTION + STATE MACHINE
# =====================================================================================
class BoxPickPlace:
    """Gắp hộp trên bàn rồi đặt sang vị trí khác.

    Các pha (mỗi pha chạy tới khi hội tụ hoặc hết số frame cho phép):
        0  Pre-grasp  — đưa gripper lên trên hộp
        1  Approach   — hạ xuống độ cao gắp
        2  Grasp      — đóng gripper
        3  Lift       — nâng hộp lên
        4  Transport  — mang tới trên vị trí đặt
        5  Lower      — hạ xuống
        6  Release    — mở gripper
        7  Retract    — rút tay lên
    """

    PHASES = ("Pre-grasp", "Approach", "Grasp", "Lift", "Transport", "Lower", "Release", "Retract")
    PHASE_TIMEOUT = (300, 200, 120, 200, 300, 200, 120, 200)  # số frame tối đa cho từng pha
    MIN_STEPS = 30          # mỗi pha chạy ít nhất bấy nhiêu frame trước khi xét hội tụ
    POS_TOLERANCE = 0.01    # m - sai số vị trí wrist_3 để coi là tới đích
    GRIPPER_SETTLED = 0.002 # rad - góc ngón thay đổi ít hơn mức này sau 10 frame => ngón đã dừng
    WARMUP_FRAMES = 120

    def __init__(self) -> None:
        self.articulation: Articulation | None = None
        self._tool_prim: GeomPrim | None = None
        self._box = RigidPrim(BOX_PRIM)
        self._world_binding: mg.WorldBinding | None = None
        self._controller: RmpFlowController | None = None
        self._dof_index: dict[str, int] = {}
        self._gripper_idx: list[int] = []
        self._gripper_history: list[np.ndarray] = []
        self._phase = 0
        self._step = 0
        self._t = 0.0
        self._warmup = self.WARMUP_FRAMES

    # ---------------------------------------------------------------------------------
    # PART 4. KHỞI TẠO CUMOTION (gọi 1 lần, trước khi Play)
    # ---------------------------------------------------------------------------------
    async def setup(self) -> None:
        self.articulation = Articulation(ROBOT_PRIM)
        await omni.kit.app.get_app().next_update_async()
        dof_names = self.articulation.dof_names
        self._dof_index = {name: i for i, name in enumerate(dof_names)}
        missing = [n for n in ARM_JOINTS + list(GRIPPER_OPEN) if n not in self._dof_index]
        if missing:
            raise RuntimeError(f"Articulation thiếu khớp {missing}. Các DOF hiện có: {dof_names}")
        self._gripper_idx = [self._dof_index[n] for n in GRIPPER_OPEN]

        # Tư thế mặc định khi bấm Play/reset: arm ở HOME, gripper MỞ. Đặt trước khi Play để robot
        # không bao giờ xuất hiện ở tư thế duỗi ngang (sẽ đâm vào bàn và hất văng hộp).
        self.articulation.set_default_state(dof_positions=[self._initial_positions()])

        # (4a) World cho cuMotion: quét mọi prim có collision quanh robot và đăng ký làm vật cản,
        #      TRỪ robot (tự va chạm đã có trong XRDF), hộp (ta muốn chạm vào nó) và bệ (robot đứng trên đó).
        robot_pos, robot_ori = self.articulation.get_world_poses()
        obstacles = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos.numpy()[0],
            search_box_minimum=[-3.0, -3.0, -3.0],
            search_box_maximum=[3.0, 3.0, 3.0],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[ROBOT_PRIM, BOX_PRIM, PEDESTAL_PRIM],
        )
        print(f"[cuMotion] Vật cản được theo dõi: {obstacles}")
        strategy = mg.ObstacleStrategy()  # Cube mặc định được biểu diễn dạng hộp (CUBE) chính xác
        for prim_type in (Mesh, Cone, Cylinder):
            strategy.set_default_configuration(prim_type, mg.ObstacleConfiguration("obb", 0.01))
        self._world_binding = mg.WorldBinding(
            world_interface=CumotionWorldInterface(),
            obstacle_strategy=strategy,
            tracked_prims=obstacles,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self._world_binding.initialize()
        # cuMotion làm việc trong hệ toạ độ gốc robot => cho nó biết robot đang đặt ở đâu trong world.
        self._world_binding.get_world_interface().update_world_to_robot_root_transforms(poses=(robot_pos, robot_ori))
        self._world_binding.synchronize_transforms()

        # (4b) Mô hình động học robot: URDF (chuỗi khớp) + XRDF (c-space 6 khớp arm, tool frame,
        #      collision sphere). Gripper KHÔNG nằm trong c-space => cuMotion chỉ giải IK cho 6 khớp arm.
        cumotion_robot = load_cumotion_robot(
            directory=args.robot_dir, urdf_filename="robot.urdf", xrdf_filename="robot.xrdf"
        )
        if TOOL_FRAME not in cumotion_robot.robot_description.tool_frame_names():
            raise RuntimeError(f"robot.xrdf không có tool frame '{TOOL_FRAME}'")

        # (4c) RMPflow: bộ điều khiển phản ứng (reactive). Mỗi frame nó nhận target Cartesian của
        #      tool frame và trả về vị trí/vận tốc khớp kế tiếp — tức là IK + tránh va chạm + giới hạn
        #      khớp được giải đồng thời. Tham số đọc từ ur/rmp_flow.yaml.
        self._controller = RmpFlowController(
            cumotion_robot=cumotion_robot,
            cumotion_world_interface=self._world_binding.get_world_interface(),
            robot_joint_space=dof_names,
            robot_site_space=[TOOL_FRAME],
            tool_frame=TOOL_FRAME,
        )
        # cspace_target_rmp kéo khớp về tư thế lúc reset; giảm trọng số để nó không cản target Cartesian.
        config = self._controller.get_rmp_flow_config()
        config.set_param("cspace_target_rmp/metric_scalar", 1.0)
        config.set_param("joint_velocity_cap_rmp/max_velocity", RMP_MAX_JOINT_VELOCITY)

    def after_play(self) -> None:
        """Gọi sau khi Play: lấy link của tool frame, chỉnh drive, đặt robot về home."""
        # Dùng RigidPrim (đọc pose từ physics) chứ KHÔNG dùng GeomPrim: GeomPrim đọc transform USD,
        # vốn không được cập nhật khi mô phỏng chạy => luôn trả về pose lúc load.
        names = self.articulation.link_names
        self._tool_prim = RigidPrim(self.articulation.link_paths[0][names.index(TOOL_FRAME)])

        wrist_idx = [self._dof_index[n] for n in ARM_JOINTS[3:]]
        self.articulation.set_dof_max_efforts([WRIST_MAX_EFFORT] * 3, dof_indices=wrist_idx)

        # Position target chỉ có tác dụng khi khớp có stiffness > 0.
        stiffness, damping = self.articulation.get_dof_gains(dof_indices=self._gripper_idx)
        stiffness = stiffness.numpy()[0]
        print(f"[gripper] stiffness = {np.round(stiffness, 3)}, damping = {np.round(damping.numpy()[0], 3)}")
        if np.any(stiffness < 1e-3):
            print(f"[gripper] Có khớp ngón stiffness = 0 -> gán stiffness={GRIPPER_FALLBACK_STIFFNESS}")
            self.articulation.set_dof_gains(
                stiffnesses=[GRIPPER_FALLBACK_STIFFNESS] * len(self._gripper_idx),
                dampings=[GRIPPER_FALLBACK_DAMPING] * len(self._gripper_idx),
                dof_indices=self._gripper_idx,
            )

        init = self._initial_positions()
        self.articulation.set_dof_positions([init])
        self.articulation.set_dof_position_targets([init])

    def reset(self) -> None:
        self._phase, self._step, self._t = 0, 0, 0.0
        self._warmup = self.WARMUP_FRAMES

    # ---------------------------------------------------------------------------------
    # Tiện ích
    # ---------------------------------------------------------------------------------
    def _initial_positions(self) -> list[float]:
        q = [0.0] * len(self._dof_index)
        for name, value in zip(ARM_JOINTS, ARM_HOME):
            q[self._dof_index[name]] = value
        for name, value in GRIPPER_OPEN.items():
            q[self._dof_index[name]] = value
        return q

    def _tool_target(self) -> np.ndarray:
        """Vị trí mục tiêu (world) của mặt bích wrist_3 cho pha hiện tại.

        Ta muốn đầu ngón cách mặt bàn GRASP_CLEARANCE khi gắp/đặt, nên:
            z_bích = z_mặt_bàn + GRASP_CLEARANCE + FINGERTIP_OFFSET
        Khi đặt, cộng thêm 5 mm để hộp không bị ép xuống mặt bàn.
        """
        z_grasp = TABLE_TOP_Z + GRASP_CLEARANCE + FINGERTIP_OFFSET
        z_place = z_grasp + 0.005
        (bx, by), (px, py) = BOX_START_XY, BOX_PLACE_XY
        targets = (
            (bx, by, z_grasp + LIFT_HEIGHT),  # 0 Pre-grasp
            (bx, by, z_grasp),                # 1 Approach
            (bx, by, z_grasp),                # 2 Grasp (đứng yên)
            (bx, by, z_grasp + LIFT_HEIGHT),  # 3 Lift
            (px, py, z_place + LIFT_HEIGHT),  # 4 Transport
            (px, py, z_place),                # 5 Lower
            (px, py, z_place),                # 6 Release (đứng yên)
            (px, py, z_place + LIFT_HEIGHT),  # 7 Retract
        )
        return np.array(targets[self._phase], dtype=np.float32)

    def _setpoint(self, position: np.ndarray) -> mg.RobotState:
        """Đóng gói target Cartesian (vị trí + hướng của tool frame) cho RMPflow."""
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=[TOOL_FRAME],
                positions=([TOOL_FRAME], wp.array([position.tolist()], dtype=wp.float32)),
                orientations=([TOOL_FRAME], wp.array([DOWN_ORIENTATION.tolist()], dtype=wp.float32)),
            )
        )

    def _measured_state(self) -> mg.RobotState:
        """Trạng thái khớp đo từ mô phỏng (feedback cho RMPflow)."""
        names = self.articulation.dof_names
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self.articulation.get_dof_positions()),
                velocities=(names, self.articulation.get_dof_velocities()),
            )
        )

    def set_gripper(self, config: dict[str, float]) -> None:
        values = wp.array([config[n] for n in GRIPPER_OPEN], dtype=wp.float32)
        self.articulation.set_dof_position_targets(values, dof_indices=self._gripper_idx)

    def _gripper_settled(self) -> bool:
        """Ngón coi là đã dừng khi góc các khớp gần như không đổi trong 10 frame gần nhất."""
        q = self.articulation.get_dof_positions().numpy()[0][self._gripper_idx]
        self._gripper_history = (self._gripper_history + [q])[-10:]
        if len(self._gripper_history) < 10:
            return False
        return bool(np.abs(self._gripper_history[-1] - self._gripper_history[0]).max() < self.GRIPPER_SETTLED)

    def _tool_error(self) -> float:
        pos = self._tool_prim.get_world_poses()[0].numpy()[0]
        return float(np.linalg.norm(pos - self._tool_target()))

    def box_position(self) -> np.ndarray:
        return self._box.get_world_poses()[0].numpy()[0]

    def box_height(self) -> float:
        return float(self.box_position()[2])

    def done(self) -> bool:
        return self._phase >= len(self.PHASES)

    # ---------------------------------------------------------------------------------
    # PART 5. STATE MACHINE — gọi mỗi frame vật lý
    # ---------------------------------------------------------------------------------
    def step(self) -> None:
        if self.done():
            return

        # (5a) Warm-up: giữ robot ở home + gripper mở cho tới khi ổn định.
        if self._warmup > 0:
            init = np.array(self._initial_positions())
            self.articulation.set_dof_position_targets([init.tolist()])
            q = self.articulation.get_dof_positions().numpy()[0]
            self._warmup = 0 if np.abs(q - init).max() < 0.02 else self._warmup - 1
            return

        # (5b) Đầu mỗi pha: in nhãn; với các pha bắt đầu chuyển động mới (0, 3, 7) thì reset
        #      RMPflow để tích phân bắt đầu từ trạng thái khớp đo được, tránh giật.
        if self._step == 0:
            print(f"Pha {self._phase}: {self.PHASES[self._phase]}  -> target {np.round(self._tool_target(), 3)}")
            if self._phase in (0, 3, 7):
                if not self._controller.reset(self._measured_state(), self._setpoint(self._tool_target()), t=0.0):
                    raise RuntimeError("RmpFlowController.reset thất bại")
                self._t = 0.0

        # (5c) Hành động của pha: gripper (2, 6) hoặc cánh tay (các pha còn lại).
        if self._phase == 2:
            self.set_gripper(GRIPPER_CLOSED)
        elif self._phase == 6:
            self.set_gripper(GRIPPER_OPEN)
        else:
            # Đồng bộ vị trí vật cản (nếu chúng di chuyển) rồi hỏi RMPflow lệnh khớp kế tiếp.
            wi = self._world_binding.get_world_interface()
            wi.update_world_to_robot_root_transforms(self.articulation.get_world_poses())
            self._world_binding.synchronize_transforms()
            desired = self._controller.forward(self._measured_state(), self._setpoint(self._tool_target()), self._t)
            if desired is not None and desired.joints.positions is not None:
                # Chỉ gửi target cho 6 khớp arm (position_indices); gripper giữ target riêng.
                self.articulation.set_dof_position_targets(
                    positions=desired.joints.positions, dof_indices=desired.joints.position_indices
                )

        self._t += PHYSICS_DT
        self._step += 1

        # (5d) Điều kiện chuyển pha.
        converged = False
        if self._step >= self.MIN_STEPS:
            if self._phase in (2, 6):
                converged = self._gripper_settled()  # ngón đã dừng: chạm hộp (đóng) hoặc tới vị trí mở
            else:
                converged = self._tool_error() < self.POS_TOLERANCE
        timeout = self._step >= self.PHASE_TIMEOUT[self._phase]
        if converged or timeout:
            if timeout and not converged:
                print(f"  Pha {self._phase} hết giờ (sai số {self._tool_error():.3f} m)")
            if self._phase == 3:
                lifted = self.box_height() - (TABLE_TOP_Z + BOX_SIZE / 2)
                print(f"  Hộp được nâng {lifted * 100:.1f} cm -> {'GẮP THÀNH CÔNG' if lifted > 0.05 else 'GẮP TRƯỢT'}")
            self._phase += 1
            self._step = 0
            self._gripper_history = []
            if self.done():
                print(f"Xong. Hộp đang ở {np.round(self.box_position(), 3)} (đích: {BOX_PLACE_XY})")


# =====================================================================================
# PART 6. VÒNG LẶP MÔ PHỎNG
# =====================================================================================
def run_gripper_test(task: BoxPickPlace, app: SimulationApp) -> None:
    """Đóng/mở gripper mỗi 3 giây và in góc khớp — dùng để kiểm tra dấu/giới hạn khớp ngón."""
    closed, frame = False, 0
    while app.is_running():
        if app_utils.is_playing():
            if frame % 180 == 0:
                closed = not closed
                task.set_gripper(GRIPPER_CLOSED if closed else GRIPPER_OPEN)
                print("ĐÓNG" if closed else "MỞ")
            if frame % 60 == 59:
                q = task.articulation.get_dof_positions().numpy()[0]
                print({n: round(float(q[task._dof_index[n]]), 3) for n in GRIPPER_OPEN})
            frame += 1
            if args.test and frame >= 360:
                return
        app.update()


def main(app: SimulationApp) -> None:
    SimulationManager.setup_simulation(dt=PHYSICS_DT, device=args.device)

    app.run_coroutine(build_scene())
    task = BoxPickPlace()
    app.run_coroutine(task.setup())
    app.update()

    app_utils.play()
    app.update()
    task.after_play()

    if args.gripper_test:
        run_gripper_test(task, app)
        return

    needs_reset = False
    while app.is_running():
        app.update()
        if app_utils.is_playing() and SimulationManager.is_simulating():
            if needs_reset:  # người dùng vừa bấm Stop rồi Play lại
                task.after_play()
                task.reset()
                needs_reset = False
            task.step()
            if args.test and task.done():
                break
        elif not app_utils.is_playing():
            needs_reset = True


if __name__ == "__main__":
    try:
        main(simulation_app)
    except Exception:
        import traceback

        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nThoát...")
    finally:
        simulation_app.close()
