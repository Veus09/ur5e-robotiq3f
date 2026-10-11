# UR5e + Robotiq 3F cho Isaac Sim

Asset robot UR5e được tích hợp với gripper Robotiq 3F, dùng cho mô phỏng thao tác **pick and place** trong NVIDIA Isaac Sim.

Repo này cung cấp model USD, mesh, URDF/XRDF và cấu hình RMPflow để dùng làm robot tùy chỉnh trong các ví dụ motion planning của Isaac Sim.

## Nội dung repo

```text
ur/
├── ur.usda                 # Stage USD chính của robot
├── robot.urdf              # Mô hình kinematic cho PINK/cuMotion
├── robot.xrdf              # Cấu hình cuMotion: c-space, tool frame, collision
├── rmp_flow.yaml           # Tham số cuMotion RMPflow
├── meshes/                 # Mesh và material của UR5e
└── payloads/               # USD payload, physics, materials, gripper
```

### Cấu hình robot

- Robot: UR5e, 6 khớp chính
- Gripper: Robotiq 3F, 3 ngón
- Tool frame hiện tại: `wrist_3_link`
- C-space của cuMotion: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`
- File cấu hình chính cho cuMotion: `ur/robot.urdf`, `ur/robot.xrdf`, `ur/rmp_flow.yaml`

## Yêu cầu

- NVIDIA Isaac Sim 6.0.0
- GPU NVIDIA và driver tương thích Isaac Sim
- Các extension cần thiết:
  - `isaacsim.robot_motion.cumotion`
  - `isaacsim.robot_motion.experimental`
  - `isaacsim.core.prims`
  - `isaacsim.core.api`
- Python đi kèm Isaac Sim (`python.sh`), không dùng Python hệ thống để chạy các ví dụ Isaac Sim.

## Lấy repo

```bash
git clone https://github.com/Veus09/ur5e-robotiq3f.git
cd ur5e-robotiq3f
```

## Mở robot trong Isaac Sim

1. Mở Isaac Sim 6.0.0.
2. Tạo một stage mới.
3. Chọn **File > Open** và mở file `ur/ur.usda`.
4. Kiểm tra robot đã xuất hiện với đủ UR5e và Robotiq 3F.
5. Tạo hoặc mở một scene có mặt bàn và vật thể hình hộp để thử nghiệm.
6. Kiểm tra Physics Scene, articulation root và collision geometry trước khi chạy controller.

Có thể mở stage từ terminal bằng Python launcher của Isaac Sim:

```bash
cd /path/to/isaac-sim
./python.sh -m isaacsim --no-window \
  --/persistent/isaac/asset_root/default=/path/to/isaac-sim/assets \
  /path/to/ur5e-robotiq3f/ur/ur.usda
```

Nếu chỉ cần giao diện Isaac Sim, mở ứng dụng theo cách cài đặt thông thường rồi dùng **File > Open** là đủ.

## Liên kết physics của gripper

`ur/ur.usda` hiện có một sublayer tham chiếu tới physics của gripper bằng đường dẫn tương đối bên ngoài repo:

```text
../../robotiq-3f-gripper_articulated_redo/robotiq-3f-gripper_articulated/payloads/Physics/physics.usda
```

Khi clone repo ở máy khác, cần bảo đảm dependency này tồn tại đúng vị trí tương đối, hoặc sửa tham chiếu trong `ur/ur.usda` thành đường dẫn tới bản physics Robotiq 3F đang sử dụng. Nếu Isaac Sim báo lỗi không tìm thấy layer, đây là vị trí cần kiểm tra đầu tiên.

## Dùng với cuMotion RMPflow

`robot.xrdf` đã khai báo c-space, tool frame và collision sphere; `rmp_flow.yaml` chứa tham số RMPflow. Khi chạy ví dụ pick-and-place của Isaac Sim, trỏ `--xrdf-dir` tới thư mục `ur`:

```bash
cd /path/to/isaac-sim
./python.sh standalone_examples/tutorials/manipulation/tutorial_9_pick_place_cumotion.py \
  --xrdf-dir /path/to/ur5e-robotiq3f/ur \
  --urdf robot.urdf \
  --xrdf robot.xrdf
```

Nếu script cho phép chọn file RMPflow riêng, dùng:

```text
/path/to/ur5e-robotiq3f/ur/rmp_flow.yaml
```

Trong scene và controller, cần bảo đảm tên articulation và tool frame khớp với asset. Tool frame mặc định của repo là `wrist_3_link`; nếu controller cần TCP ở đầu gripper, hãy thêm một frame TCP vào XRDF và dùng đúng tên frame đó trong controller.

## Điều khiển Robotiq 3F

Tutorial Isaac Sim dùng gripper 2F-140 làm ví dụ. Robotiq 3F trong repo có các nhóm joint sau:

```text
palm_finger_1_joint
palm_finger_2_joint
finger_1_joint_1 ... finger_1_joint_3
finger_2_joint_1 ... finger_2_joint_3
finger_middle_joint_1 ... finger_middle_joint_3
```

Không nên dùng nguyên tên joint `finger_joint` của ví dụ 2F-140. Hãy lấy danh sách DOF thực tế từ articulation rồi gửi target cho các joint của Robotiq 3F:

```python
from isaacsim.core.api.robots import Robot

robot = Robot(prim_path="/World/ur")
robot.initialize()
print(robot.dof_names)
```

Khi thiết kế controller, nên tách điều khiển thành hai phần:

- Arm: 6 joint UR5e do cuMotion RMPflow hoặc PINK điều khiển.
- Gripper: các joint Robotiq 3F được điều khiển bằng position targets, theo một cấu hình mở và một cấu hình đóng phù hợp với vật thể.

Đầu tiên hãy chạy thử mở/đóng gripper không có vật thể. Sau đó mới bật motion planning và thao tác gắp để dễ phân biệt lỗi joint drive với lỗi planner.

## Quy trình pick and place

Workflow tương ứng với Tutorial 9 của Isaac Sim:

1. Reset robot về `default_joint_positions` trong `robot.xrdf`.
2. Mở gripper Robotiq 3F.
3. Đưa end-effector tới vị trí phía trên vật thể.
4. Hạ xuống vị trí grasp.
5. Đóng Robotiq 3F và chờ các joint ổn định.
6. Nâng vật thể lên khỏi mặt bàn.
7. Di chuyển tới vị trí đặt.
8. Hạ vật thể xuống.
9. Mở gripper để nhả vật thể.
10. Đưa robot về vị trí an toàn hoặc vị trí ban đầu.

Nên kiểm tra từng phase bằng target Cartesian riêng trước khi ghép thành state machine hoàn chỉnh. Khi bật obstacle avoidance, đăng ký mặt bàn và các vật cản vào world interface của cuMotion.

## Tham khảo chính thức

- [Isaac Sim 6.0 - Tutorial 9: Pick and Place Example](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_pickplace_example.html)
- [Isaac Sim 6.0 - Setup a Manipulator](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_import_assemble_manipulator.html)
- [Isaac Sim 6.0 - Configure a Manipulator](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_configure_manipulator.html)
- [Isaac Sim 6.0 - Generate Robot Configuration](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/tutorial_generate_robot_config.html)

## Ghi chú

Repo hiện tập trung vào asset và robot configuration. Các script state machine pick-and-place cần được chạy từ bộ ví dụ Isaac Sim hoặc đặt trong một project riêng, sau đó trỏ tới thư mục `ur` bằng `--xrdf-dir`.