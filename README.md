# TAP02 – Pick-and-place cell with ROS 2 Jazzy + MoveIt 2 (FANUC LR Mate 200iD)

## Packages
| Package | Content |
|---|---|
| `lrmate200id_support` | URDF/Xacro + meshes (tool0 = flange face, Z out of the flange) |
| `lrmate200id_moveit_config` | Own MoveIt 2 config made with the Setup Assistant (group `manipulator`, KDL, OMPL in `config/ompl_planning.yaml`) |
| `pose_manager_pkg` | `cell_layout.py` (all scene/pose numbers), `moveit_client.py` (rclpy client for move_group), node `manage_poses` (Parts 2–3) |
| `scene_builder_pkg` | node `spawn_scene` (PlanningScene: piece, pick table, pole, deposit table) |
| `planner_execution_pkg` | `compare_planners` (4A/4C), `fine_approach` (4B/4D, cubic vs quintic), `pick_place_cycle` (full cycle) |

## Build
```bash
cd ~/ws_lrmate200id
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

## Run (each command in its own terminal, after `source install/setup.bash`)
```bash
# 0. MoveIt + RViz
ros2 launch lrmate200id_moveit_config demo.launch.py

# 1. Collision scene
ros2 run scene_builder_pkg spawn_scene

# 2. Parts 2-3: HOME transform + IK of pick/place (writes ~/tap02_results/key_joints.yaml)
ros2 run pose_manager_pkg manage_poses

# 3. Part 4A: planner comparison (plan only, 10 trials per planner)
ros2 run planner_execution_pkg compare_planners --ros-args -p segment:=4A -p trials:=10

# 4. Part 4B: cubic vs quintic fine approach at the pick station (moves the robot)
ros2 run planner_execution_pkg fine_approach --ros-args -p station:=pick

# 5. Full cycle 4A -> 4B -> 4C -> 4D for the video
ros2 run planner_execution_pkg pick_place_cycle --ros-args -p planner:=RRTConnectkConfigDefault
```
Plots and CSV files are written to `~/tap02_results/`.
