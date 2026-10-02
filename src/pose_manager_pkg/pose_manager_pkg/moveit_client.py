"""Small MoveIt 2 client for Python (ROS 2 Jazzy) using only rclpy.

`moveit_commander` is a ROS 1 package and does not exist in ROS 2. Instead of
it, this helper talks directly to the services/actions that `move_group`
already offers (they are started by demo.launch.py):

  /compute_ik              moveit_msgs/srv/GetPositionIK
  /compute_fk              moveit_msgs/srv/GetPositionFK
  /compute_cartesian_path  moveit_msgs/srv/GetCartesianPath
  /apply_planning_scene    moveit_msgs/srv/ApplyPlanningScene
  /move_action             moveit_msgs/action/MoveGroup      (planning)
  /execute_trajectory      moveit_msgs/action/ExecuteTrajectory

All calls are blocking: call them from main(), never from inside a callback.
"""
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject,
                             Constraints, JointConstraint, MotionPlanRequest,
                             MoveItErrorCodes, PlanningScene, RobotState)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath,
                             GetPositionFK, GetPositionIK)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from pose_manager_pkg import cell_layout as cell

SUCCESS = MoveItErrorCodes.SUCCESS


def make_pose(position, orientation):
    p = Pose()
    p.position.x, p.position.y, p.position.z = [float(v) for v in position]
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = [float(v) for v in orientation]
    return p


def pose_to_matrix(pose):
    """geometry_msgs/Pose -> 4x4 homogeneous transform (numpy)."""
    x, y, z, w = (pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w)
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


class MoveItClient:
    def __init__(self, node: Node, group=cell.GROUP, eef_link=cell.EEF_LINK,
                 frame=cell.PLANNING_FRAME):
        self.node = node
        self.group = group
        self.eef_link = eef_link
        self.frame = frame
        self.joint_names = list(cell.JOINT_NAMES)
        self._last_js = None

        self.ik_cli = node.create_client(GetPositionIK, '/compute_ik')
        self.fk_cli = node.create_client(GetPositionFK, '/compute_fk')
        self.cart_cli = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene_cli = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.move_ac = ActionClient(node, MoveGroup, '/move_action')
        self.exec_ac = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        node.create_subscription(JointState, '/joint_states', self._js_cb, 10)

    # ------------------------------------------------------------ basics --
    def _js_cb(self, msg):
        self._last_js = msg

    def _wait(self, future, timeout=None):
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise RuntimeError('Timed out waiting for move_group')
        return future.result()

    def wait_for_servers(self, timeout=15.0):
        log = self.node.get_logger()
        for name, cli in [('/compute_ik', self.ik_cli), ('/compute_fk', self.fk_cli),
                          ('/compute_cartesian_path', self.cart_cli),
                          ('/apply_planning_scene', self.scene_cli)]:
            if not cli.wait_for_service(timeout_sec=timeout):
                raise RuntimeError(f'Service {name} not available. Is demo.launch.py running?')
        for name, ac in [('/move_action', self.move_ac),
                         ('/execute_trajectory', self.exec_ac)]:
            if not ac.wait_for_server(timeout_sec=timeout):
                raise RuntimeError(f'Action {name} not available. Is demo.launch.py running?')
        log.info('Connected to move_group.')

    def current_joints(self, timeout=5.0):
        """Latest arm joint positions from /joint_states (ordered joint_1..6)."""
        self._last_js = None
        end = self.node.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self._last_js is None and self.node.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._last_js is None:
            raise RuntimeError('No /joint_states received')
        m = dict(zip(self._last_js.name, self._last_js.position))
        return [m[j] for j in self.joint_names]

    def robot_state(self, joints=None):
        """RobotState diff: keeps attached objects, overrides joint values."""
        rs = RobotState()
        rs.is_diff = True
        if joints is not None:
            rs.joint_state.name = list(self.joint_names)
            rs.joint_state.position = [float(v) for v in joints]
        return rs

    # --------------------------------------------------------- IK and FK --
    def compute_ik(self, pose, seed, avoid_collisions=True, timeout=1.0):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.group
        r.ik_link_name = self.eef_link
        r.robot_state = self.robot_state(seed)
        r.avoid_collisions = avoid_collisions
        r.pose_stamped = PoseStamped()
        r.pose_stamped.header.frame_id = self.frame
        r.pose_stamped.pose = pose
        r.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        res = self._wait(self.ik_cli.call_async(req), timeout + 5.0)
        if res.error_code.val != SUCCESS:
            return None, res.error_code.val
        m = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [m[j] for j in self.joint_names], SUCCESS

    def compute_fk(self, joints, link=None):
        """Returns geometry_msgs/Pose of `link` (default tool0) in the planning frame."""
        req = GetPositionFK.Request()
        req.header.frame_id = self.frame
        req.fk_link_names = [link or self.eef_link]
        rs = RobotState()
        rs.joint_state.name = list(self.joint_names)
        rs.joint_state.position = [float(v) for v in joints]
        req.robot_state = rs
        res = self._wait(self.fk_cli.call_async(req), 5.0)
        if res.error_code.val != SUCCESS:
            raise RuntimeError(f'FK failed, code {res.error_code.val}')
        return res.pose_stamped[0].pose

    # ------------------------------------------------------ planning scene --
    def apply_scene(self, scene_msg):
        req = ApplyPlanningScene.Request()
        req.scene = scene_msg
        res = self._wait(self.scene_cli.call_async(req), 10.0)
        return res.success

    @staticmethod
    def collision_object(obj_id, kind, dims, center, frame=cell.PLANNING_FRAME,
                         operation=CollisionObject.ADD):
        co = CollisionObject()
        co.header.frame_id = frame
        co.id = obj_id
        co.operation = operation
        if operation == CollisionObject.ADD:
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX if kind == 'box' else SolidPrimitive.CYLINDER
            prim.dimensions = [float(d) for d in dims]
            co.primitives.append(prim)
            co.primitive_poses.append(make_pose(center, (0, 0, 0, 1)))
            # Jazzy also uses co.pose as the object frame; identity keeps it simple
            co.pose = make_pose((0, 0, 0), (0, 0, 0, 1))
        return co

    def attach(self, obj_id, link=None, touch_links=('link_6', 'tool0')):
        """Moves an existing world object onto the robot (grasp)."""
        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        aco = AttachedCollisionObject()
        aco.link_name = link or self.eef_link
        aco.object.id = obj_id
        aco.object.header.frame_id = self.frame
        aco.object.operation = CollisionObject.ADD   # no geometry -> take it from the world
        aco.touch_links = list(touch_links)
        ps.robot_state.attached_collision_objects.append(aco)
        return self.apply_scene(ps)

    def detach(self, obj_id, link=None):
        """Releases an attached object; MoveIt puts it back in the world where it is."""
        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.is_diff = True
        aco = AttachedCollisionObject()
        aco.link_name = link or self.eef_link
        aco.object.id = obj_id
        aco.object.operation = CollisionObject.REMOVE
        ps.robot_state.attached_collision_objects.append(aco)
        return self.apply_scene(ps)

    # ----------------------------------------------------------- planning --
    def plan_to_joints(self, goal_joints, start_joints=None, planner_id='',
                       pipeline_id='ompl', planning_time=5.0, attempts=1,
                       vel_scale=0.3, acc_scale=0.3, tolerance=1e-3):
        """Plans (does NOT execute). Returns (ok, RobotTrajectory, planning_time_s, code)."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = self.group
        req.pipeline_id = pipeline_id
        req.planner_id = planner_id
        req.num_planning_attempts = attempts
        req.allowed_planning_time = float(planning_time)
        req.max_velocity_scaling_factor = float(vel_scale)
        req.max_acceleration_scaling_factor = float(acc_scale)
        req.start_state = self.robot_state(start_joints)
        c = Constraints()
        for name, val in zip(self.joint_names, goal_joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = tolerance
            jc.tolerance_below = tolerance
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal.request = req
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        gh = self._wait(self.move_ac.send_goal_async(goal), 10.0)
        if not gh.accepted:
            return False, None, float('nan'), -1
        res = self._wait(gh.get_result_async(), planning_time + 30.0).result
        ok = res.error_code.val == SUCCESS
        return ok, res.planned_trajectory, res.planning_time, res.error_code.val

    def cartesian_path(self, waypoints, start_joints, max_step=0.01,
                       avoid_collisions=True):
        """Straight-line path through `waypoints` (list of Pose). Returns (traj, fraction)."""
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.group_name = self.group
        req.link_name = self.eef_link
        req.start_state = self.robot_state(start_joints)
        req.waypoints = list(waypoints)
        req.max_step = float(max_step)
        if hasattr(req, 'jump_threshold'):
            req.jump_threshold = 0.0          # 0 = disabled
        req.avoid_collisions = avoid_collisions
        res = self._wait(self.cart_cli.call_async(req), 30.0)
        return res.solution, res.fraction

    # ---------------------------------------------------------- execution --
    def execute(self, robot_trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_trajectory
        gh = self._wait(self.exec_ac.send_goal_async(goal), 10.0)
        if not gh.accepted:
            return False
        jt = robot_trajectory.joint_trajectory
        dur = 10.0
        if jt.points:
            t = jt.points[-1].time_from_start
            dur = t.sec + t.nanosec * 1e-9
        res = self._wait(gh.get_result_async(), dur + 30.0).result
        return res.error_code.val == SUCCESS


# ----------------------------------------------------------- utilities ----
def traj_arrays(robot_trajectory):
    """RobotTrajectory -> (t[N], q[N,6], qd[N,6] or None, qdd[N,6] or None)."""
    jt = robot_trajectory.joint_trajectory
    t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                  for p in jt.points])
    order = [jt.joint_names.index(j) for j in cell.JOINT_NAMES]
    q = np.array([[p.positions[i] for i in order] for p in jt.points])
    qd = qdd = None
    if jt.points and len(jt.points[0].velocities) == len(order):
        qd = np.array([[p.velocities[i] for i in order] for p in jt.points])
    if jt.points and len(jt.points[0].accelerations) == len(order):
        qdd = np.array([[p.accelerations[i] for i in order] for p in jt.points])
    return t, q, qd, qdd


def seconds_to_duration(t):
    ns = int(round(t * 1e9))
    return Duration(sec=ns // 1000000000, nanosec=ns % 1000000000)


def spawn_cell_scene(mi, logger=None, reset_only=False):
    """Detach the piece if needed, remove our objects, then add the whole cell."""
    mi.detach(cell.PIECE_ID)
    clear = PlanningScene()
    clear.is_diff = True
    for obj_id in cell.SCENE_OBJECTS:
        clear.world.collision_objects.append(
            MoveItClient.collision_object(obj_id, None, None, None,
                                          operation=CollisionObject.REMOVE))
    mi.apply_scene(clear)
    if reset_only:
        return True
    scene = PlanningScene()
    scene.is_diff = True
    for obj_id, (kind, dims, center) in cell.SCENE_OBJECTS.items():
        scene.world.collision_objects.append(
            MoveItClient.collision_object(obj_id, kind, dims, center))
        if logger:
            logger.info(f'  {obj_id:14s} {kind:8s} dims={dims} center={center}')
    return mi.apply_scene(scene)


def load_key_joints(path=cell.DEFAULT_JOINTS_FILE):
    import os
    import yaml
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise RuntimeError(f'{path} not found. Run first: ros2 run pose_manager_pkg manage_poses')
    with open(path) as f:
        return yaml.safe_load(f)['joints_rad']
