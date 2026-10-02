"""Parts 2 and 3: key poses, MoveIt IK, and homogeneous transforms for MATLAB.

What it does:
  1. Reads the pick / pre_pick / place / pre_place poses from cell_layout.py.
  2. Solves each one with MoveIt's IK (/compute_ik -> KDL), collision-aware,
     seeding every solve with the previous one so all poses share a branch.
  3. Verifies each solution with MoveIt FK (/compute_fk) and prints the 4x4
     transform world->tool0 (compare it against your DH model in MATLAB).
  4. Prints the HOME transform too (Part 2).
  5. Saves joints + transforms to ~/tap02_results/key_joints.yaml, which the
     motion nodes read, and prints ready-to-paste SRDF <group_state> lines.
  6. Publishes the poses as a PoseArray on /key_poses (add it in RViz).

Run AFTER the scene exists (scene_builder_pkg spawn_scene) so IK avoids it.
"""
import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from pose_manager_pkg import cell_layout as cell
from pose_manager_pkg.moveit_client import MoveItClient, make_pose, pose_to_matrix


class PoseManagerNode(Node):
    def __init__(self):
        super().__init__('pose_manager_node')
        self.declare_parameter('output_file', cell.DEFAULT_JOINTS_FILE)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PoseArray, '/key_poses', qos)
        self.mi = MoveItClient(self)

    def run(self):
        log = self.get_logger()
        self.mi.wait_for_servers()
        np.set_printoptions(precision=4, suppress=True)

        solutions = {'home': list(cell.HOME_JOINTS)}
        transforms = {}

        # ---- Part 2: HOME ------------------------------------------------
        T_home = pose_to_matrix(self.mi.compute_fk(cell.HOME_JOINTS))
        transforms['home'] = T_home
        log.info(f'HOME  T(world->tool0) =\n{T_home}')

        # ---- Part 3: IK of the key poses ---------------------------------
        for name, seed_name in cell.IK_ORDER:
            spec = cell.KEY_POSES[name]
            target = make_pose(spec['position'], spec['orientation'])
            q, code = self.mi.compute_ik(target, solutions[seed_name])
            if q is None:
                log.error(f'IK FAILED for {name} (MoveIt error code {code}). '
                          'Pose unreachable or in collision -> edit cell_layout.py')
                return False
            solutions[name] = q
            T = pose_to_matrix(self.mi.compute_fk(q))
            transforms[name] = T
            T_des = pose_to_matrix(target)
            pos_err = np.linalg.norm(T[:3, 3] - T_des[:3, 3])
            log.info(f'{name.upper()}: q [rad] = {np.round(q, 4).tolist()}\n'
                     f'  q [deg] = {np.round(np.degrees(q), 2).tolist()}\n'
                     f'  T(world->tool0) from MoveIt FK =\n{T}\n'
                     f'  position error vs requested pose = {pos_err * 1000:.3f} mm')

        self.save(solutions, transforms)
        self.print_srdf(solutions)
        self.publish(solutions)
        return True

    def save(self, solutions, transforms):
        path = os.path.expanduser(self.get_parameter('output_file').value)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {'joint_names': cell.JOINT_NAMES,
                'joints_rad': {k: [float(v) for v in q] for k, q in solutions.items()},
                'T_world_tool0': {k: np.round(T, 6).tolist() for k, T in transforms.items()}}
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.get_logger().info(f'Saved joints/transforms to {path}')

    def print_srdf(self, solutions):
        lines = ['Paste these in the Setup Assistant (Robot Poses tab) or in the SRDF:']
        for name in ('pre_pick', 'pre_place'):
            lines.append(f'<group_state name="{name.upper()}" group="{cell.GROUP}">')
            for j, v in zip(cell.JOINT_NAMES, solutions[name]):
                lines.append(f'    <joint name="{j}" value="{v:.4f}"/>')
            lines.append('</group_state>')
        self.get_logger().info('\n'.join(lines))

    def publish(self, solutions):
        msg = PoseArray()
        msg.header.frame_id = cell.PLANNING_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        for name in ('pre_pick', 'pick', 'pre_place', 'place'):
            msg.poses.append(self.mi.compute_fk(solutions[name]))
        self._msg = msg
        self.create_timer(1.0, self._republish)   # re-published so RViz always gets it
        self._republish()
        self.get_logger().info('Publishing /key_poses (PoseArray) -> add it in RViz.')

    def _republish(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._msg)


def main(args=None):
    rclpy.init(args=args)
    node = PoseManagerNode()
    try:
        if node.run():
            node.get_logger().info('Done. Publishing /key_poses for RViz (Ctrl+C to exit).')
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
