"""jacobian_logger: log q, qd and the tool0 Jacobian while a segment (tramo) is active.

Segment control
  * topic /tramo (std_msgs/String): "4B", "4D", ... starts a new file,
    "" stops logging.
  * parameter `tramo`: if non-empty, logging starts immediately with that name.

For every /joint_states message while a tramo is active, one row is written to
  <out_dir>/jac_<tramo>.csv        (out_dir default: ~/tap02_results)

Columns (Jacobians flattened column-major, k = 6*col + row, like the old C++ node):
  t, q1..q6,
  Jm0..Jm35   MoveIt 2 (moveit_py)        -> NaN if moveit_py is not available
  Jk0..Jk35   KDL (PyKDL)                 -> NaN if PyKDL is not available
  Jn0..Jn35   NumPy, built from the URDF  (always available, reference)
  qd1..qd6    joint velocities from /joint_states (NaN if not published)
  vx vy vz wx wy wz   tool0 twist = Jn @ qd (base_link frame, ref. point tool0)
  w_manip     Yoshikawa manipulability sqrt(det(J J^T))
  sigma_min   smallest singular value of Jn (-> 0 near a singularity)
  cond        condition number sigma_max / sigma_min
All Jacobians are 6x6 [v; w], expressed in base_link, reference point = tool0 origin.
"""
import csv
import math
import os
import traceback

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from lrmate_analysis.kinematics import KdlJacobian, MoveItJacobian, UrdfChain


def flat(J):
    """Column-major flatten (same order as the previous C++ logger)."""
    return J.flatten(order='F').tolist()


class JacobianLogger(Node):

    def __init__(self):
        super().__init__('jacobian_logger')
        self.group = self.declare_parameter('group', 'manipulator').value
        self.base = self.declare_parameter('base_link', 'base_link').value
        self.tip = self.declare_parameter('tip_link', 'tool0').value
        self.out_dir = os.path.expanduser(
            self.declare_parameter('out_dir', '~/tap02_results').value)
        start_tramo = self.declare_parameter('tramo', '').value
        urdf = self.declare_parameter('robot_description', '').value
        srdf = self.declare_parameter('robot_description_semantic', '').value
        if not urdf:
            raise RuntimeError('robot_description is empty: start the node with the launch '
                               'file (ros2 launch lrmate_analysis jacobian_logger.launch.py)')
        os.makedirs(self.out_dir, exist_ok=True)

        # --- Reference implementation (NumPy) ---------------------------------
        self.chain = UrdfChain(urdf, self.base, self.tip)
        self.names = self.chain.names
        self.n = len(self.names)

        # --- KDL -------------------------------------------------------------
        self.kdl = None
        try:
            self.kdl = KdlJacobian(self.chain)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'PyKDL not available, Jk columns will be NaN ({e})')

        # --- MoveIt 2 ----------------------------------------------------------
        self.moveit = None
        if srdf:
            try:
                self.moveit = MoveItJacobian(urdf, srdf, self.group, self.tip, self.names)
            except ImportError as e:
                self.get_logger().error(
                    f'moveit_py is not installed ({e}). Install it with '
                    '"sudo apt install ros-jazzy-moveit-py". Jm columns will be NaN.')
            except Exception:  # noqa: BLE001
                self.get_logger().error('MoveIt Jacobian disabled, Jm columns will be NaN:\n'
                                        + traceback.format_exc())
        else:
            self.get_logger().warn('robot_description_semantic empty, Jm columns will be NaN')

        self.file = None
        self.writer = None
        self.tramo = ''
        self.stats = None

        self.create_subscription(JointState, 'joint_states', self.on_joints, 50)
        self.create_subscription(String, 'tramo', self.on_tramo, 10)
        self.get_logger().info(
            f'Ready: chain {self.base}->{self.tip} ({self.n} joints: {", ".join(self.names)}) | '
            f'MoveIt: {"yes" if self.moveit else "no"} | KDL: {"yes" if self.kdl else "no"} | '
            f'output: {self.out_dir}')
        if start_tramo:
            self.start(start_tramo)

    # ------------------------------------------------------------------ files
    def header(self):
        h = ['t'] + [f'q{i + 1}' for i in range(self.n)]
        for tag in ('Jm', 'Jk', 'Jn'):
            h += [f'{tag}{k}' for k in range(6 * self.n)]
        h += [f'qd{i + 1}' for i in range(self.n)]
        h += ['vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'w_manip', 'sigma_min', 'cond']
        return h

    def start(self, tramo):
        self.stop()
        self.tramo = tramo
        path = os.path.join(self.out_dir, f'jac_{tramo}.csv')
        self.file = open(path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header())
        self.stats = {'rows': 0, 'smin': math.inf, 'dk': 0.0, 'dm': 0.0}
        self.get_logger().info(f'Logging tramo {tramo} -> {path}')

    def stop(self):
        if self.file is None:
            return
        self.file.close()
        s = self.stats
        msg = (f'Tramo {self.tramo} closed: {s["rows"]} rows, '
               f'min sigma_min = {s["smin"]:.4g}')
        if self.kdl:
            msg += f', max|Jk-Jn| = {s["dk"]:.2e}'
        if self.moveit:
            msg += f', max|Jm-Jn| = {s["dm"]:.2e}'
        self.get_logger().info(msg)
        self.file = self.writer = None
        self.tramo = ''

    # -------------------------------------------------------------- callbacks
    def on_tramo(self, msg):
        if msg.data:
            self.start(msg.data)
        else:
            self.stop()
            self.get_logger().info('Logging stopped')

    def on_joints(self, msg):
        if self.writer is None:
            return
        pos = dict(zip(msg.name, msg.position))
        if any(n not in pos for n in self.names):
            return                                          # incomplete message
        vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        q = np.array([pos[n] for n in self.names])
        qd = np.array([vel.get(n, math.nan) for n in self.names])

        Jn = self.chain.jacobian(q)
        nan = np.full((6, self.n), math.nan)
        Jk = self.kdl.jacobian(q) if self.kdl else nan
        Jm = nan
        if self.moveit:
            try:
                Jm = self.moveit.jacobian(pos)
            except Exception:  # noqa: BLE001
                self.get_logger().error('MoveIt Jacobian failed, disabling it:\n'
                                        + traceback.format_exc())
                self.moveit = None

        sv = np.linalg.svd(Jn, compute_uv=False)
        w = float(np.sqrt(max(np.linalg.det(Jn @ Jn.T), 0.0)))
        smin = float(sv[-1])
        cond = float(sv[0] / smin) if smin > 1e-12 else math.inf
        twist = Jn @ qd

        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9

        row = [t] + q.tolist() + flat(Jm) + flat(Jk) + flat(Jn) + qd.tolist() \
            + twist.tolist() + [w, smin, cond]
        self.writer.writerow([f'{v:.10g}' for v in row])

        s = self.stats
        s['rows'] += 1
        s['smin'] = min(s['smin'], smin)
        if self.kdl:
            s['dk'] = max(s['dk'], float(np.abs(Jk - Jn).max()))
        if self.moveit:
            s['dm'] = max(s['dm'], float(np.abs(Jm - Jn).max()))


def main(args=None):
    rclpy.init(args=args)
    node = JacobianLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
