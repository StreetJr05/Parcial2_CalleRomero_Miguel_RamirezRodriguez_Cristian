"""Kinematics helpers for the Jacobian logger.

Three independent ways of getting the geometric Jacobian of `tip` w.r.t.
`base`, all as a 6xN matrix [v; w] (linear on top, angular below), expressed
in the base frame, with the reference point at the origin of `tip`:

* UrdfChain.jacobian()   pure NumPy, built from the URDF (always available)
* KdlJacobian            PyKDL ChainJntToJacSolver (needs python_orocos_kdl_vendor)
* MoveItJacobian         moveit_py RobotState.get_jacobian (needs moveit_py)
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


# ----------------------------------------------------------------------------
# URDF parsing + NumPy kinematics
# ----------------------------------------------------------------------------
def _floats(text, default):
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def rpy_to_matrix(r, p, y):
    """URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle(axis, q):
    """Rodrigues rotation about a unit axis."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class JointSpec:
    def __init__(self, name, jtype, xyz, rpy, axis):
        self.name = name
        self.type = jtype
        self.xyz = xyz
        self.rpy = rpy
        self.axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis
        self.T_origin = np.eye(4)
        self.T_origin[:3, :3] = rpy_to_matrix(*rpy)
        self.T_origin[:3, 3] = xyz

    @property
    def movable(self):
        return self.type in ('revolute', 'continuous', 'prismatic')


class UrdfChain:
    """Serial chain base -> tip parsed from a URDF string."""

    def __init__(self, urdf_xml, base, tip):
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for j in root.findall('joint'):
            origin = j.find('origin')
            axis = j.find('axis')
            by_child[j.find('child').get('link')] = (
                j.find('parent').get('link'),
                JointSpec(
                    j.get('name'), j.get('type'),
                    _floats(origin.get('xyz') if origin is not None else None, [0, 0, 0]),
                    _floats(origin.get('rpy') if origin is not None else None, [0, 0, 0]),
                    _floats(axis.get('xyz') if axis is not None else None, [1, 0, 0]),
                ))
        joints, link = [], tip
        while link != base:
            if link not in by_child:
                raise RuntimeError(f'URDF: no chain from {base} to {tip} (stuck at {link})')
            parent, spec = by_child[link]
            joints.append(spec)
            link = parent
        self.joints = joints[::-1]                      # base -> tip order
        self.active = [j for j in self.joints if j.movable]
        self.names = [j.name for j in self.active]

    def fk_and_jacobian(self, q):
        """Return (T_base_tip 4x4, J 6xN) for joint vector q (order = self.names)."""
        T = np.eye(4)
        z_axes, origins, kinds = [], [], []
        i = 0
        for j in self.joints:
            T = T @ j.T_origin                     # frame of the joint (before motion)
            if j.movable:
                z = T[:3, :3] @ j.axis             # joint axis in base frame
                z_axes.append(z)
                origins.append(T[:3, 3].copy())
                kinds.append(j.type)
                M = np.eye(4)
                if j.type == 'prismatic':
                    M[:3, 3] = j.axis * q[i]
                else:
                    M[:3, :3] = axis_angle(j.axis, q[i])
                T = T @ M
                i += 1
        p_e = T[:3, 3]
        J = np.zeros((6, len(self.active)))
        for k, (z, p, kind) in enumerate(zip(z_axes, origins, kinds)):
            if kind == 'prismatic':
                J[:3, k] = z
            else:
                J[:3, k] = np.cross(z, p_e - p)    # v contribution
                J[3:, k] = z                       # w contribution
        return T, J

    def jacobian(self, q):
        return self.fk_and_jacobian(q)[1]


# ----------------------------------------------------------------------------
# PyKDL backend (KDL chain built from the same parsed URDF; there is no
# kdl_parser_py in ROS 2, so we build the segments ourselves exactly like
# kdl_parser does in C++)
# ----------------------------------------------------------------------------
class KdlJacobian:
    def __init__(self, chain: UrdfChain):
        import PyKDL as kdl
        self.kdl = kdl
        fixed = getattr(kdl.Joint, 'Fixed') if hasattr(kdl.Joint, 'Fixed') \
            else getattr(kdl.Joint, 'None')
        kchain = kdl.Chain()
        for j in chain.joints:
            F = kdl.Frame(kdl.Rotation.RPY(*j.rpy), kdl.Vector(*j.xyz))
            if j.type in ('revolute', 'continuous'):
                kj = kdl.Joint(j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.RotAxis)
            elif j.type == 'prismatic':
                kj = kdl.Joint(j.name, F.p, F.M * kdl.Vector(*j.axis), kdl.Joint.TransAxis)
            else:
                kj = kdl.Joint(j.name, fixed)
            kchain.addSegment(kdl.Segment(j.name + '_seg', kj, F))
        self.n = kchain.getNrOfJoints()
        self.solver = kdl.ChainJntToJacSolver(kchain)
        self._chain = kchain                           # keep alive

    def jacobian(self, q):
        qa = self.kdl.JntArray(self.n)
        for i, v in enumerate(q):
            qa[i] = float(v)
        Jk = self.kdl.Jacobian(self.n)
        self.solver.JntToJac(qa, Jk)
        return np.array([[Jk[r, c] for c in range(self.n)] for r in range(6)])


# ----------------------------------------------------------------------------
# MoveIt 2 backend (moveit_py)
# ----------------------------------------------------------------------------
class MoveItJacobian:
    """MoveIt 2 Jacobian through moveit_py.

    Every moveit_py call that changed between releases is tried in more than
    one form, and the constructor computes one Jacobian as a self-test, so any
    problem shows up at startup with the real error message instead of as NaN.
    """

    def __init__(self, urdf_xml, srdf_xml, group, tip, joint_names):
        from moveit.core.robot_model import RobotModel
        from moveit.core.robot_state import RobotState
        tmp = tempfile.mkdtemp(prefix='jac_logger_')
        urdf_path = os.path.join(tmp, 'robot.urdf')
        srdf_path = os.path.join(tmp, 'robot.srdf')
        with open(urdf_path, 'w') as f:
            f.write(urdf_xml)
        with open(srdf_path, 'w') as f:
            f.write(srdf_xml)
        try:
            self.model = RobotModel(urdf_xml_path=urdf_path, srdf_xml_path=srdf_path)
        except TypeError:
            self.model = RobotModel(urdf_path, srdf_path)
        if not self.model.has_joint_model_group(group):
            raise RuntimeError(f'MoveIt model has no planning group "{group}"')
        self.state = RobotState(self.model)
        self.group, self.tip = group, tip
        self.names = list(joint_names)            # joint_1..joint_6 (same order as the group)
        self.jacobian({n: 0.1 for n in self.names})   # self-test: raises if anything is wrong

    def _set_positions(self, q_by_name):
        values = np.array([q_by_name[n] for n in self.names], float)
        try:
            self.state.set_joint_group_positions(self.group, values)
        except Exception:  # noqa: BLE001
            self.state.joint_positions = {n: float(q_by_name[n]) for n in self.names}
        self.state.update()

    def jacobian(self, q_by_name):
        self._set_positions(q_by_name)
        ref = np.zeros(3)
        attempts = (
            lambda: self.state.get_jacobian(self.group, self.tip, ref),
            lambda: self.state.get_jacobian(joint_model_group_name=self.group,
                                            link_name=self.tip,
                                            reference_point_position=ref),
            lambda: self.state.get_jacobian(self.group, ref),   # group tip == tool0 here
        )
        errors = []
        for attempt in attempts:
            try:
                J = np.asarray(attempt(), float)
            except Exception as e:  # noqa: BLE001
                errors.append(f'{type(e).__name__}: {e}')
                continue
            if J.shape != (6, len(self.names)):
                errors.append(f'unexpected Jacobian shape {J.shape}')
                continue
            return J
        raise RuntimeError('RobotState.get_jacobian failed: ' + ' | '.join(errors))
