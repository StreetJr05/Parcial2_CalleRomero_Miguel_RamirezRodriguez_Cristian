"""Parts 4B / 4D: fine approach with cubic vs quintic time profiles.

For one station (pick or place):
  RED  (ida)      pre_X -> X     v <= 0.200 m/s, a <= 0.300 m/s^2
  ---- grasp (attach piece) at pick / release (detach) at place ----
  BLUE (retorno)  X -> pre_X     v <= 0.100 m/s, a <= 0.020 m/s^2

For each tramo and each profile (cubic, quintic):
  1. s(t) is designed through 3 via points (trajectory_profiles.py) and
     time-scaled to respect the limits.
  2. s(t) is sampled every `waypoint_dt` s -> Cartesian waypoints on the line.
  3. /compute_cartesian_path turns the waypoints into joint states (IK along
     the line, collision-checked).
  4. Re-parameterisation: MoveIt's own timing is discarded. Every joint point
     is sent through FK, its distance s along the line is measured, and it
     gets the time t at which OUR profile reaches that s. Joint velocities and
     accelerations are then finite differences over those times.
  5. Metrics + plots: Cartesian s/v/a/jerk, joint accelerations, tool speed
     obtained from FK vs commanded (preview of Part 5).
The better profile is executed (auto = continuous acceleration first, then lower
RMS joint jerk; or force it with profile:=cubic|quintic).

Outputs in output_dir (default ~/tap02_results): PNG plots + one CSV per
profile/tramo with t, s, v, a, q, qd, qdd, tool position and tool speed
(use the CSV in MATLAB for the Jacobian check: v_tool = J(q) * qd).
"""
import csv
import os

import matplotlib
import numpy as np
import rclpy
from rclpy.node import Node

from planner_execution_pkg.trajectory_profiles import design_profile
from pose_manager_pkg import cell_layout as cell
from pose_manager_pkg.moveit_client import (MoveItClient, load_key_joints, make_pose,
                                            seconds_to_duration, traj_arrays)

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

KINDS = ('cubic', 'quintic')
STATIONS = {'pick': ('pre_pick', 'pick'), 'place': ('pre_place', 'place')}


class FineApproach:
    """Reusable logic (also used by the full-cycle node)."""

    def __init__(self, node, mi, out_dir, via_fractions=(0.2, 0.5, 0.8),
                 waypoint_dt=0.05):
        self.node = node
        self.log = node.get_logger()
        self.mi = mi
        self.out_dir = os.path.expanduser(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.via = tuple(via_fractions)
        self.dt = waypoint_dt

    # ------------------------------------------------------------------
    def build(self, start_joints, goal_joints, limits, kind):
        """Cartesian straight line start->goal retimed with a `kind` profile."""
        start_pose = self.mi.compute_fk(start_joints)
        goal_pose = self.mi.compute_fk(goal_joints)
        p0 = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])
        p1 = np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z])
        quat = (start_pose.orientation.x, start_pose.orientation.y,
                start_pose.orientation.z, start_pose.orientation.w)
        length = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / length

        prof = design_profile(length, limits['v_max'], limits['a_max'], kind, self.via)
        T = prof.duration
        t_wp = np.append(np.arange(self.dt, T, self.dt), T)
        s_wp = prof.evaluate(t_wp)[0]
        waypoints = [make_pose(p0 + s * u, quat) for s in s_wp]

        traj, fraction = self.mi.cartesian_path(waypoints, start_joints,
                                                max_step=0.01, avoid_collisions=True)
        if fraction < 0.999:
            raise RuntimeError(f'computeCartesianPath only achieved {fraction * 100:.1f} % '
                               f'of the {kind} path (collision or IK failure on the line)')

        # ---- re-parameterisation with our profile ------------------------
        _, q, _, _ = traj_arrays(traj)
        P = np.array([[ps.position.x, ps.position.y, ps.position.z]
                      for ps in (self.mi.compute_fk(qi) for qi in q)])
        s_pts = np.clip((P - p0) @ u, 0.0, length)
        t_pts = prof.time_of_s(s_pts)
        t_pts[0], t_pts[-1] = 0.0, T
        idx = [0]                       # time stamps must be strictly increasing
        for i in range(1, len(t_pts)):
            if t_pts[i] - t_pts[idx[-1]] > 1e-3:
                idx.append(i)
        if idx[-1] != len(t_pts) - 1:   # always finish exactly on the goal point
            if len(idx) > 1:
                idx[-1] = len(t_pts) - 1
            else:
                idx.append(len(t_pts) - 1)
        keep = np.array(idx)
        q, P, t_pts = q[keep], P[keep], t_pts[keep]

        # Joint velocity/acceleration by the chain rule on the path parameter s:
        #   qd  = q'(s) * s_dot
        #   qdd = q''(s) * s_dot^2 + q'(s) * s_ddot
        # q(s) is smooth along a 10 cm line, so it is fitted with a degree-5
        # polynomial per joint (noise-free derivatives); s_dot and s_ddot are
        # the exact values of our profile. q'(s) is J^-1 * u, the Part-5 link.
        s_meas = np.clip((P - p0) @ u, 0.0, length)
        x = s_meas / length
        _, s_dot, s_ddot, _ = prof.evaluate(t_pts)
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)
        fit_err = 0.0
        for j in range(6):
            c = np.polyfit(x, q[:, j], 5)
            fit_err = max(fit_err, float(np.max(np.abs(np.polyval(c, x) - q[:, j]))))
            d1 = np.polyval(np.polyder(c, 1), x) / length
            d2 = np.polyval(np.polyder(c, 2), x) / length ** 2
            qd[:, j] = d1 * s_dot
            qdd[:, j] = d2 * s_dot ** 2 + d1 * s_ddot
        if fit_err > 1e-3:
            self.log.warning(f'q(s) polynomial fit residual {fit_err:.2e} rad (path near a '
                             'singularity?) -> joint derivatives less accurate')
        # jerk: the robot is at rest (q_dd = 0) before t=0 and after T, so pad
        # with zeros -> an acceleration step at start/stop shows up as jerk
        t_pad = np.concatenate(([-(t_pts[1] - t_pts[0])], t_pts, [2 * T - t_pts[-2]]))
        qdd_pad = np.vstack((np.zeros(6), qdd, np.zeros(6)))
        jerk = np.gradient(qdd_pad, t_pad, axis=0)
        v_tool = np.linalg.norm(np.gradient(P, t_pts, axis=0), axis=1)

        jt = traj.joint_trajectory
        order = [jt.joint_names.index(j) for j in cell.JOINT_NAMES]
        pts = [jt.points[i] for i in keep]
        for k, pt in enumerate(pts):
            pos, vel, acc = [0.0] * 6, [0.0] * 6, [0.0] * 6
            for c, idx in enumerate(order):
                pos[idx], vel[idx], acc[idx] = float(q[k, c]), float(qd[k, c]), float(qdd[k, c])
            pt.positions, pt.velocities, pt.accelerations = pos, vel, acc
            pt.time_from_start = seconds_to_duration(float(t_pts[k]))
        jt.points = pts

        v_pk, a_pk, j_pk = prof.peaks()
        s_c, v_c, a_c, _ = prof.evaluate(t_pts)
        # acceleration jumps at the knots (0 for the quintic)
        eps = 1e-7
        jumps = [abs(prof.evaluate([tk + eps])[2, 0] - prof.evaluate([tk - eps])[2, 0])
                 for tk in prof.t[1:-1]]
        jumps += [abs(prof.evaluate([0.0])[2, 0]), abs(prof.evaluate([T])[2, 0])]
        return {
            'kind': kind, 'profile': prof, 'traj': traj, 'length': length,
            't': t_pts, 'q': q, 'qd': qd, 'qdd': qdd, 'P': P, 's': s_c, 'v': v_c, 'a': a_c,
            'v_tool': v_tool,
            'metrics': {
                'duration_s': T, 'v_peak': v_pk, 'a_peak': a_pk, 'jerk_peak_in_segments': j_pk,
                'max_accel_jump': max(jumps),
                'max_abs_qdd': float(np.max(np.abs(qdd))),
                'rms_joint_jerk': float(np.sqrt(np.mean(jerk ** 2))),
                'max_tool_speed_error': float(np.max(np.abs(v_tool - np.abs(v_c)))),
                'n_points': len(t_pts)}}

    # ------------------------------------------------------------------
    def compare(self, start_joints, goal_joints, limits, tag):
        res = {k: self.build(start_joints, goal_joints, limits, k) for k in KINDS}
        lines = [f'--- {tag}: L = {res["cubic"]["length"] * 1000:.1f} mm, '
                 f'limits v<={limits["v_max"]} m/s, a<={limits["a_max"]} m/s^2 ---',
                 f'{"metric":24s}{"cubic":>14s}{"quintic":>14s}']
        for m in res['cubic']['metrics']:
            lines.append(f'{m:24s}{res["cubic"]["metrics"][m]:14.4f}'
                         f'{res["quintic"]["metrics"][m]:14.4f}')
        self.log.info('\n'.join(lines))
        self.plot(res, limits, tag)
        self.save_csv(res, tag)
        return res

    @staticmethod
    def choose(res, forced='auto'):
        """auto: prefer the profile whose acceleration is continuous (no torque
        steps -> bounded jerk); tie-break by RMS joint jerk."""
        if forced in KINDS:
            return forced
        return min(KINDS, key=lambda k: (res[k]['metrics']['max_accel_jump'] > 1e-6,
                                         res[k]['metrics']['rms_joint_jerk']))

    # ------------------------------------------------------------------
    def plot(self, res, limits, tag):
        colors = {'cubic': 'tab:blue', 'quintic': 'tab:red'}
        # Cartesian profile
        fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=False)
        for k in KINDS:
            prof = res[k]['profile']
            tg = np.linspace(0, prof.duration, 1500)
            s, v, a, j = prof.evaluate(tg)
            for i, (y, lab) in enumerate([(s, 's [m]'), (v, 'v [m/s]'), (a, 'a [m/s²]'),
                                          (j, 'jerk [m/s³]')]):
                ax[i].plot(tg, y, color=colors[k], label=f'{k} (T={prof.duration:.2f} s)')
                ax[i].set_ylabel(lab)
            ax[0].plot(prof.t, prof.s, 'o', color=colors[k], ms=5)
        for sign in (1, -1):
            ax[1].axhline(sign * limits['v_max'], ls='--', c='k', lw=0.8)
            ax[2].axhline(sign * limits['a_max'], ls='--', c='k', lw=0.8)
        ax[0].set_title(f'{tag}: Cartesian time law along the line (o = via points)')
        ax[-1].set_xlabel('t [s]')
        for a_ in ax:
            a_.grid(alpha=0.3)
        ax[0].legend()
        fig.tight_layout()
        f1 = os.path.join(self.out_dir, f'{tag}_cartesian_profiles.png')
        fig.savefig(f1, dpi=130)
        plt.close(fig)

        # joint accelerations + tool speed
        fig, ax = plt.subplots(3, 1, figsize=(9, 9))
        for i, k in enumerate(KINDS):
            r = res[k]
            for j in range(6):
                ax[i].plot(r['t'], r['qdd'][:, j], label=f'J{j + 1}')
            ax[i].set_title(f'{k}: joint accelerations after retiming '
                            f'(max |q̈| = {r["metrics"]["max_abs_qdd"]:.3f} rad/s², '
                            f'RMS jerk = {r["metrics"]["rms_joint_jerk"]:.3f} rad/s³)')
            ax[i].set_ylabel('q̈ [rad/s²]')
            ax[i].grid(alpha=0.3)
            ax[i].legend(ncol=6, fontsize=7)
            ax[2].plot(r['t'], np.abs(r['v']), '--', color=colors[k], label=f'{k} commanded')
            ax[2].plot(r['t'], r['v_tool'], color=colors[k], alpha=0.6,
                       label=f'{k} obtained (FK finite diff.)')
        ax[2].axhline(limits['v_max'], ls='--', c='k', lw=0.8)
        ax[2].set_title('tool0 linear speed: commanded vs obtained')
        ax[2].set_xlabel('t [s]')
        ax[2].set_ylabel('|v| [m/s]')
        ax[2].grid(alpha=0.3)
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        f2 = os.path.join(self.out_dir, f'{tag}_joint_accel_tool_speed.png')
        fig.savefig(f2, dpi=130)
        plt.close(fig)
        self.log.info(f'Plots: {f1}\n       {f2}')

    def save_csv(self, res, tag):
        for k in KINDS:
            r = res[k]
            path = os.path.join(self.out_dir, f'{tag}_{k}.csv')
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t', 's', 'v', 'a'] + [f'q{i}' for i in range(1, 7)]
                           + [f'qd{i}' for i in range(1, 7)] + [f'qdd{i}' for i in range(1, 7)]
                           + ['x', 'y', 'z', 'v_tool'])
                for i in range(len(r['t'])):
                    w.writerow([r['t'][i], r['s'][i], r['v'][i], r['a'][i]]
                               + list(r['q'][i]) + list(r['qd'][i]) + list(r['qdd'][i])
                               + list(r['P'][i]) + [r['v_tool'][i]])

    # ------------------------------------------------------------------
    def run_station(self, station, joints, execute=True, profile='auto'):
        """RED approach, grasp/release, BLUE retreat. Returns the chosen profile."""
        pre, target = STATIONS[station]
        seg = '4B' if station == 'pick' else '4D'

        start = self.mi.current_joints() if execute else joints[pre]
        if execute and np.max(np.abs(np.array(start) - joints[pre])) > 0.01:
            raise RuntimeError(f'Robot is not at {pre}; move there first (go_to_start:=true)')
        red = self.compare(start, joints[target], cell.LIMITS_RED, f'{seg}_red_{pre}_to_{target}')
        chosen = self.choose(red, profile)
        self.log.info(f'{seg} RED: using {chosen.upper()} profile')
        if execute and not self.mi.execute(red[chosen]['traj']):
            raise RuntimeError('Execution of the RED segment failed')

        if execute:
            if station == 'pick':
                ok = self.mi.attach(cell.PIECE_ID)
                self.log.info(f'Piece attached to {cell.EEF_LINK}: {ok}')
            else:
                ok = self.mi.detach(cell.PIECE_ID)
                self.log.info(f'Piece released on the deposit table: {ok}')

        start = self.mi.current_joints() if execute else joints[target]
        blue = self.compare(start, joints[pre], cell.LIMITS_BLUE,
                            f'{seg}_blue_{target}_to_{pre}')
        chosen_b = profile if profile in KINDS else chosen
        if execute and not self.mi.execute(blue[chosen_b]['traj']):
            raise RuntimeError('Execution of the BLUE segment failed')
        return chosen


class FineApproachNode(Node):
    def __init__(self):
        super().__init__('fine_approach_node')
        self.declare_parameter('station', 'pick')          # pick (4B) | place (4D)
        self.declare_parameter('profile', 'auto')          # auto | cubic | quintic
        self.declare_parameter('execute', True)
        self.declare_parameter('go_to_start', True)        # plan to pre_X first
        self.declare_parameter('planner', 'RRTConnectkConfigDefault')
        self.declare_parameter('via_fractions', [0.2, 0.5, 0.8])
        self.declare_parameter('waypoint_dt', 0.05)
        self.declare_parameter('joints_file', cell.DEFAULT_JOINTS_FILE)
        self.declare_parameter('output_dir', cell.DEFAULT_OUTPUT_DIR)
        self.mi = MoveItClient(self)

    def run(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.mi.wait_for_servers()
        joints = {k: np.array(v) for k, v in load_key_joints(g('joints_file')).items()}
        station = g('station')
        if station not in STATIONS:
            raise ValueError("station must be 'pick' or 'place'")
        pre = STATIONS[station][0]

        if g('execute') and g('go_to_start'):
            cur = self.mi.current_joints()
            if np.max(np.abs(np.array(cur) - joints[pre])) > 0.01:
                self.get_logger().info(f'Moving to {pre} with {g("planner")} ...')
                ok, traj, _, code = self.mi.plan_to_joints(joints[pre], planner_id=g('planner'))
                if not ok or not self.mi.execute(traj):
                    raise RuntimeError(f'Could not reach {pre} (code {code})')

        fa = FineApproach(self, self.mi, g('output_dir'), g('via_fractions'), g('waypoint_dt'))
        fa.run_station(station, joints, execute=g('execute'), profile=g('profile'))
        self.get_logger().info('Fine approach finished.')


def main(args=None):
    rclpy.init(args=args)
    node = FineApproachNode()
    try:
        node.run()
    except (RuntimeError, ValueError) as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
