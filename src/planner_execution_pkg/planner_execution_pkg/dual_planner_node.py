"""Parts 4A / 4C: compare OMPL planners on the obstacle-avoidance segments.

Why the old node could not work:
  * `moveit_commander` is ROS 1 only; in ROS 2 Jazzy the import fails.
  * It planned to a named target 'pre_pick' that does not exist in the SRDF.
  * The second plan went to HOME, so the two planners never solved the same problem.
  * demo.launch.py registered the planner configs under the group
    'manipulator_arm' (the group is 'manipulator') with Humble-style keys, so
    'RRTstarkConfigDefault' was not a valid config for the group.

What this node does (plan only, nothing moves unless execute:=true):
  segment:=4A  HOME -> pre_pick       segment:=4C  pre_pick -> pre_place
  For every planner in `planners`, `trials` independent plans (sampling-based
  planners are random, one sample proves nothing). For every plan:
    - planning time [s] (reported by move_group)
    - joint-space path length  sum ||q_{i+1} - q_i||  [rad]
    - tool0 path length (MoveIt FK on every point)   [m]
    - smoothness: total turning angle of the joint-space path [rad]
      (sum of angles between consecutive segments; 0 = straight line)
    - trajectory duration after time parameterisation [s]
  Mean +- std table in the log, CSV + box plots in output_dir.

NOTE on RRT*: it is an *optimising* planner; MoveIt lets it refine the path
until `planning_time` is used up, so its planning time ~ planning_time by design.
That is the trade-off to discuss: more time -> shorter path.

Each successful plan is also shown in RViz (Planned Path) as it is computed.
"""
import csv
import os

import matplotlib
import numpy as np
import rclpy
from rclpy.node import Node

from pose_manager_pkg import cell_layout as cell
from pose_manager_pkg.moveit_client import MoveItClient, load_key_joints, traj_arrays

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SEGMENTS = {'4A': ('home', 'pre_pick'), '4C': ('pre_pick', 'pre_place')}
METRICS = ['planning_time_s', 'joint_path_rad', 'tool_path_m', 'turning_angle_rad',
           'duration_s']


def path_metrics(mi, traj):
    t, q, _, _ = traj_arrays(traj)
    dq = np.diff(q, axis=0)
    seg = np.linalg.norm(dq, axis=1)
    dq, seg = dq[seg > 1e-9], seg[seg > 1e-9]
    cosang = np.sum(dq[1:] * dq[:-1], axis=1) / (seg[1:] * seg[:-1])
    turning = float(np.sum(np.arccos(np.clip(cosang, -1.0, 1.0))))
    P = np.array([[p.position.x, p.position.y, p.position.z]
                  for p in (mi.compute_fk(qi) for qi in q)])
    return {'joint_path_rad': float(np.sum(seg)),
            'tool_path_m': float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
            'turning_angle_rad': turning,
            'duration_s': float(t[-1])}


class DualPlannerNode(Node):
    def __init__(self):
        super().__init__('dual_planner_node')
        self.declare_parameter('segment', '4A')
        self.declare_parameter('planners', ['RRTConnectkConfigDefault', 'RRTstarkConfigDefault'])
        self.declare_parameter('trials', 10)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('execute', False)   # execute one plan of `execute_planner`
        self.declare_parameter('execute_planner', 'RRTConnectkConfigDefault')
        self.declare_parameter('joints_file', cell.DEFAULT_JOINTS_FILE)
        self.declare_parameter('output_dir', cell.DEFAULT_OUTPUT_DIR)
        self.mi = MoveItClient(self)

    def evaluate_planners(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        log = self.get_logger()
        self.mi.wait_for_servers()
        joints = load_key_joints(g('joints_file'))
        seg = g('segment')
        if seg not in SEGMENTS:
            raise RuntimeError("segment must be '4A' or '4C'")
        start_name, goal_name = SEGMENTS[seg]
        start, goal = joints[start_name], joints[goal_name]

        results = {p: [] for p in g('planners')}
        for planner in g('planners'):
            log.info(f'--- {planner}: {g("trials")} trials, {seg} {start_name} -> {goal_name}')
            fails = 0
            for k in range(g('trials')):
                ok, traj, t_plan, code = self.mi.plan_to_joints(
                    goal, start_joints=start, planner_id=planner,
                    planning_time=g('planning_time'))
                if not ok:
                    fails += 1
                    log.warning(f'  trial {k + 1}: FAILED (code {code})')
                    continue
                m = path_metrics(self.mi, traj)
                m['planning_time_s'] = t_plan
                results[planner].append(m)
                log.info(f'  trial {k + 1}: t_plan={t_plan:.3f}s  joint={m["joint_path_rad"]:.3f}rad'
                         f'  tool={m["tool_path_m"]:.3f}m  turn={m["turning_angle_rad"]:.2f}rad')
            log.info(f'  success rate {g("trials") - fails}/{g("trials")}')

        self.report(results, seg)

        if g('execute'):
            planner = g('execute_planner')
            log.info(f'Executing {seg} with {planner} (robot must be at {start_name})')
            ok, traj, _, code = self.mi.plan_to_joints(goal, planner_id=planner,
                                                       planning_time=g('planning_time'))
            if ok:
                self.mi.execute(traj)
            else:
                log.error(f'Planning failed (code {code})')

    def report(self, results, seg):
        out = os.path.expanduser(self.get_parameter('output_dir').value)
        os.makedirs(out, exist_ok=True)
        planners = [p for p in results if results[p]]
        if not planners:
            self.get_logger().error('No successful plans; nothing to report.')
            return
        short = [p.replace('kConfigDefault', '') for p in planners]
        lines = [f'===== {seg} planner comparison (mean ± std) =====',
                 f'{"metric":20s}' + ''.join(f'{s:>24s}' for s in short)]
        for m in METRICS:
            row = f'{m:20s}'
            for p in planners:
                v = np.array([r[m] for r in results[p]])
                row += f'{v.mean():14.3f} ± {v.std():7.3f}'
            lines.append(row)
        self.get_logger().info('\n'.join(lines))

        with open(os.path.join(out, f'{seg}_planner_comparison.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['planner', 'trial'] + METRICS)
            for p in planners:
                for i, r in enumerate(results[p]):
                    w.writerow([p, i + 1] + [r[m] for m in METRICS])

        fig, ax = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4))
        for a, m in zip(ax, METRICS):
            a.boxplot([[r[m] for r in results[p]] for p in planners])
            a.set_xticks(range(1, len(short) + 1))
            a.set_xticklabels(short)
            a.set_title(m)
            a.grid(alpha=0.3)
        fig.suptitle(f'{seg}: OMPL planner comparison')
        fig.tight_layout()
        png = os.path.join(out, f'{seg}_planner_comparison.png')
        fig.savefig(png, dpi=120)
        plt.close(fig)
        self.get_logger().info(f'Saved {png} and CSV')


def main(args=None):
    rclpy.init(args=args)
    node = DualPlannerNode()
    try:
        node.evaluate_planners()
    except RuntimeError as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
