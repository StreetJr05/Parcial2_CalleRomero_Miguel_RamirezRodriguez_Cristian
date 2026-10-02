"""Full cycle for the video: HOME -> 4A -> 4B -> 4C -> 4D -> HOME, animated in RViz.

Uses the planner chosen in 4A (param `planner`) for the free motions and the
profile chosen in 4B (param `profile`, default auto) for both fine approaches.
The scene is rebuilt at the start, so the piece is always back on the pick table.
"""
import numpy as np
import rclpy
from rclpy.node import Node

from planner_execution_pkg.fine_approach_node import FineApproach
from pose_manager_pkg import cell_layout as cell
from pose_manager_pkg.moveit_client import MoveItClient, load_key_joints, spawn_cell_scene


class PickPlaceCycle(Node):
    def __init__(self):
        super().__init__('pick_place_cycle')
        self.declare_parameter('planner', 'RRTConnectkConfigDefault')
        self.declare_parameter('profile', 'auto')
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('via_fractions', [0.2, 0.5, 0.8])
        self.declare_parameter('waypoint_dt', 0.05)
        self.declare_parameter('joints_file', cell.DEFAULT_JOINTS_FILE)
        self.declare_parameter('output_dir', cell.DEFAULT_OUTPUT_DIR)
        self.mi = MoveItClient(self)

    def free_move(self, goal, label):
        g = self.get_parameter
        self.get_logger().info(f'>>> {label} with {g("planner").value}')
        ok, traj, t_plan, code = self.mi.plan_to_joints(
            goal, planner_id=g('planner').value, planning_time=g('planning_time').value)
        if not ok:
            raise RuntimeError(f'{label}: planning failed (MoveIt code {code})')
        self.get_logger().info(f'    planned in {t_plan:.2f} s')
        if not self.mi.execute(traj):
            raise RuntimeError(f'{label}: execution failed')

    def run(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        log = self.get_logger()
        self.mi.wait_for_servers()
        joints = {k: np.array(v) for k, v in load_key_joints(g('joints_file')).items()}

        # robot to HOME first (with an empty scene, so it cannot be "stuck" in the piece)
        spawn_cell_scene(self.mi, reset_only=True)
        if np.max(np.abs(np.array(self.mi.current_joints()) - joints['home'])) > 0.01:
            self.free_move(joints['home'], 'Going HOME')
        spawn_cell_scene(self.mi, log)

        fa = FineApproach(self, self.mi, g('output_dir'), g('via_fractions'), g('waypoint_dt'))
        self.free_move(joints['pre_pick'], '4A  HOME -> pre_pick (obstacle avoidance)')
        log.info('>>> 4B  fine approach pick')
        chosen = fa.run_station('pick', joints, execute=True, profile=g('profile'))
        self.free_move(joints['pre_place'], '4C  pre_pick -> pre_place (obstacle avoidance)')
        log.info(f'>>> 4D  fine approach place ({chosen})')
        fa.run_station('place', joints, execute=True, profile=chosen)
        self.free_move(joints['home'], 'Back HOME')
        log.info('Cycle complete.')


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceCycle()
    try:
        node.run()
    except RuntimeError as e:
        node.get_logger().error(str(e))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
