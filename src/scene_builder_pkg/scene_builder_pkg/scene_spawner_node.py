"""Part 4: collision scene (piece, pick table, fixed pole, deposit table).

Why the old version was unreliable:
  * It published on /collision_object right after creating the publisher.
    DDS discovery takes time, so the first messages are often lost, and there
    is no confirmation that MoveIt received anything.
  * setup.py pointed to 'scene_builder_pkg.scene_spawner' but the file is
    scene_spawner_node.py -> 'ros2 run' could not even start the node.
  * The piece floated in the air (nothing under it) and the pole was not on the
    way between HOME and pick, so there was nothing to avoid.

This version uses the /apply_planning_scene SERVICE: synchronous, and it
returns success=True only after move_group has applied the change.
The geometry lives in pose_manager_pkg/cell_layout.py (shared with the poses).

  ros2 run scene_builder_pkg spawn_scene                      # (re)build the cell
  ros2 run scene_builder_pkg spawn_scene --ros-args -p reset:=true   # clear it
"""
import rclpy
from rclpy.node import Node

from pose_manager_pkg.moveit_client import MoveItClient, spawn_cell_scene


class SceneSpawner(Node):
    def __init__(self):
        super().__init__('scene_spawner')
        self.declare_parameter('reset', False)
        self.mi = MoveItClient(self)

    def spawn_objects(self):
        log = self.get_logger()
        self.mi.wait_for_servers()
        reset = self.get_parameter('reset').value
        ok = spawn_cell_scene(self.mi, log, reset_only=reset)
        if reset:
            log.info('Scene cleared.')
        elif ok:
            log.info('Scene applied (RViz -> MotionPlanning -> Scene Objects).')
        else:
            log.error('move_group rejected the planning scene.')
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = SceneSpawner()
    try:
        node.spawn_objects()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
