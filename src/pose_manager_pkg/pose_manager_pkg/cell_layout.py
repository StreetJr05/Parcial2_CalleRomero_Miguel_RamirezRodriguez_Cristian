"""Geometry of the assembly cell: ONE place where every number lives.

The scene node, the pose node and the motion nodes all import from here, so the
piece, the tables and the pick/place poses can never get out of sync.

Frames: everything is expressed in 'world' (= base_link, the SRDF virtual joint
is fixed and has identity transform). Units: metres / radians.

tool0 convention (after the URDF fix): origin on the flange face, +Z pointing
out of the flange. "Tool pointing down" therefore means tool0 Z = -world Z.
"""
import math

PLANNING_FRAME = 'world'
GROUP = 'manipulator'
EEF_LINK = 'tool0'
JOINT_NAMES = [f'joint_{i}' for i in range(1, 7)]
HOME_JOINTS = [0.0] * 6

# ---------------------------------------------------------------- scene ----
TABLE_SIZE = (0.25, 0.25, 0.20)          # x, y, z of both tables
PIECE_SIZE = (0.04, 0.04, 0.04)          # cube that is picked
PICK_TABLE_XY = (0.00, -0.50)
PLACE_TABLE_XY = (0.00, 0.50)
POLE_XY = (0.32, -0.32)                  # between HOME and the pick station
POLE_HEIGHT = 0.60
POLE_RADIUS = 0.04

TABLE_TOP_Z = TABLE_SIZE[2]
GAP = 0.001                               # avoid "touching" contacts

# Collision objects: id -> (type, dimensions, (x, y, z) centre)
# CYLINDER dimensions follow shape_msgs: [height, radius]
SCENE_OBJECTS = {
    'pick_table': ('box', list(TABLE_SIZE),
                   (PICK_TABLE_XY[0], PICK_TABLE_XY[1], TABLE_TOP_Z / 2)),
    'deposit_table': ('box', list(TABLE_SIZE),
                      (PLACE_TABLE_XY[0], PLACE_TABLE_XY[1], TABLE_TOP_Z / 2)),
    'fixed_pole': ('cylinder', [POLE_HEIGHT, POLE_RADIUS],
                   (POLE_XY[0], POLE_XY[1], POLE_HEIGHT / 2)),
    'target_piece': ('box', list(PIECE_SIZE),
                     (PICK_TABLE_XY[0], PICK_TABLE_XY[1],
                      TABLE_TOP_Z + GAP + PIECE_SIZE[2] / 2)),
}
PIECE_ID = 'target_piece'

# ---------------------------------------------------------------- poses ----
GRASP_CLEARANCE = 0.005   # flange stops 5 mm above the piece (vacuum-cup idea)
APPROACH_DIST = 0.10      # pre-pick / pre-place are 10 cm above pick / place


def tool_down_quaternion(yaw):
    """Quaternion (x, y, z, w) of Rz(yaw) * Rx(pi): tool Z pointing down."""
    return (math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0, 0.0)


def _pose(x, y, z):
    yaw = math.atan2(y, x)           # align tool with the radial direction -> joint_6 ~ 0
    return {'position': (x, y, z), 'orientation': tool_down_quaternion(yaw)}


_piece_top_pick = TABLE_TOP_Z + GAP + PIECE_SIZE[2]
_piece_top_place = TABLE_TOP_Z + 2 * GAP + PIECE_SIZE[2]   # released 2 mm above table

KEY_POSES = {
    'pick': _pose(*PICK_TABLE_XY, _piece_top_pick + GRASP_CLEARANCE),
    'pre_pick': _pose(*PICK_TABLE_XY, _piece_top_pick + GRASP_CLEARANCE + APPROACH_DIST),
    'place': _pose(*PLACE_TABLE_XY, _piece_top_place + GRASP_CLEARANCE),
    'pre_place': _pose(*PLACE_TABLE_XY,
                       _piece_top_place + GRASP_CLEARANCE + APPROACH_DIST),
}

# IK is solved in this order; each solution seeds the next one so pre_pick/pick
# (and pre_place/place) end up on the same arm/wrist branch.
IK_ORDER = [('pre_pick', 'home'), ('pick', 'pre_pick'),
            ('pre_place', 'home'), ('place', 'pre_place')]

# Consultancy speed limits for the fine approach (m/s, m/s^2)
LIMITS_RED = {'v_max': 0.200, 'a_max': 0.300}    # going down (ida)
LIMITS_BLUE = {'v_max': 0.100, 'a_max': 0.020}   # going back up (retorno)

DEFAULT_JOINTS_FILE = '~/tap02_results/key_joints.yaml'
DEFAULT_OUTPUT_DIR = '~/tap02_results'
