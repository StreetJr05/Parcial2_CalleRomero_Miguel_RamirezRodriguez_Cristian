from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        'lrmate200id', package_name='lrmate200id_moveit_config'
    ).to_moveit_configs()

    args = [
        DeclareLaunchArgument('out_dir', default_value='~/tap02_results'),
        DeclareLaunchArgument('tramo', default_value='',
                              description='Start logging this segment right away'),
    ]

    logger = Node(
        package='lrmate_analysis',
        executable='jacobian_logger',
        output='screen',
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            {'group': 'manipulator', 'base_link': 'base_link', 'tip_link': 'tool0',
             'out_dir': LaunchConfiguration('out_dir'),
             'tramo': LaunchConfiguration('tramo')},
        ],
    )
    return LaunchDescription(args + [logger])
