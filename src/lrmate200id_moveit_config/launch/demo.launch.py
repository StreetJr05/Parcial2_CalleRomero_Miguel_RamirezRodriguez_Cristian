from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    # The OMPL pipeline (planners, adapters, group settings) is now read from
    # config/ompl_planning.yaml instead of being hard-coded here.
    moveit_config = (
        MoveItConfigsBuilder("lrmate200id", package_name="lrmate200id_moveit_config")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )
    return generate_demo_launch(moveit_config)
