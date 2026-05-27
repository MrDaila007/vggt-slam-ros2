"""
Lightweight launch: only the point-cloud node (no SLAM backend).
Useful for quick depth visualisation from a live camera.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('vggt_slam_ros2')
    default_params = PathJoinSubstitution([pkg, 'config', 'params.yaml'])

    args = [
        DeclareLaunchArgument('params_file',  default_value=default_params),
        DeclareLaunchArgument('image_topic',  default_value='/camera/image_raw'),
        DeclareLaunchArgument('namespace',    default_value='vggt_pc'),
    ]

    pc_node = Node(
        package='vggt_slam_ros2',
        executable='pointcloud_node',
        name='vggt_pointcloud_node',
        namespace=LaunchConfiguration('namespace'),
        parameters=[LaunchConfiguration('params_file')],
        remappings=[('image_raw', LaunchConfiguration('image_topic'))],
        output='screen',
    )

    return LaunchDescription(args + [pc_node])
