"""
Main launch file for VGGT SLAM.

Usage:
  ros2 launch vggt_slam_ros2 vggt_slam.launch.py

Remapping camera topics (example for a RealSense D435):
  ros2 launch vggt_slam_ros2 vggt_slam.launch.py \
    image_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import PushRosNamespace


def generate_launch_description():
    pkg = FindPackageShare('vggt_slam_ros2')
    default_params = PathJoinSubstitution([pkg, 'config', 'params.yaml'])

    # ---- launch arguments ------------------------------------------------
    args = [
        DeclareLaunchArgument('params_file',    default_value=default_params),
        DeclareLaunchArgument('image_topic',    default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('namespace',      default_value='vggt_slam'),
        DeclareLaunchArgument('autostart',      default_value='true'),
    ]

    params_file = LaunchConfiguration('params_file')
    image_topic = LaunchConfiguration('image_topic')
    info_topic  = LaunchConfiguration('camera_info_topic')
    ns          = LaunchConfiguration('namespace')

    # ---- SLAM lifecycle node ---------------------------------------------
    slam_node = LifecycleNode(
        package='vggt_slam_ros2',
        executable='slam_node',
        name='vggt_slam_node',
        namespace=ns,
        parameters=[params_file],
        remappings=[
            ('image_raw',    image_topic),
            ('camera_info',  info_topic),
        ],
        output='screen',
    )

    # ---- Lifecycle manager (auto-activates the node) ---------------------
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        namespace=ns,
        parameters=[{
            'autostart': LaunchConfiguration('autostart'),
            'node_names': ['vggt_slam_node'],
        }],
        output='screen',
    )

    # ---- RViz (optional) -------------------------------------------------
    # Uncomment if you want RViz to launch automatically:
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', PathJoinSubstitution([pkg, 'config', 'vggt_slam.rviz'])],
    # )

    return LaunchDescription(args + [slam_node, lifecycle_manager])
