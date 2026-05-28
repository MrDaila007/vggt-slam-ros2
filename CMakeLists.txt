cmake_minimum_required(VERSION 3.8)
project(vggt_slam_ros2)

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# --------------------------------------------------------------------------- #
# Custom interface generation
# --------------------------------------------------------------------------- #
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SaveMap.srv"
  DEPENDENCIES std_msgs
)

# --------------------------------------------------------------------------- #
# Python package installation
# NOTE: rosidl_generate_interfaces internally calls ament_python_install_package
# for the generated interface bindings. We must NOT call it again for the same
# package name — that would create duplicate CMake targets.
# Instead, install the node Python sources directly into site-packages.
# --------------------------------------------------------------------------- #
install(
  DIRECTORY vggt_slam_ros2/
  DESTINATION lib/python3/dist-packages/vggt_slam_ros2/
)

# Install launch, config, and srv directories
install(
  DIRECTORY launch config srv
  DESTINATION share/${PROJECT_NAME}
)

# Install node entry-point scripts (makes `ros2 run` work)
install(
  PROGRAMS
    vggt_slam_ros2/nodes/slam_node.py
    vggt_slam_ros2/nodes/pointcloud_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
