cmake_minimum_required(VERSION 3.8)
project(vggt_slam_ros2)

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

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
# Install Python modules next to rosidl-generated srv bindings.
# rosidl owns vggt_slam_ros2/__init__.py under local/lib/pythonX.Y/dist-packages/;
# installing core/nodes/utils elsewhere makes `import vggt_slam_ros2.nodes` fail.
set(_PY_PKG_DIR
  "local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/dist-packages/${PROJECT_NAME}"
)
install(
  DIRECTORY
    vggt_slam_ros2/core
    vggt_slam_ros2/nodes
    vggt_slam_ros2/utils
  DESTINATION ${_PY_PKG_DIR}
)

# Install launch, config, and srv directories
install(
  DIRECTORY launch config srv
  DESTINATION share/${PROJECT_NAME}
)

# Install node entry-point wrappers (ros2 run / launch expect names without .py)
install(
  PROGRAMS
    scripts/slam_node
    scripts/pointcloud_node
  DESTINATION lib/${PROJECT_NAME}
)

# --------------------------------------------------------------------------- #
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
