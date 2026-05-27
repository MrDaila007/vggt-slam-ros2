from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vggt_slam_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Danila',
    maintainer_email='danilae007@gmail.com',
    description='ROS2 Visual SLAM using VGGT for real-time dense 3D reconstruction',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = vggt_slam_ros2.nodes.slam_node:main',
            'pointcloud_node = vggt_slam_ros2.nodes.pointcloud_node:main',
        ],
    },
)
