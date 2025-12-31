from setuptools import setup, find_packages

package_name = 'scene_graph_interfaces'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='your-email@example.com',
    description='Python interface definitions for scene graph processing',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_interface_node = scene_graph_interfaces.visual_interface.ros2_visual_interface_node:main',
            'vlm_service_node = scene_graph_interfaces.vlm_service.ros2_vlm_service_node:main',
        ],
    },
)
