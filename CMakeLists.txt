cmake_minimum_required(VERSION 3.8)
project(scene_graph_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Generate ROS2 message and service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Custom graph object messages
  "msg/GraphObject.msg"
  "msg/GraphObjects.msg"
  # Scene graph related messages
  "msg/ObjectInfo.msg"
  "msg/ObjectAttribute.msg"
  "msg/ObjectSpatialContext.msg"
  "msg/ObjectPart.msg"
  "msg/ObjectSceneGraph.msg"
  # Additional detection messages
  "msg/DetectedObject.msg"
  "msg/DetectedObjects.msg"
  "msg/ObjectDescription.msg"
  "msg/ObjectDescriptions.msg"
  # VLM Inference service
  "srv/VLMInference.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

ament_package()
