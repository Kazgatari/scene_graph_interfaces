"""ROS2 Visual Interface Base Class

Ported from ROS1 implementation with the following changes:
- Uses rclpy instead of rospy
- Uses message_filters for ROS2
- Uses TF2 for coordinate transformations
- Uses service clients/servers for ROS2
"""

import numpy as np
import struct
from typing import List, Optional
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32, PointStamped
from std_msgs.msg import String, Int32, Header
import message_filters
import tf2_ros
import tf2_geometry_msgs
from image_geometry import PinholeCameraModel
from scene_graph_interfaces.msg import GraphObject, GraphObjects

# TODO: Import the correct ROS2 service types
# from scene_graph_interfaces.srv import VLMInference, VLMInferenceRequest


class VisualInterfaceBase(Node):
    """Base class for visual interface processing using ROS2
    
    Handles:
    - Synchronized sensor input (image, depth, odometry)
    - Camera calibration and coordinate transformations
    - 3D world position calculation from 2D detections
    - VLM service calls for object detection
    """
    
    def __init__(self, node_name: str = 'visual_interface_node'):
        """Initialize the VisualInterfaceBase node
        
        Args:
            node_name: Name of the ROS2 node
        """
        super().__init__(node_name)
        
        # Declare parameters (ROS2 way to get config)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('tf_buffer_duration', 60.0)
        
        self.camera_frame = self.get_parameter('camera_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        tf_buffer_duration = self.get_parameter('tf_buffer_duration').value
        
        # Initialize TF2 for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=tf_buffer_duration)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info(
            f"TF2 initialized with {tf_buffer_duration}s buffer: "
            f"transforming from '{self.camera_frame}' to '{self.world_frame}'"
        )
        
        # Synchronized input subscribers
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/points')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        
        # Create time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.odom_sub],
            queue_size=100,
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Initialize camera model
        self.camera_model = PinholeCameraModel()
        self.camera_info = None
        
        # Legacy camera intrinsics (fallback if camera_info not available)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0
        
        # Publisher for detected graph objects
        # Note: Using custom Python dataclass for GraphObjects since we're Python-only
        self.graph_objects_pub = self.create_publisher(
            GraphObjects,
            '/scene_graph/seen_graph_objects',
            10
        )
        
        # VLM service client
        self.vlm_service = None
        self.init_vlm_service()
        
        # Object ID counter
        self.object_id_counter = 0
    
    def init_vlm_service(self):
        """Initialize VLM service client with error handling
        
        TODO: Update service type to match ROS2 VLMInference service
        """
        try:
            # TODO: Implement ROS2 service client initialization
            # self.vlm_service = self.create_client(VLMInference, 'vlm_inference')
            # while not self.vlm_service.wait_for_service(timeout_sec=2.0):
            #     self.get_logger().warn('VLM service not available, waiting...')
            self.get_logger().info("VLM service initialization TODO")
        except Exception as e:
            self.get_logger().error(f"Error initializing VLM service: {e}")
            self.vlm_service = None
    
    ###############################################################
    ############## Custom Object Localization Methods #############
    ###############################################################
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics and initialize camera model
        
        Args:
            msg: Camera info message containing intrinsic parameters
        """
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)
        
        # Update legacy intrinsics
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        
        self.get_logger().info("Camera model initialized from camera_info")
    
    def is_camera_model_ready(self) -> bool:
        """Check if camera model has been initialized
        
        Returns:
            True if camera info has been received, False otherwise
        """
        return self.camera_info is not None
    
    def sample_depth_in_bbox(self, bbox: List[float], depth_msg: PointCloud2) -> List[float]:
        """Sample depth values within bounding box from point cloud
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            depth_msg: PointCloud2 message containing depth data
            
        Returns:
            List of sampled depth values, or [2.0] if no valid depths found
        """
        x1, y1, x2, y2 = bbox
        
        # Convert to integers and ensure within bounds
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        depth_values = []
        
        try:
            # Parse point cloud data
            point_step = depth_msg.point_step
            row_step = depth_msg.row_step
            
            # Calculate sampling parameters
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            step_x = max(1, bbox_width // 10)
            step_y = max(1, bbox_height // 10)
            
            # Sample points within bounding box
            for y in range(y1, y2, step_y):
                for x in range(x1, x2, step_x):
                    array_position = y * row_step + x * point_step
                    
                    if array_position + 12 <= len(depth_msg.data):
                        # Extract XYZ coordinates (PointXYZ format)
                        x_bytes = depth_msg.data[array_position:array_position+4]
                        y_bytes = depth_msg.data[array_position+4:array_position+8]
                        z_bytes = depth_msg.data[array_position+8:array_position+12]
                        
                        # Unpack as float32
                        x_val = struct.unpack('f', x_bytes)[0]
                        y_val = struct.unpack('f', y_bytes)[0]
                        z_val = struct.unpack('f', z_bytes)[0]
                        
                        # Use Z as depth, validate > 10cm
                        if not (np.isnan(z_val) or np.isinf(z_val)) and z_val > 0.1:
                            depth_values.append(z_val)
            
            return depth_values if depth_values else [2.0]
            
        except Exception as e:
            self.get_logger().warn(f"Error sampling depth: {e}")
            return [2.0]
    
    def pixel_to_camera_coords(self, pixel_x: float, pixel_y: float, depth: float) -> np.ndarray:
        """Convert pixel coordinates + depth to 3D camera coordinates
        
        Uses image_geometry for accurate conversion including distortion handling.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            depth: Depth value in meters
            
        Returns:
            3D point in camera frame [x, y, z]
        """
        if self.camera_info is None:
            self.get_logger().warn_once("Camera info not yet received, using default intrinsics")
            camera_x = (pixel_x - self.cx) * depth / self.fx
            camera_y = (pixel_y - self.cy) * depth / self.fy
            camera_z = depth
            return np.array([camera_x, camera_y, camera_z])
        
        # Use image_geometry's PinholeCameraModel
        ray = self.camera_model.projectPixelTo3dRay((pixel_x, pixel_y))
        
        # Scale the ray by depth
        camera_x = ray[0] * depth
        camera_y = ray[1] * depth
        camera_z = ray[2] * depth
        
        return np.array([camera_x, camera_y, camera_z])
    
    def transform_to_world_coords(
        self,
        camera_point: np.ndarray,
        timestamp: Optional[rclpy.time.Time] = None
    ) -> np.ndarray:
        """Transform point from camera frame to world frame using TF2
        
        Args:
            camera_point: 3D point in camera frame [x, y, z]
            timestamp: Image timestamp for exact robot pose lookup
            
        Returns:
            3D point in world frame [x, y, z]
        """
        try:
            # Create PointStamped in camera frame
            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            
            if timestamp is not None:
                point_camera.header.stamp = timestamp
            else:
                # Use latest transform if no timestamp
                point_camera.header.stamp = rclpy.time.Time()
                self.get_logger().warn_once(
                    "No timestamp provided, using latest transform (may be inaccurate)"
                )
            
            point_camera.point.x = float(camera_point[0])
            point_camera.point.y = float(camera_point[1])
            point_camera.point.z = float(camera_point[2])
            
            # Transform to world frame
            try:
                point_world = self.tf_buffer.transform(
                    point_camera,
                    self.world_frame,
                    timeout=rclpy.duration.Duration(seconds=2.0)
                )
                
                return np.array([
                    point_world.point.x,
                    point_world.point.y,
                    point_world.point.z
                ])
                
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f"TF timestamp too old (>buffer): {e}. Using latest transform."
                )
                # Retry with latest transform
                point_camera.header.stamp = rclpy.time.Time()
                try:
                    point_world = self.tf_buffer.transform(
                        point_camera,
                        self.world_frame,
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    return np.array([point_world.point.x, point_world.point.y, point_world.point.z])
                except Exception as e2:
                    self.get_logger().warn(f"Fallback transform failed: {e2}. Using camera coordinates.")
                    return camera_point
                    
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
                self.get_logger().warn(f"TF transform failed: {e}. Using camera coordinates.")
                return camera_point
            
        except Exception as e:
            self.get_logger().warn(f"Coordinate transform failed: {e}, using camera coordinates")
            return camera_point
    
    def calculate_3d_world_position(
        self,
        bbox: List[float],
        depth_msg: PointCloud2,
        timestamp: Optional[rclpy.time.Time] = None
    ) -> np.ndarray:
        """Calculate 3D world position from 2D bounding box and depth data
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth_msg: PointCloud2 depth message
            timestamp: Image timestamp for accurate pose lookup
            
        Returns:
            3D position in world frame
        """
        x1, y1, x2, y2 = bbox
        
        # Get center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Sample depth values
        depth_samples = self.sample_depth_in_bbox(bbox, depth_msg)
        estimated_depth = np.median(depth_samples)
        
        # Convert to 3D camera coordinates
        camera_point = self.pixel_to_camera_coords(center_x, center_y, estimated_depth)
        
        # Transform to world coordinates
        world_point = self.transform_to_world_coords(camera_point, timestamp)
        
        return world_point
    
    def create_3d_bounding_box(self, position: np.ndarray, size: np.ndarray) -> List[Point32]:
        """Create 3D bounding box from center position and size
        
        Args:
            position: Center position [x, y, z]
            size: Dimensions [width, height, depth]
            
        Returns:
            List of Point32 representing min and max corners
        """
        size = self.sanitize_object_size(size)
        half_x, half_y, half_z = size / 2.0
        
        min_corner = Point32(
            x=float(position[0] - half_x),
            y=float(position[1] - half_y),
            z=float(position[2] - half_z)
        )
        max_corner = Point32(
            x=float(position[0] + half_x),
            y=float(position[1] + half_y),
            z=float(position[2] + half_z)
        )
        
        return [min_corner, max_corner]
    
    def sanitize_object_size(self, size: np.ndarray) -> np.ndarray:
        """Ensure object size has valid positive dimensions
        
        Args:
            size: Object size [x, y, z]
            
        Returns:
            Validated size array
        """
        if size is None or len(size) != 3:
            return np.array([0.2, 0.2, 0.2])
        
        min_dimension = 0.05  # 5cm minimum
        size = np.maximum(size, min_dimension)
        size = np.nan_to_num(size, nan=0.2, posinf=1.0, neginf=0.05)
        
        return size
    
    def estimate_object_size_from_bbox(self, bbox: List[float], depth: float) -> np.ndarray:
        """Estimate 3D object size from 2D bounding box and depth
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth: Estimated depth of object
            
        Returns:
            3D size estimate [width, height, depth]
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate 2D size in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        width_pixels = max(width_pixels, 5.0)
        height_pixels = max(height_pixels, 5.0)
        
        if self.camera_info is None:
            width_real = (width_pixels * depth) / self.fx
            height_real = (height_pixels * depth) / self.fy
        else:
            # Use image_geometry for accurate estimation
            top_left_ray = self.camera_model.projectPixelTo3dRay((x1, y1))
            bottom_right_ray = self.camera_model.projectPixelTo3dRay((x2, y2))
            
            top_left_3d = np.array(top_left_ray) * depth
            bottom_right_3d = np.array(bottom_right_ray) * depth
            
            width_real = abs(bottom_right_3d[0] - top_left_3d[0])
            height_real = abs(bottom_right_3d[1] - top_left_3d[1])
        
        # Estimate depth dimension using object heuristics
        aspect_ratio = width_pixels / height_pixels
        
        if aspect_ratio > 2.0:  # Wide objects
            depth_real = min(width_real, height_real) * 0.5
        elif aspect_ratio < 0.5:  # Tall objects
            depth_real = width_real * 0.8
        else:  # Roughly square
            depth_real = min(width_real, height_real) * 0.7
        
        # Ensure reasonable bounds
        width_real = np.clip(width_real, 0.05, 2.0)
        height_real = np.clip(height_real, 0.05, 2.0)
        depth_real = np.clip(depth_real, 0.05, 2.0)
        
        return np.array([width_real, height_real, depth_real])
    
    def validate_detection_data(
        self,
        bbox: List[float],
        world_position: np.ndarray,
        object_size: np.ndarray
    ) -> bool:
        """Validate detection data for reasonable values
        
        Args:
            bbox: Bounding box coordinates
            world_position: 3D world position
            object_size: Object dimensions
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check bounding box
        if len(bbox) != 4 or any(coord < 0 for coord in bbox):
            return False
        
        # Check world position
        if not np.all(np.isfinite(world_position)):
            return False
        
        # Check object size
        if not np.all(object_size > 0) or np.any(object_size > 5.0):
            return False
        
        return True
    
    ###############################################################
    ############## End of custom Object Localization ##############
    ###############################################################

    def process_vlm_detections(
        self,
        image_msg: Image,
        depth_msg: PointCloud2,
        odom_msg: Odometry
    ) -> List:
        """Process VLM detections and return graph objects
        
        TODO: Implement VLM service call and detection processing
        
        Args:
            image_msg: Input RGB image
            depth_msg: Input depth point cloud
            odom_msg: Robot odometry
            
        Returns:
            List of detected graph objects
        """
        # TODO: Implement VLM inference call
        # - Initialize service if not available
        # - Call VLM service with image
        # - Process response and calculate 3D positions
        # - Return list of GraphObject messages
        
        self.get_logger().debug("VLM detection processing TODO")
        return []
    
    def synchronized_callback(
        self,
        image_msg: Image,
        depth_msg: PointCloud2,
        odom_msg: Odometry
    ):
        """Process synchronized sensor data
        
        TODO: Implement publishing of detected objects
        
        Args:
            image_msg: RGB image message
            depth_msg: Depth point cloud message
            odom_msg: Odometry message
        """
        try:
            graph_objects = self.process_vlm_detections(image_msg, depth_msg, odom_msg)
            
            if graph_objects:
                # TODO: Create and publish GraphObjects message
                # graph_objects_msg = GraphObjects()
                # graph_objects_msg.header = Header(
                #     stamp=self.get_clock().now().to_msg(),
                #     frame_id='world'
                # )
                # graph_objects_msg.objects = graph_objects
                # self.graph_objects_pub.publish(graph_objects_msg)
                
                self.get_logger().info(f"Published {len(graph_objects)} detected objects")
            else:
                self.get_logger().debug("No objects detected in current frame")
                
        except Exception as e:
            self.get_logger().error(f"Error in synchronized callback: {e}")


def main(args=None):
    """Main entry point for the ROS2 node
    
    Args:
        args: Command line arguments
    """
    rclpy.init(args=args)
    
    # Create and run the visual interface node
    visual_interface = VisualInterfaceBase('visual_interface_node')
    
    try:
        visual_interface.get_logger().info(
            "Visual Interface Node started, waiting for synchronized messages..."
        )
        rclpy.spin(visual_interface)
    except KeyboardInterrupt:
        visual_interface.get_logger().info("Visual Interface Node shutting down...")
    finally:
        visual_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

