"""ROS2 VLM Service Node

Ported from ROS1 implementation for Vision Language Model (VLM) inference.


- external: Containerized Google API (OpenAI-compatible)

Features:
- Object detection with attributes and relationships
- Persistent object tracking across frames
- Rate limiting for external APIs
- JSON response parsing and validation
"""

import json
import base64
import time
import numpy as np
from typing import Dict, List, Optional
import requests
import cv2
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

# TODO: Import ROS2 service and message types from scene_graph_interfaces
# from scene_graph_interfaces.srv import VLMInference, VLMInferenceResponse
# from scene_graph_interfaces.msg import (
#     ObjectSceneGraph, ObjectInfo, ObjectAttribute,
#     ObjectPart, ObjectSpatialContext
# )


class VLMServiceNode(Node):
    """ROS2 Service node for Vision Language Model inference
    
    Provides vlm_inference service that:
    1. Accepts RGB images
    2. Performs object detection using configured VLM backend
    3. Returns detected objects with attributes and relationships
    4. Maintains persistent object tracking across frames
    """
    
    def __init__(self, node_name: str = 'vlm_service_node'):
        """Initialize the VLM Service Node
        
        Args:
            node_name: Name of the ROS2 node
        """
        super().__init__(node_name)
        
        # Declare parameters for backend selection and configuration
        self.declare_parameter('backend', 'external')  # florence, vLLM, external
        self.declare_parameter('vlm_url', 'http://localhost:8002/v1/chat/completions')
        self.declare_parameter('florence_url', 'http://florence:8001')
        self.declare_parameter('external_url', 'http://localhost:8002/v1')
        self.declare_parameter('external_rate_limit', 4.0)
        
        # Get parameters
        self.backend = self.get_parameter('backend').value
        self.vlm_url = self.get_parameter('vlm_url').value
        self.florence_url = self.get_parameter('florence_url').value
        self.external_url = self.get_parameter('external_url').value
        self.external_rate_limit = self.get_parameter('external_rate_limit').value
        
        self.get_logger().info(f"VLM Service Node started with backend: {self.backend}")
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Object tracking
        self.known_objects: List[Dict] = []
        self.next_persistent_id = 1
        
        # Matching thresholds
        self.position_threshold = 50.0
        self.name_match_weight = 0.7
        self.position_match_weight = 0.3
        
        # External API rate limiting
        self.external_client_ready = False
        self.last_external_request_time = 0.0
        self.external_request_interval = self.external_rate_limit
        
        if self.backend == 'external':
            self.external_client_ready = True
            self.get_logger().info(
                f"External API ready to use containerized Google API at: {self.external_url}"
            )
        
        # TODO: Create ROS2 service
        # self.service = self.create_service(
        #     VLMInference,
        #     'vlm_inference',
        #     self.handle_vlm_inference
        # )
        
        # Publisher for relationships
        self.relationships_pub = self.create_publisher(String, '/scene_graph/object_relationships', 10)
        
        # Detection prompt
        self.detection_prompt = """Analyze this image and detect all objects. You must respond with ONLY valid JSON in the exact format below, with no additional text, explanations, or markdown formatting.

CRITICAL: Your response must be valid JSON that can be parsed. Do not include any text before or after the JSON. Do not use "..." or truncate any values.

Required JSON format:
{
  "objects": [
    {
      "bounding_box": [x_min, y_min, x_max, y_max],
      "label": "object_name",
      "attributes": {
        "color": "color_value",
        "style": "style_value"
      },
      "relations": [
        {
          "type": "relation_type",
          "target": "related_object_name"
        }
      ]
    }
  ]
}

Rules:
- Bounding box coordinates are percentile values from 0 to 999 (0 = top/left, 999 = bottom/right)
- Use complete attribute values, never "..." or truncation
- Include empty string "" for unknown attributes
- Relations describe spatial relationships like: "on", "under", "attached_to", "in", "at"
- Response must be parseable JSON only

Detect all objects in the image and respond with the JSON:"""
    
    def image_to_base64(self, image_msg: Image) -> Optional[str]:
        """Convert ROS Image message to base64 encoded string
        
        Args:
            image_msg: ROS2 Image message
            
        Returns:
            Base64 encoded JPEG image string, or None if conversion failed
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
            
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', cv_image)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
            
        except Exception as e:
            self.get_logger().error(f"Error converting image to base64: {e}")
            return None
    
    
    def call_external_api(self, image_base64: str) -> Optional[Dict]:
        """Call external containerized Google API using OpenAI-compatible format
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Parsed detection result or None if API call failed
        """
        try:
            if not self.external_client_ready:
                self.get_logger().error("External API client not ready")
                return None
            
            # Rate limiting
            current_time = time.time()
            time_since_last_request = current_time - self.last_external_request_time
            
            if time_since_last_request < self.external_request_interval:
                wait_time = self.external_request_interval - time_since_last_request
                self.get_logger().info(
                    f"Rate limiting: waiting {wait_time:.2f}s before next API call"
                )
                time.sleep(wait_time)
            
            # Update last request time
            self.last_external_request_time = time.time()
            
            # Prepare OpenAI-compatible request
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.external_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.detection_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7
            }
            
            self.get_logger().info("Sending request to containerized Google API...")
            
            # Track request time
            request_start_time = time.time()
            
            response = requests.post(
                f"{self.external_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=360
            )
            
            # Calculate request duration
            request_duration = time.time() - request_start_time
            self.get_logger().info(f"External API request completed in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Log performance info if available
                if "performance" in result:
                    perf = result["performance"]
                    self.get_logger().info(
                        f"Inference time: {perf.get('inference_time_seconds', 'N/A')}s"
                    )
                
                return self.parse_json_response(content)
            elif response.status_code == 429:
                self.get_logger().warn("Rate limit exceeded on external API")
                return None
            else:
                self.get_logger().error(f"External API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error calling external API: {e}")
            return None
    
    def parse_json_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from VLM, handling markdown code blocks
        
        Args:
            response_text: Response text from VLM API
            
        Returns:
            Parsed JSON dict or None if parsing failed
        """
        try:
            # Clean up response text (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON parsing error: {e}")
            self.get_logger().error(f"Response text: {response_text[:500]}...")
            return None
    
    def calculate_object_similarity(self, obj1: Dict, obj2: Dict) -> float: # TODO: change similarity based on kdtrees or other spatial methods
        """Calculate similarity between two objects based on name and position
        
        Args:
            obj1: First object detection data
            obj2: Second object detection data
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Name similarity (exact match or contains)
        name1 = obj1.get("label", "").lower()
        name2 = obj2.get("label", "").lower()
        
        name_score = 0.0
        if name1 == name2:
            name_score = 1.0
        elif name1 in name2 or name2 in name1:
            name_score = 0.8
        elif any(word in name2.split() for word in name1.split()) or \
             any(word in name1.split() for word in name2.split()):
            name_score = 0.6
        
        # Position similarity (center of bounding boxes)
        bbox1 = obj1.get("bounding_box", [0, 0, 0, 0])
        bbox2 = obj2.get("bounding_box", [0, 0, 0, 0])
        
        center1_x = (bbox1[0] + bbox1[2]) / 2.0
        center1_y = (bbox1[1] + bbox1[3]) / 2.0
        center2_x = (bbox2[0] + bbox2[2]) / 2.0
        center2_y = (bbox2[1] + bbox2[3]) / 2.0
        
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        position_score = max(0.0, 1.0 - (distance / self.position_threshold))
        
        # Combined similarity score
        total_score = (name_score * self.name_match_weight +
                      position_score * self.position_match_weight)
        
        return total_score
    
    def find_or_create_persistent_id(self, obj_data: Dict) -> int:
        """Find existing object or create new persistent ID
        
        Args:
            obj_data: Object detection data
            
        Returns:
            Persistent ID for the object
        """
        best_match_id = None
        best_similarity = 0.0
        similarity_threshold = 0.6
        
        # Check against known objects
        for known_obj in self.known_objects:
            similarity = self.calculate_object_similarity(obj_data, known_obj["data"])
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match_id = known_obj["persistent_id"]
        
        if best_match_id is not None:
            # Update the known object with current data
            for known_obj in self.known_objects:
                if known_obj["persistent_id"] == best_match_id:
                    known_obj["data"] = obj_data.copy()
                    known_obj["last_seen"] = time.time()
                    break
            return best_match_id
        else:
            # Create new persistent ID
            new_id = self.next_persistent_id
            self.next_persistent_id += 1
            
            # Add to known objects
            self.known_objects.append({
                "persistent_id": new_id,
                "data": obj_data.copy(),
                "last_seen": time.time()
            })
            
            return new_id
    
    def cleanup_old_objects(self, max_age_seconds: int = 300):
        """Remove objects that haven't been seen for a while
        
        Args:
            max_age_seconds: Maximum age of objects to keep (default 5 minutes)
        """
        current_time = time.time()
        self.known_objects = [
            obj for obj in self.known_objects
            if current_time - obj["last_seen"] < max_age_seconds
        ]
    
    def create_object_scene_graph(self, detection_result: Optional[Dict]) -> List:
        """Convert detection result to ObjectSceneGraph messages
        
        TODO: Implement ObjectSceneGraph message creation with the correct message types
        
        Args:
            detection_result: Detection result from VLM API
            
        Returns:
            List of ObjectSceneGraph messages
        """
        try:
            if not detection_result or "objects" not in detection_result:
                return []
            
            # Clean up old objects periodically
            #self.cleanup_old_objects()
            
            scene_graphs = []
            object_mapping = {}
            relations_data = []
            
            # First pass: Create all objects with persistent IDs
            for obj_data in detection_result["objects"]:
                # TODO: Create ObjectSceneGraph message
                # scene_graph = ObjectSceneGraph()
                
                # Get or create persistent ID
                persistent_id = self.find_or_create_persistent_id(obj_data)
                
                # TODO: Create main_object as ObjectInfo
                # scene_graph.main_object = ObjectInfo()
                # scene_graph.main_object.id = persistent_id
                # scene_graph.main_object.name = obj_data.get("label", "unknown")
                
                object_mapping[obj_data.get("label", "unknown")] = persistent_id
                
                # Store relations data for later mapping
                relations_data.append({
                    "object_id": persistent_id,
                    "object_name": obj_data.get("label", "unknown"),
                    "relations": obj_data.get("relations", [])
                })
                
                # TODO: Set attributes, spatial_context, and environment
                
                # For now, just collect the data
                self.get_logger().debug(
                    f"Processing object: {obj_data.get('label', 'unknown')} "
                    f"(ID: {persistent_id})"
                )
            
            # Second pass: Map relationships and publish them
            self.map_and_publish_relationships(relations_data, object_mapping)
            
            self.get_logger().info(f"Created scene graphs for {len(scene_graphs)} objects")
                
            return scene_graphs
            
        except Exception as e:
            self.get_logger().error(f"Error creating ObjectSceneGraph: {e}")
            return []
    
    def map_and_publish_relationships(self, relations_data: List[Dict], object_mapping: Dict):
        """Map relation targets to actual object IDs and publish relationships
        
        Args:
            relations_data: List of object relation data
            object_mapping: Mapping from object names to persistent IDs
        """
        try:
            relationships = {}
            
            for obj_relations in relations_data:
                source_id = obj_relations["object_id"]
                source_name = obj_relations["object_name"]
                relations = obj_relations["relations"]
                
                for relation in relations:
                    target_name = relation.get("target", "")
                    relation_type = relation.get("type", "")
                    
                    if target_name and relation_type:
                        # Try to find target object ID by exact name match
                        target_id = object_mapping.get(target_name)
                        
                        # If exact match not found, try partial matching
                        if target_id is None:
                            for obj_name, obj_id in object_mapping.items():
                                if target_name.lower() in obj_name.lower() or \
                                   obj_name.lower() in target_name.lower():
                                    target_id = obj_id
                                    break
                        
                        if target_id is not None:
                            if source_id not in relationships:
                                relationships[source_id] = {}
                            relationships[source_id][target_id] = relation_type
                        else:
                            self.get_logger().warn(
                                f"Could not find target object '{target_name}' "
                                f"for relationship from '{source_name}'"
                            )
            
            # Publish relationships
            if relationships:
                relationships_msg = String()
                relationships_msg.data = json.dumps(relationships)
                self.relationships_pub.publish(relationships_msg)
                self.get_logger().info(f"Published {len(relationships)} object relationships")
                    
        except Exception as e:
            self.get_logger().error(f"Error mapping and publishing relationships: {e}")
    
    def handle_vlm_inference(self, request, response):
        """Handle VLM inference service request
        
        TODO: Implement service request handler when service types are defined
        
        Args:
            request: VLMInferenceRequest
            response: VLMInferenceResponse (to be filled)
            
        Returns:
            VLMInferenceResponse with detection results
        """
        try:
            # TODO: Implement service handler
            # - Convert image to base64
            # - Call appropriate backend API
            # - Create ObjectSceneGraph messages
            # - Populate response with results
            
            self.get_logger().debug("VLM inference request received TODO")
            response.success = False
            response.error_message = "VLM inference not yet implemented in ROS2 version"
            
        except Exception as e:
            self.get_logger().error(f"Error in VLM inference: {e}")
            response.success = False
            response.error_message = str(e)
        
        return response


def main(args=None):
    """Main entry point for the ROS2 VLM service node
    
    Args:
        args: Command line arguments
    """
    rclpy.init(args=args)
    
    try:
        # Create the VLM service node
        vlm_node = VLMServiceNode('vlm_service_node')
        
        # Validate backend parameter
        backend = vlm_node.backend
        if backend not in ['florence', 'vLLM', 'external']:
            vlm_node.get_logger().error(
                f"Invalid backend parameter: {backend}. "
                f"Must be 'florence', 'vLLM', or 'external'"
            )
            return 1
        
        if backend == 'external':
            # Test connection to external API container
            try:
                import requests
                external_url = vlm_node.external_url
                base_url = external_url.rstrip('/v1')
                
                vlm_node.get_logger().info(
                    f"Testing connection to containerized Google API at: {external_url}"
                )
                
                test_response = requests.get(f"{base_url}/health", timeout=5)
                if test_response.status_code == 200:
                    vlm_node.get_logger().info("External API health check passed")
                else:
                    vlm_node.get_logger().warn(
                        f"External API health check failed: {test_response.status_code}"
                    )
            except Exception as e:
                vlm_node.get_logger().warn(f"Could not reach external API container: {e}")
        
        vlm_node.get_logger().info("VLM Service Node ready")
        rclpy.spin(vlm_node)
        
    except KeyboardInterrupt:
        print("\nVLM Service Node shutting down...")
    except Exception as e:
        print(f"Error starting VLM Service Node: {e}", file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
