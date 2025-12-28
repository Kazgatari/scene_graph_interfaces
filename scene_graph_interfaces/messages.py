"""Message definitions for scene graph interfaces using Python dataclasses."""

from dataclasses import dataclass, field
from typing import List
from geometry_msgs.msg import Point32


@dataclass
class ObjectSegment:
    """Represents a detected object with segmentation information.
    
    Attributes:
        class_name: The detected object class/category
        bounding_box: List of Point32 representing the bounding box corners
        segment: List of Point32 representing the segmentation mask points
    """
    class_name: str
    bounding_box: List[Point32] = field(default_factory=list)
    segment: List[Point32] = field(default_factory=list)


@dataclass
class ObjectSegmentList:
    """Container for multiple detected objects with segmentation information.
    
    Attributes:
        objects: List of ObjectSegment detections
    """
    objects: List[ObjectSegment] = field(default_factory=list)
