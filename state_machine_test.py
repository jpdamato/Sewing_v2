from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import deque
import numpy as np

class State(Enum):
    IDLE = auto()
    SCISSOR_DETECTED = auto()
    CLOTH_PREPARED = auto()
    STITCHING = auto()
    THREAD_360_CHECK = auto()
    TASK_8_CLOTH_CENTERED = auto()
    TASK_9_CLOTH_LATERAL = auto()
    
class DetectedObject(Enum):
    NEEDLE = "needle"
    THREAD = "thread"
    SCISSOR = "scissor"
    CLOTH = "cloth"

@dataclass
class StitchData:
    frame_number: int
    position: tuple
    confidence: float
    mask: Optional[np.ndarray] = None

@dataclass
class YOLOSegmentationWrapper:
    """Wrapper for YOLO segmentation model"""
    model = None
    
    def __init__(self, model_path: str = 'yolov8n-seg.pt'):
        """
        Initialize YOLO segmentation model
        
        Args:
            model_path: Path to YOLO segmentation weights
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            print("Warning: ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
    
    def predict(self, frame: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        Run YOLO segmentation on frame
        
        Args:
            frame: Input image (numpy array)
            conf: Confidence threshold
            
        Returns:
            List of detections with format:
            {
                'class': str,
                'confidence': float,
                'bbox': (x1, y1, x2, y2),
                'mask': np.ndarray,  # Segmentation mask
                'area': float
            }
        """
        if self.model is None:
            return []
        
        results = self.model.predict(frame, conf=conf, verbose=False)
        detections = []
        
        for result in results:
            if result.masks is None:
                continue
                
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]
                
                detection = {
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': tuple(box.xyxy[0].cpu().numpy()),
                    'mask': mask.data[0].cpu().numpy(),
                    'area': float(mask.data[0].sum())
                }
                detections.append(detection)
        
        return detections

@dataclass
class TaskDetectionStateMachine:
    # Configuration
    scissor_frame_threshold: int = 5
    stitch_count_threshold: int = 8
    confidence_threshold: float = 0.25
    
    # YOLO model
    yolo_model: Optional[YOLOSegmentationWrapper] = None
    
    # State tracking
    current_state: State = State.IDLE
    frame_count: int = 0
    
    # Detection tracking
    scissor_frames: int = 0
    stitches: List[StitchData] = field(default_factory=list)
    recent_detections: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Flags
    cloth_prepared: bool = False
    needle_parts_detected: bool = False
    
    def __post_init__(self):
        """Initialize YOLO model if not provided"""
        if self.yolo_model is None:
            self.yolo_model = YOLOSegmentationWrapper()
    
    def process_frame(self, frame: np.ndarray, detections: Optional[List[Dict]] = None) -> Dict:
        """
        Process a frame with YOLO segmentation.
        
        Args:
            frame: Input image (numpy array, BGR format)
            detections: Optional pre-computed detections. If None, YOLO will be called.
        
        Returns:
            Dict with current state, actions, and measurements
        """
        self.frame_count += 1
        
        # Run YOLO segmentation if detections not provided
        if detections is None:
            detections = self.yolo_model.predict(frame, conf=self.confidence_threshold)
        
        self.recent_detections.append(detections)
        
        # Extract detected objects
        detected_objects = {d['class'] for d in detections}
        
        result = {
            'frame': self.frame_count,
            'state': self.current_state,
            'actions': [],
            'measurements': {},
            'detections': detections
        }
        
        # State machine logic
        if self.current_state == State.IDLE:
            result.update(self._handle_idle(detected_objects, detections))
            
        elif self.current_state == State.SCISSOR_DETECTED:
            result.update(self._handle_scissor_detected(detected_objects, detections))
            
        elif self.current_state == State.CLOTH_PREPARED:
            result.update(self._handle_cloth_prepared(detected_objects, detections))
            
        elif self.current_state == State.STITCHING:
            result.update(self._handle_stitching(detected_objects, detections))
            
        elif self.current_state == State.THREAD_360_CHECK:
            result.update(self._handle_thread_360(detected_objects, detections))
            
        elif self.current_state in [State.TASK_8_CLOTH_CENTERED, State.TASK_9_CLOTH_LATERAL]:
            result.update(self._handle_tasks(detected_objects, detections))
        
        result['state'] = self.current_state
        return result
    
    def _handle_idle(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """SOP 30: Detect scissor to start preparing cloth"""
        if DetectedObject.SCISSOR.value in detected_objects:
            self.scissor_frames += 1
            if self.scissor_frames >= self.scissor_frame_threshold:
                self.current_state = State.SCISSOR_DETECTED
                return {'actions': ['Scissor detected - preparing cloth']}
        else:
            self.scissor_frames = 0
        return {'actions': []}
    
    def _handle_scissor_detected(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """Continue tracking scissor presence"""
        if DetectedObject.SCISSOR.value in detected_objects:
            self.scissor_frames += 1
            return {'actions': ['Cloth preparation in progress']}
        else:
            self.cloth_prepared = True
            self.current_state = State.CLOTH_PREPARED
            return {'actions': ['Cloth prepared - waiting for needle']}
    
    def _handle_cloth_prepared(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """After scissor disappears and needle detected, start tasks 8, 9, 10"""
        if DetectedObject.NEEDLE.value in detected_objects:
            cloth_position = self._detect_cloth_position(detections)
            
            if cloth_position == 'centered_circle':
                self.current_state = State.TASK_8_CLOTH_CENTERED
                return {'actions': ['Starting Task 8 - Cloth centered and circle']}
            elif cloth_position == 'lateral':
                self.current_state = State.TASK_9_CLOTH_LATERAL
                return {'actions': ['Starting Task 9 - Cloth lateral']}
            else:
                self.current_state = State.STITCHING
                return {'actions': ['Starting Task 10 - Stitching detection']}
        return {'actions': []}
    
    def _handle_stitching(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """SOP 10: Rule 16-17 - Track stitches and check for 360 after 7-8 stitches"""
        actions = []
        
        needle_detections = [d for d in detections if d['class'] == DetectedObject.NEEDLE.value]
        
        if needle_detections:
            for nd in needle_detections:
                if nd.get('confidence', 1.0) < 0.7 or self._is_partial_detection(nd):
                    stitch = StitchData(
                        frame_number=self.frame_count,
                        position=self._get_center(nd['bbox']),
                        confidence=nd.get('confidence', 1.0),
                        mask=nd.get('mask')
                    )
                    self.stitches.append(stitch)
                    actions.append(f'Stitch #{len(self.stitches)} detected and saved')
        
        if len(self.stitches) >= self.stitch_count_threshold:
            if DetectedObject.THREAD.value in detected_objects:
                self.current_state = State.THREAD_360_CHECK
                measurements = self._compute_stitch_measurements()
                actions.append('7-8 stitches completed - checking for 360° thread')
                return {
                    'actions': actions,
                    'measurements': measurements
                }
        
        return {'actions': actions}
    
    def _handle_thread_360(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """Handle 360° check after detecting thread"""
        thread_detections = [d for d in detections if d['class'] == DetectedObject.THREAD.value]
        
        if thread_detections:
            # Analyze thread mask for 360° coverage
            thread = thread_detections[0]
            if 'mask' in thread and thread['mask'] is not None:
                coverage = self._check_360_coverage(thread['mask'])
                return {
                    'actions': [f'Performing 360° thread check - Coverage: {coverage:.1f}%'],
                    'measurements': self._compute_stitch_measurements()
                }
        
        return {
            'actions': ['Performing 360° thread check'],
            'measurements': self._compute_stitch_measurements()
        }
    
    def _handle_tasks(self, detected_objects: set, detections: List[Dict]) -> Dict:
        """Handle Task 8 and Task 9 execution"""
        if self.current_state == State.TASK_8_CLOTH_CENTERED:
            return {'actions': ['Executing Task 8 - Centered circular cloth processing']}
        elif self.current_state == State.TASK_9_CLOTH_LATERAL:
            return {'actions': ['Executing Task 9 - Lateral cloth processing']}
        return {'actions': []}
    
    def _detect_cloth_position(self, detections: List[Dict]) -> str:
        """Determine if cloth is centered/circle or lateral using segmentation mask"""
        cloth_detections = [d for d in detections if d['class'] == DetectedObject.CLOTH.value]
        
        if not cloth_detections:
            return 'unknown'
        
        cloth = cloth_detections[0]
        bbox = cloth['bbox']
        
        # Use segmentation mask if available for better analysis
        if 'mask' in cloth and cloth['mask'] is not None:
            mask = cloth['mask']
            circularity = self._compute_circularity(mask)
            center_x = self._get_mask_center(mask)[0]
            
            # Normalized to 0-1 range
            if 0.4 < center_x < 0.6 and circularity > 0.7:
                return 'centered_circle'
            elif center_x < 0.3 or center_x > 0.7:
                return 'lateral'
        else:
            # Fallback to bbox analysis
            center_x = (bbox[0] + bbox[2]) / 2
            aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
            
            if 0.4 < center_x < 0.6 and 0.8 < aspect_ratio < 1.2:
                return 'centered_circle'
            elif center_x < 0.3 or center_x > 0.7:
                return 'lateral'
        
        return 'unknown'
    
    def _is_partial_detection(self, detection: Dict) -> bool:
        """Check if detection appears to be partial using mask area"""
        if 'area' in detection:
            return detection['area'] < 500  # Small area suggests partial detection
        
        bbox = detection['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return area < 0.01
    
    def _get_center(self, bbox: tuple) -> tuple:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _get_mask_center(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute center of mass from segmentation mask"""
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            return (0.5, 0.5)
        
        center_x = np.mean(x_indices) / mask.shape[1]
        center_y = np.mean(y_indices) / mask.shape[0]
        return (center_x, center_y)
    
    def _compute_circularity(self, mask: np.ndarray) -> float:
        """Compute circularity of mask (1.0 = perfect circle)"""
        area = np.sum(mask > 0)
        if area == 0:
            return 0.0
        
        # Find perimeter
        from scipy import ndimage
        eroded = ndimage.binary_erosion(mask)
        perimeter = np.sum(mask > eroded)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)
    
    def _check_360_coverage(self, mask: np.ndarray) -> float:
        """Check 360° coverage of thread mask around center"""
        center = self._get_mask_center(mask)
        y_indices, x_indices = np.where(mask > 0)
        
        if len(x_indices) == 0:
            return 0.0
        
        # Convert to polar coordinates
        angles = np.arctan2(
            y_indices - center[1] * mask.shape[0],
            x_indices - center[0] * mask.shape[1]
        )
        
        # Check coverage across 360 degrees
        angle_bins = np.linspace(-np.pi, np.pi, 36)
        coverage = np.histogram(angles, bins=angle_bins)[0]
        coverage_percentage = np.sum(coverage > 0) / len(angle_bins) * 100
        
        return coverage_percentage
    
    def _compute_stitch_measurements(self) -> Dict:
        """Compute measurements from saved stitches"""
        if len(self.stitches) < 2:
            return {}
        
        distances = []
        for i in range(1, len(self.stitches)):
            p1 = self.stitches[i-1].position
            p2 = self.stitches[i].position
            dist = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            distances.append(dist)
        
        return {
            'total_stitches': len(self.stitches),
            'avg_stitch_distance': sum(distances) / len(distances) if distances else 0,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0
        }
    
    def reset(self):
        """Reset the state machine"""
        self.current_state = State.IDLE
        self.frame_count = 0
        self.scissor_frames = 0
        self.stitches.clear()
        self.recent_detections.clear()
        self.cloth_prepared = False
        self.needle_parts_detected = False


# 1. Using with OpenCV video capture:
import cv2

# Example usage
if __name__ == "__main__":

    sm = TaskDetectionStateMachine()
    sm.yolo_model = YOLOSegmentationWrapper('edwards_insipiris_best_14jan.pt')

    cap = cv2.VideoCapture("E:/Resources/Novathena/INSIPIRIS/operation 30_A.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 3 != 0 :
            continue

        frame = cv2.resize(frame,(1280,720))

        result = sm.process_frame(frame)
        print(f"State: {result['state'].name}")
        print(f"Actions: {result['actions']}")
        
        # Draw detections on frame
        for det in result['detections']:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, det['class'], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('YOLO Segmentation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

