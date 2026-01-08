"""
Camera Manager Module

Handles multi-camera capture, synchronization, and frame management.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: str
    position: str
    source: int  # Device index or IP address
    resolution: Tuple[int, int]
    fps: int = 30


@dataclass
class CaptureFrame:
    """Container for a captured frame."""
    camera_id: str
    frame: np.ndarray
    timestamp: datetime
    frame_number: int


class CameraManager:
    """
    Manages multiple cameras for synchronized capture.
    
    Features:
    - Multi-camera initialization and control
    - Synchronized capture across all cameras
    - Frame buffering and queue management
    - Automatic reconnection on failure
    """
    
    def __init__(
        self,
        cameras: List[CameraConfig],
        sync_timeout_ms: int = 100,
        buffer_size: int = 3
    ):
        """
        Initialize camera manager.
        
        Args:
            cameras: List of camera configurations
            sync_timeout_ms: Maximum time to wait for all cameras to capture
            buffer_size: Frame buffer size per camera
        """
        self.camera_configs = {cam.id: cam for cam in cameras}
        self.sync_timeout_ms = sync_timeout_ms
        self.buffer_size = buffer_size
        
        # Camera handles
        self._cameras: Dict[str, cv2.VideoCapture] = {}
        self._frame_counts: Dict[str, int] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._capture_queues: Dict[str, Queue] = {}
        
        # State
        self._is_running = False
        
        logger.info(f"Initializing camera manager with {len(cameras)} cameras")
    
    def initialize(self) -> bool:
        """
        Initialize all cameras.
        
        Returns:
            True if all cameras initialized successfully
        """
        success = True
        
        for cam_id, config in self.camera_configs.items():
            if not self._init_camera(cam_id, config):
                success = False
                logger.error(f"Failed to initialize camera {cam_id}")
        
        if success:
            logger.info("All cameras initialized successfully")
        
        return success
    
    def _init_camera(self, cam_id: str, config: CameraConfig) -> bool:
        """Initialize a single camera."""
        try:
            # Create capture device
            cap = cv2.VideoCapture(config.source)
            
            if not cap.isOpened():
                logger.error(f"Could not open camera {cam_id} at source {config.source}")
                return False
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            
            # Set framerate
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            
            # Set buffer size (reduce latency)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Could not read from camera {cam_id}")
                cap.release()
                return False
            
            # Store camera
            self._cameras[cam_id] = cap
            self._frame_counts[cam_id] = 0
            self._capture_queues[cam_id] = Queue(maxsize=self.buffer_size)
            
            logger.info(
                f"Camera {cam_id} initialized: {frame.shape[1]}x{frame.shape[0]} "
                f"at source {config.source}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera {cam_id}: {e}")
            return False
    
    def capture_all(self) -> Dict[str, CaptureFrame]:
        """
        Capture frames from all cameras synchronously.
        
        Returns:
            Dictionary mapping camera_id to CaptureFrame
        """
        captures = {}
        timestamp = datetime.utcnow()
        
        with self._lock:
            for cam_id, cap in self._cameras.items():
                try:
                    ret, frame = cap.read()
                    
                    if ret:
                        self._frame_counts[cam_id] += 1
                        captures[cam_id] = CaptureFrame(
                            camera_id=cam_id,
                            frame=frame,
                            timestamp=timestamp,
                            frame_number=self._frame_counts[cam_id]
                        )
                    else:
                        logger.warning(f"Failed to capture from camera {cam_id}")
                        # Attempt reconnection
                        self._reconnect_camera(cam_id)
                        
                except Exception as e:
                    logger.error(f"Error capturing from camera {cam_id}: {e}")
        
        return captures
    
    def capture_single(self, camera_id: str) -> Optional[CaptureFrame]:
        """
        Capture a frame from a single camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            CaptureFrame if successful, None otherwise
        """
        if camera_id not in self._cameras:
            logger.error(f"Camera {camera_id} not found")
            return None
        
        with self._lock:
            cap = self._cameras[camera_id]
            try:
                ret, frame = cap.read()
                
                if ret:
                    self._frame_counts[camera_id] += 1
                    return CaptureFrame(
                        camera_id=camera_id,
                        frame=frame,
                        timestamp=datetime.utcnow(),
                        frame_number=self._frame_counts[camera_id]
                    )
                else:
                    logger.warning(f"Failed to capture from camera {camera_id}")
                    self._reconnect_camera(camera_id)
                    return None
                    
            except Exception as e:
                logger.error(f"Error capturing from camera {camera_id}: {e}")
                return None
    
    def _reconnect_camera(self, cam_id: str):
        """Attempt to reconnect a camera."""
        if cam_id not in self.camera_configs:
            return
        
        config = self.camera_configs[cam_id]
        
        # Release existing
        if cam_id in self._cameras:
            try:
                self._cameras[cam_id].release()
            except:
                pass
        
        # Try to reinitialize
        logger.info(f"Attempting to reconnect camera {cam_id}")
        self._init_camera(cam_id, config)
    
    def get_frame_images(
        self,
        captures: Dict[str, CaptureFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Extract raw images from capture frames.
        
        Args:
            captures: Dictionary of CaptureFrame objects
            
        Returns:
            Dictionary mapping camera_id to numpy array
        """
        return {
            cam_id: capture.frame
            for cam_id, capture in captures.items()
        }
    
    def release(self):
        """Release all camera resources."""
        with self._lock:
            for cam_id, cap in self._cameras.items():
                try:
                    cap.release()
                    logger.info(f"Released camera {cam_id}")
                except Exception as e:
                    logger.error(f"Error releasing camera {cam_id}: {e}")
            
            self._cameras.clear()
            self._frame_counts.clear()
    
    def get_camera_status(self) -> Dict[str, bool]:
        """
        Get status of all cameras.
        
        Returns:
            Dictionary mapping camera_id to connected status
        """
        status = {}
        
        with self._lock:
            for cam_id in self.camera_configs:
                if cam_id in self._cameras:
                    status[cam_id] = self._cameras[cam_id].isOpened()
                else:
                    status[cam_id] = False
        
        return status
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class SimulatedCameraManager(CameraManager):
    """
    Simulated camera manager for testing without physical cameras.
    
    Generates random or file-based test images.
    """
    
    def __init__(
        self,
        cameras: List[CameraConfig],
        test_images_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize simulated camera manager.
        
        Args:
            cameras: List of camera configurations
            test_images_dir: Optional directory with test images
        """
        super().__init__(cameras, **kwargs)
        self.test_images_dir = test_images_dir
        self._test_images: Dict[str, List[np.ndarray]] = {}
        self._image_indices: Dict[str, int] = {}
    
    def initialize(self) -> bool:
        """Initialize simulated cameras."""
        for cam_id, config in self.camera_configs.items():
            # Create placeholder for simulated images
            self._test_images[cam_id] = []
            self._image_indices[cam_id] = 0
            self._frame_counts[cam_id] = 0
            
            # Load test images if directory provided
            if self.test_images_dir:
                self._load_test_images(cam_id)
            
            logger.info(f"Simulated camera {cam_id} initialized")
        
        return True
    
    def _load_test_images(self, cam_id: str):
        """Load test images for a camera."""
        import os
        
        cam_dir = os.path.join(self.test_images_dir, cam_id)
        if not os.path.exists(cam_dir):
            return
        
        for filename in sorted(os.listdir(cam_dir)):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                filepath = os.path.join(cam_dir, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    self._test_images[cam_id].append(img)
        
        logger.info(f"Loaded {len(self._test_images[cam_id])} test images for {cam_id}")
    
    def capture_all(self) -> Dict[str, CaptureFrame]:
        """Capture simulated frames from all cameras."""
        captures = {}
        timestamp = datetime.utcnow()
        
        for cam_id, config in self.camera_configs.items():
            frame = self._generate_frame(cam_id, config)
            self._frame_counts[cam_id] += 1
            
            captures[cam_id] = CaptureFrame(
                camera_id=cam_id,
                frame=frame,
                timestamp=timestamp,
                frame_number=self._frame_counts[cam_id]
            )
        
        return captures
    
    def _generate_frame(
        self,
        cam_id: str,
        config: CameraConfig
    ) -> np.ndarray:
        """Generate a frame for a simulated camera."""
        # If test images available, use those
        if self._test_images.get(cam_id):
            idx = self._image_indices[cam_id] % len(self._test_images[cam_id])
            self._image_indices[cam_id] += 1
            return self._test_images[cam_id][idx].copy()
        
        # Otherwise generate a random frame
        frame = np.random.randint(
            0, 255,
            (config.resolution[1], config.resolution[0], 3),
            dtype=np.uint8
        )
        
        # Add camera ID label
        cv2.putText(
            frame,
            f"Camera: {cam_id}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )
        
        return frame
    
    def release(self):
        """Release simulated cameras."""
        self._test_images.clear()
        self._image_indices.clear()
        self._frame_counts.clear()
        logger.info("Simulated cameras released")


def create_camera_manager(
    config: Dict,
    simulated: bool = False
) -> CameraManager:
    """
    Factory function to create a CameraManager from config.
    
    Args:
        config: Configuration dictionary
        simulated: If True, create simulated cameras for testing
        
    Returns:
        Configured CameraManager instance
    """
    camera_config = config.get("cameras", {})
    
    # Build camera configs
    cameras = []
    for cam_data in camera_config.get("devices", []):
        cameras.append(CameraConfig(
            id=cam_data["id"],
            position=cam_data.get("position", "unknown"),
            source=cam_data.get("source", 0),
            resolution=tuple(cam_data.get("resolution", [1920, 1080])),
            fps=cam_data.get("fps", 30)
        ))
    
    capture_config = camera_config.get("capture", {})
    
    if simulated:
        return SimulatedCameraManager(
            cameras=cameras,
            sync_timeout_ms=capture_config.get("sync_timeout_ms", 100),
            buffer_size=capture_config.get("buffer_size", 3)
        )
    else:
        return CameraManager(
            cameras=cameras,
            sync_timeout_ms=capture_config.get("sync_timeout_ms", 100),
            buffer_size=capture_config.get("buffer_size", 3)
        )
