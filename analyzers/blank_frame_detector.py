"""
Blank Frame Detector
Detects blank, black, white, and frozen frames in video
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
from skimage.metrics import structural_similarity as ssim
import config


class BlankFrameDetector:
    """Detects problematic frames (blank, black, white, frozen)"""
    
    def __init__(self):
        """Initialize blank frame detector"""
        self.config = config.THRESHOLDS["blank_frames"]
        self.black_threshold = self.config["black_threshold"]
        self.white_threshold = self.config["white_threshold"]
        self.low_variance_threshold = self.config["low_variance_threshold"]
    
    def is_black_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame is black
        
        Args:
            frame: Frame as numpy array (H, W, C)
            
        Returns:
            True if frame is black
        """
        mean_value = np.mean(frame)
        return mean_value < self.black_threshold
    
    def is_white_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame is white/blank
        
        Args:
            frame: Frame as numpy array (H, W, C)
            
        Returns:
            True if frame is white
        """
        mean_value = np.mean(frame)
        return mean_value > self.white_threshold
    
    def is_low_variance_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame has low variance (potentially frozen/static)
        
        Args:
            frame: Frame as numpy array (H, W, C)
            
        Returns:
            True if frame has low variance
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        variance = np.var(gray)
        return variance < self.low_variance_threshold
    
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate difference between two frames
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Mean absolute difference
        """
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        return np.mean(diff)
    
    def detect_frozen_frames(self, frames: np.ndarray, threshold: float = 0.995) -> List[int]:
        """
        Detect frozen frames by comparing consecutive frames using SSIM
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            threshold: SSIM threshold (higher means more similar)
            
        Returns:
            List of frozen frame indices
        """
        frozen_frames = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            
            similarity = ssim(gray1, gray2, data_range=255)
            if similarity > threshold:
                frozen_frames.append(i + 1)
        
        return frozen_frames
    
    def analyze(self, frames: np.ndarray, fps: float = 30.0) -> Dict:
        """
        Analyze video for blank, black, white, and frozen frames
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            fps: Frames per second
            
        Returns:
            Dictionary with blank frame analysis results
        """
        num_frames = len(frames)
        
        black_frames = []
        white_frames = []
        low_variance_frames = []
        
        # Analyze each frame
        for i, frame in enumerate(frames):
            if self.is_black_frame(frame):
                black_frames.append(i)
            
            if self.is_white_frame(frame):
                white_frames.append(i)
            
            if self.is_low_variance_frame(frame):
                low_variance_frames.append(i)
        
        # Detect frozen frames
        frozen_frames = self.detect_frozen_frames(frames)
        
        # Calculate percentages
        black_percentage = (len(black_frames) / num_frames) * 100
        white_percentage = (len(white_frames) / num_frames) * 100
        low_variance_percentage = (len(low_variance_frames) / num_frames) * 100
        frozen_percentage = (len(frozen_frames) / num_frames) * 100
        
        # Convert frame indices to timestamps
        def frames_to_timestamps(frame_indices: List[int]) -> List[float]:
            return [idx / fps for idx in frame_indices]
        
        black_timestamps = frames_to_timestamps(black_frames)
        white_timestamps = frames_to_timestamps(white_frames)
        frozen_timestamps = frames_to_timestamps(frozen_frames)
        
        # Determine if there are issues
        has_issues = (
            len(black_frames) > 0 or
            len(white_frames) > 0 or
            len(frozen_frames) > num_frames * 0.05  # More than 5% frozen
        )
        
        # Overall quality assessment
        if not has_issues:
            quality = "Excellent"
        elif black_percentage < 1 and white_percentage < 1 and frozen_percentage < 5:
            quality = "Good"
        elif black_percentage < 5 and white_percentage < 5 and frozen_percentage < 10:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Create detailed issue list
        issues = []
        if len(black_frames) > 0:
            issues.append(f"{len(black_frames)} black frames detected ({black_percentage:.2f}%)")
        if len(white_frames) > 0:
            issues.append(f"{len(white_frames)} white frames detected ({white_percentage:.2f}%)")
        if len(frozen_frames) > 0:
            issues.append(f"{len(frozen_frames)} frozen frames detected ({frozen_percentage:.2f}%)")
        
        return {
            "has_issues": has_issues,
            "quality": quality,
            "total_frames": num_frames,
            
            # Black frames
            "black_frames": black_frames,
            "num_black_frames": len(black_frames),
            "black_percentage": black_percentage,
            "black_timestamps": black_timestamps,
            
            # White frames
            "white_frames": white_frames,
            "num_white_frames": len(white_frames),
            "white_percentage": white_percentage,
            "white_timestamps": white_timestamps,
            
            # Low variance frames
            "low_variance_frames": low_variance_frames,
            "num_low_variance_frames": len(low_variance_frames),
            "low_variance_percentage": low_variance_percentage,
            
            # Frozen frames
            "frozen_frames": frozen_frames,
            "num_frozen_frames": len(frozen_frames),
            "frozen_percentage": frozen_percentage,
            "frozen_timestamps": frozen_timestamps,
            
            # Summary
            "issues": issues,
        }
