"""
Temporal Analyzer
Detects temporal distortions including flickering and jitter
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
from scipy import signal
from skimage.metrics import structural_similarity as ssim
import config


class TemporalAnalyzer:
    """Analyzes temporal distortions in video"""
    
    def __init__(self):
        """Initialize temporal analyzer"""
        self.flicker_config = config.THRESHOLDS["flickering"]
        self.jitter_config = config.THRESHOLDS["jitter"]
    
    def calculate_brightness(self, frame: np.ndarray) -> float:
        """
        Calculate average brightness of frame
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB
            
        Returns:
            Average brightness value
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        return np.mean(gray)
    
    def detect_flickering(self, frames: np.ndarray) -> Dict:
        """
        Detect flickering by analyzing brightness variations
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            
        Returns:
            Dictionary with flickering analysis results
        """
        num_frames = len(frames)
        
        # Calculate brightness for each frame
        brightness_values = [self.calculate_brightness(frame) for frame in frames]
        brightness_array = np.array(brightness_values)
        
        # Calculate frame-to-frame brightness changes
        brightness_changes = np.diff(brightness_array)
        
        # Calculate statistics
        brightness_std = np.std(brightness_changes)
        brightness_mean_change = np.mean(np.abs(brightness_changes))
        
        # Detect sudden spikes
        spike_threshold = self.flicker_config["spike_threshold"]
        spike_frames = np.where(np.abs(brightness_changes) > spike_threshold)[0]
        
        # Frequency analysis for periodic flickering
        if len(brightness_array) > 10:
            # Apply FFT to detect periodic patterns
            fft = np.fft.fft(brightness_array)
            frequencies = np.fft.fftfreq(len(brightness_array))
            magnitude = np.abs(fft)
            
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
            dominant_frequency = abs(frequencies[dominant_freq_idx])
        else:
            dominant_frequency = 0.0
        
        # Determine if flickering is present
        has_flickering = (
            brightness_std > self.flicker_config["brightness_variance_threshold"] or
            len(spike_frames) > num_frames * 0.05 or  # More than 5% frames have spikes
            dominant_frequency > self.flicker_config["frequency_threshold"]
        )
        
        # Severity assessment
        if has_flickering:
            if brightness_std > self.flicker_config["brightness_variance_threshold"] * 2:
                severity = "Severe"
            elif brightness_std > self.flicker_config["brightness_variance_threshold"] * 1.5:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:
            severity = "None"
        
        return {
            "has_flickering": has_flickering,
            "severity": severity,
            "brightness_std": brightness_std,
            "brightness_mean_change": brightness_mean_change,
            "spike_frames": spike_frames.tolist(),
            "num_spikes": len(spike_frames),
            "dominant_frequency": dominant_frequency,
            "brightness_values": brightness_values,
        }
    
    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow between two frames
        
        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)
            
        Returns:
            Optical flow magnitude
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        return magnitude
    
    def calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate structural similarity between frames
        
        Args:
            frame1: First frame (RGB)
            frame2: Second frame (RGB)
            
        Returns:
            SSIM score (0-1, higher is more similar)
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        similarity = ssim(gray1, gray2, data_range=255)
        
        return similarity
    
    def detect_jitter(self, frames: np.ndarray) -> Dict:
        """
        Detect jitter using optical flow and frame similarity
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            
        Returns:
            Dictionary with jitter analysis results
        """
        num_frames = len(frames)
        
        # Sample frames for performance (analyze every Nth frame)
        sample_rate = max(1, num_frames // 100)  # Analyze up to 100 frame pairs
        sampled_indices = range(0, num_frames - 1, sample_rate)
        
        motion_magnitudes = []
        ssim_scores = []
        
        for i in sampled_indices:
            # Calculate optical flow
            flow_mag = self.calculate_optical_flow(frames[i], frames[i + 1])
            avg_motion = np.mean(flow_mag)
            motion_magnitudes.append(avg_motion)
            
            # Calculate SSIM
            similarity = self.calculate_frame_similarity(frames[i], frames[i + 1])
            ssim_scores.append(similarity)
        
        # Calculate statistics
        motion_std = np.std(motion_magnitudes)
        motion_mean = np.mean(motion_magnitudes)
        ssim_mean = np.mean(ssim_scores)
        ssim_min = np.min(ssim_scores)
        
        # Detect sudden motion changes (potential jitter)
        motion_changes = np.abs(np.diff(motion_magnitudes))
        sudden_changes = np.where(motion_changes > motion_mean * 2)[0]
        
        # Detect SSIM drops (frame inconsistency)
        ssim_threshold = self.jitter_config["ssim_drop_threshold"]
        ssim_drops = [i for i, score in enumerate(ssim_scores) if score < ssim_threshold]
        
        # Determine if jitter is present
        has_jitter = (
            motion_std > self.jitter_config["motion_variance_threshold"] or
            len(ssim_drops) > len(ssim_scores) * 0.1 or  # More than 10% frames
            len(sudden_changes) > len(motion_magnitudes) * 0.05
        )
        
        # Severity assessment
        if has_jitter:
            if motion_std > self.jitter_config["motion_variance_threshold"] * 2:
                severity = "Severe"
            elif motion_std > self.jitter_config["motion_variance_threshold"] * 1.5:
                severity = "Moderate"
            else:
                severity = "Mild"
        else:
            severity = "None"
        
        return {
            "has_jitter": has_jitter,
            "severity": severity,
            "motion_std": motion_std,
            "motion_mean": motion_mean,
            "ssim_mean": ssim_mean,
            "ssim_min": ssim_min,
            "num_sudden_changes": len(sudden_changes),
            "num_ssim_drops": len(ssim_drops),
            "motion_magnitudes": motion_magnitudes,
            "ssim_scores": ssim_scores,
            "sampled_indices": list(sampled_indices),
        }
    
    def analyze(self, frames: np.ndarray) -> Dict:
        """
        Perform complete temporal analysis
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            
        Returns:
            Dictionary with temporal analysis results
        """
        # Detect flickering
        flicker_results = self.detect_flickering(frames)
        
        # Detect jitter
        jitter_results = self.detect_jitter(frames)
        
        # Overall temporal quality assessment
        has_temporal_issues = flicker_results["has_flickering"] or jitter_results["has_jitter"]
        
        if not has_temporal_issues:
            overall_quality = "Excellent"
        elif (flicker_results["severity"] in ["Mild", "None"] and 
              jitter_results["severity"] in ["Mild", "None"]):
            overall_quality = "Good"
        elif (flicker_results["severity"] == "Severe" or 
              jitter_results["severity"] == "Severe"):
            overall_quality = "Poor"
        else:
            overall_quality = "Fair"
        
        return {
            "flickering": flicker_results,
            "jitter": jitter_results,
            "has_temporal_issues": has_temporal_issues,
            "overall_temporal_quality": overall_quality,
        }
