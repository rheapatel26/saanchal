"""
Resolution Analyzer
Analyzes video resolution, sharpness, and clarity
"""
import cv2
import numpy as np
from typing import Dict, List
import config


class ResolutionAnalyzer:
    """Analyzes video resolution and sharpness"""
    
    def __init__(self):
        """Initialize resolution analyzer"""
        self.sharpness_threshold = config.THRESHOLDS["resolution"]["sharpness_threshold"]
    
    def calculate_sharpness(self, frame: np.ndarray) -> float:
        """
        Calculate frame sharpness using Laplacian variance
        
        Args:
            frame: Frame as numpy array (H, W, C) in RGB or BGR
            
        Returns:
            Sharpness score (higher = sharper)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    
    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect edges in frame using Canny edge detection
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Edge map
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        return edges
    
    def calculate_effective_resolution(self, frame: np.ndarray) -> Dict:
        """
        Calculate effective resolution based on edge density
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Dictionary with effective resolution metrics
        """
        edges = self.detect_edges(frame)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Estimate effective resolution as percentage of nominal resolution
        # Higher edge density suggests better effective resolution
        effective_percentage = min(100, edge_density * 1000)  # Scale factor
        
        return {
            "edge_density": edge_density,
            "effective_percentage": effective_percentage,
        }
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame
        
        Args:
            frame: Frame as numpy array (H, W, C)
            
        Returns:
            Dictionary with frame analysis results
        """
        sharpness = self.calculate_sharpness(frame)
        effective_res = self.calculate_effective_resolution(frame)
        
        # Determine if sharp
        is_sharp = sharpness >= self.sharpness_threshold
        
        return {
            "sharpness": sharpness,
            "is_sharp": is_sharp,
            "edge_density": effective_res["edge_density"],
            "effective_percentage": effective_res["effective_percentage"],
        }
    
    def analyze(self, frames: np.ndarray, metadata: Dict) -> Dict:
        """
        Analyze video resolution and quality
        
        Args:
            frames: Video frames as numpy array (N, H, W, C)
            metadata: Video metadata dictionary
            
        Returns:
            Dictionary with resolution analysis results
        """
        num_frames = len(frames)
        
        # Analyze subset of frames for performance
        sample_indices = np.linspace(0, num_frames - 1, min(30, num_frames), dtype=int)
        
        sharpness_scores = []
        edge_densities = []
        
        for idx in sample_indices:
            frame_analysis = self.analyze_frame(frames[idx])
            sharpness_scores.append(frame_analysis["sharpness"])
            edge_densities.append(frame_analysis["edge_density"])
        
        # Calculate statistics
        avg_sharpness = np.mean(sharpness_scores)
        min_sharpness = np.min(sharpness_scores)
        max_sharpness = np.max(sharpness_scores)
        std_sharpness = np.std(sharpness_scores)
        
        avg_edge_density = np.mean(edge_densities)
        
        # Determine overall sharpness quality
        if avg_sharpness >= self.sharpness_threshold * 2:
            sharpness_quality = "Excellent"
        elif avg_sharpness >= self.sharpness_threshold:
            sharpness_quality = "Good"
        elif avg_sharpness >= self.sharpness_threshold * 0.5:
            sharpness_quality = "Fair"
        else:
            sharpness_quality = "Poor"
        
        # Check for potential upscaling (low edge density with high resolution)
        nominal_pixels = metadata["width"] * metadata["height"]
        is_potentially_upscaled = (
            nominal_pixels > 1920 * 1080 and avg_edge_density < 0.05
        )
        
        return {
            "nominal_resolution": f"{metadata['width']}x{metadata['height']}",
            "nominal_width": metadata["width"],
            "nominal_height": metadata["height"],
            "total_pixels": nominal_pixels,
            "average_sharpness": avg_sharpness,
            "min_sharpness": min_sharpness,
            "max_sharpness": max_sharpness,
            "sharpness_std": std_sharpness,
            "sharpness_quality": sharpness_quality,
            "average_edge_density": avg_edge_density,
            "potentially_upscaled": is_potentially_upscaled,
            "sharpness_scores": sharpness_scores,
            "frame_indices": sample_indices.tolist(),
        }
