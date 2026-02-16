"""
Video Processor Utility
Handles video loading, frame extraction, and metadata extraction
"""
import os
import cv2
import numpy as np
import tempfile
from typing import Dict, List, Optional, Tuple
from pathlib import Path

def convert_to_native_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

try:
    import decord
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("Warning: decord not found. Falling back to OpenCV for video reading.")


class OpenCVVideoReader:
    """A wrapper for OpenCV capture to mimic decord.VideoReader interface"""
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range {self.num_frames}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            # Fallback for some video formats that fail seeking
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Convert BGR to RGB to match decord behavior
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def asnumpy(self):
        # This is for individual frames if someone calls .asnumpy() on the result of __getitem__
        # But our __getitem__ already returns a numpy array.
        # In decord, vr[idx] returns a decord object that has .asnumpy()
        # So we should wrap the frame in an object that has .asnumpy() or just make it work.
        pass

class DecordFallbackFrame(np.ndarray):
    """A numpy array subclass that has an .asnumpy() method to mimic decord frames"""
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def asnumpy(self):
        return np.asarray(self)


class OpenCVVideoReaderCompatible(OpenCVVideoReader):
    """Wrapper that returns frames with .asnumpy() method"""
    def __getitem__(self, idx):
        frame = super().__getitem__(idx)
        return DecordFallbackFrame(frame)


class VideoProcessor:
    """Handles video file processing and frame extraction"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.video_reader = None
        self.cv_capture = None
        self._load_video()
        
    def _load_video(self):
        """Load video using decord and OpenCV"""
        try:
            # Load with decord for efficient frame access if available
            if HAS_DECORD:
                self.video_reader = decord.VideoReader(self.video_path)
            else:
                self.video_reader = OpenCVVideoReaderCompatible(self.video_path)
            
            # Also load with OpenCV for metadata
            self.cv_capture = cv2.VideoCapture(self.video_path)
            
            if not self.cv_capture.isOpened():
                raise ValueError(f"Could not open video: {self.video_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading video: {str(e)}")
    
    def get_metadata(self) -> Dict:
        """
        Extract video metadata
        
        Returns:
            Dictionary containing video metadata
        """
        if self.cv_capture is None:
            raise RuntimeError("Video not loaded")
        
        metadata = {
            "width": int(self.cv_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cv_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cv_capture.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self.cv_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": 0.0,
            "codec": "",
            "file_size_mb": 0.0,
            "bitrate": None,
        }
        
        # Calculate duration
        if metadata["fps"] > 0:
            metadata["duration"] = metadata["frame_count"] / metadata["fps"]
        
        # Get codec
        fourcc = int(self.cv_capture.get(cv2.CAP_PROP_FOURCC))
        metadata["codec"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Get file size and calculate bitrate
        if os.path.exists(self.video_path):
            file_size_bytes = os.path.getsize(self.video_path)
            metadata["file_size_mb"] = file_size_bytes / (1024 * 1024)
            
            # Calculate bitrate (bits per second)
            if metadata["duration"] > 0:
                metadata["bitrate"] = int((file_size_bytes * 8) / metadata["duration"])
        
        return metadata
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get a specific frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Frame as numpy array (H, W, C) in RGB format
        """
        if self.video_reader is None:
            raise RuntimeError("Video not loaded")
        
        frame = self.video_reader[frame_idx]
        if hasattr(frame, 'asnumpy'):
            frame = frame.asnumpy()
        return np.array(frame, dtype=np.uint8)
    
    def get_frames(self, frame_indices: List[int]) -> np.ndarray:
        """
        Get multiple frames
        
        Args:
            frame_indices: List of frame indices
            
        Returns:
            Frames as numpy array (N, H, W, C) in RGB format
        """
        if self.video_reader is None:
            raise RuntimeError("Video not loaded")
        
        frames = []
        for idx in frame_indices:
            frame = self.video_reader[idx]
            if hasattr(frame, 'asnumpy'):
                frame = frame.asnumpy()
            frames.append(np.array(frame, dtype=np.uint8))
        return np.stack(frames, axis=0)
    
    def get_all_frames(self, max_frames: Optional[int] = None) -> np.ndarray:
        """
        Get all frames from video
        
        Args:
            max_frames: Maximum number of frames to extract (for performance)
            
        Returns:
            Frames as numpy array (N, H, W, C) in RGB format
        """
        if self.video_reader is None:
            raise RuntimeError("Video not loaded")
        
        total_frames = len(self.video_reader)
        
        if max_frames and total_frames > max_frames:
            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            return self.get_frames(indices.tolist())
        else:
            # Get all frames
            frames = []
            for i in range(total_frames):
                frame = self.video_reader[i]
                if hasattr(frame, 'asnumpy'):
                    frame = frame.asnumpy()
                frames.append(np.array(frame, dtype=np.uint8))
            return np.stack(frames, axis=0)
    
    def sample_frames(self, num_frames: int, method: str = "uniform") -> Tuple[np.ndarray, List[int]]:
        """
        Sample frames from video
        
        Args:
            num_frames: Number of frames to sample
            method: Sampling method ('uniform', 'random')
            
        Returns:
            Tuple of (frames array, frame indices)
        """
        if self.video_reader is None:
            raise RuntimeError("Video not loaded")
        
        total_frames = len(self.video_reader)
        
        if method == "uniform":
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif method == "random":
            indices = np.sort(np.random.choice(total_frames, num_frames, replace=False))
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        frames = self.get_frames(indices.tolist())
        return frames, indices.tolist()
    
    def get_frame_cv(self, frame_idx: int) -> np.ndarray:
        """
        Get frame using OpenCV (in BGR format)
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Frame as numpy array (H, W, C) in BGR format
        """
        if self.cv_capture is None:
            raise RuntimeError("Video not loaded")
        
        self.cv_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cv_capture.read()
        
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_idx}")
        
        return frame
    
    def __len__(self) -> int:
        """Get total number of frames"""
        if self.video_reader is None:
            raise RuntimeError("Video not loaded")
        return len(self.video_reader)
    
    def __del__(self):
        """Cleanup resources"""
        if self.cv_capture is not None:
            self.cv_capture.release()


def save_uploaded_video(uploaded_file, videos_dir: str) -> Dict:
    """
    Save uploaded Streamlit file to persistent location
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        videos_dir: Directory to save videos
        
    Returns:
        Dictionary with file path and metadata
    """
    # Create videos directory if it doesn't exist
    os.makedirs(videos_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(uploaded_file.name).stem
    suffix = Path(uploaded_file.name).suffix
    filename = f"{original_name}_{timestamp}{suffix}"
    
    file_path = os.path.join(videos_dir, filename)
    
    # Write uploaded file to persistent location
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    # Get file metadata
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    return {
        "path": file_path,
        "filename": filename,
        "original_name": uploaded_file.name,
        "size_mb": file_size_mb,
        "upload_time": timestamp
    }


def get_video_library(videos_dir: str) -> List[Dict]:
    """
    Get list of all saved videos with metadata
    
    Args:
        videos_dir: Directory containing videos
        
    Returns:
        List of dictionaries with video metadata
    """
    if not os.path.exists(videos_dir):
        return []
    
    videos = []
    for filename in os.listdir(videos_dir):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            file_path = os.path.join(videos_dir, filename)
            results_path = file_path.replace(Path(filename).suffix, '.json')
            
            # Get file metadata
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mod_time = os.path.getmtime(file_path)
            from datetime import datetime
            upload_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            
            videos.append({
                "filename": filename,
                "path": file_path,
                "size_mb": file_size_mb,
                "upload_date": upload_date,
                "has_results": os.path.exists(results_path),
                "results_path": results_path if os.path.exists(results_path) else None
            })
    
    # Sort by modification time (newest first)
    videos.sort(key=lambda x: os.path.getmtime(x["path"]), reverse=True)
    
    return videos


def save_analysis_results(video_path: str, results: Dict):
    """
    Save analysis results as JSON alongside video
    
    Args:
        video_path: Path to video file
        results: Analysis results dictionary
    """
    import json
    results_path = video_path.replace(Path(video_path).suffix, '.json')
    
    # Convert NumPy types to native Python types
    clean_results = convert_to_native_types(results)
    
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)


def load_analysis_results(video_path: str) -> Optional[Dict]:
    """
    Load previously saved analysis results
    
    Args:
        video_path: Path to video file
        
    Returns:
        Analysis results dictionary or None if not found
    """
    import json
    if not video_path:
        return None
    results_path = video_path.replace(Path(video_path).suffix, '.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def delete_video(video_path: str):
    """
    Delete video and associated results
    
    Args:
        video_path: Path to video file
    """
    # Delete video file
    if os.path.exists(video_path):
        os.remove(video_path)
    
    # Delete results file
    results_path = video_path.replace(Path(video_path).suffix, '.json')
    if os.path.exists(results_path):
        os.remove(results_path)


def cleanup_temp_file(file_path: str):
    """
    Remove temporary file (kept for backward compatibility)
    
    Args:
        file_path: Path to temporary file
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not remove temporary file {file_path}: {e}")
