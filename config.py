"""
Configuration file for Video Analysis Streamlit App
"""
import os
import torch

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAST_VQA_DIR = os.path.join(BASE_DIR, "FAST-VQA-and-FasterVQA")
PRETRAINED_WEIGHTS_DIR = os.path.join(BASE_DIR, "pretrained_weights")

# Storage paths
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
RESULTS_DIR = os.path.join(BASE_DIR, "videos")  # Results stored alongside videos

# Model configuration
MODEL_CONFIG = {
    "model_type": "FasterVQA",  # FasterVQA, FAST-VQA, FAST-VQA-M
    "config_path": os.path.join(FAST_VQA_DIR, "options/fast/f3dvqa-b.yml"),
    "weights_path": os.path.join(PRETRAINED_WEIGHTS_DIR, "FAST_VQA_3D_1_1_Scr.pth"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Analysis thresholds
THRESHOLDS = {
    # Quality score interpretation
    "quality": {
        "excellent": 0.75,  # >= 0.75
        "good": 0.5,        # >= 0.5
        "fair": 0.25,       # >= 0.25
        # < 0.25 is poor
    },
    
    # Flickering detection
    "flickering": {
        "brightness_variance_threshold": 15.0,  # Std dev of brightness changes
        "spike_threshold": 30.0,  # Sudden brightness change
        "frequency_threshold": 0.1,  # Periodic flickering detection
    },
    
    # Jitter detection
    "jitter": {
        "motion_variance_threshold": 5.0,  # Optical flow variance
        "ssim_drop_threshold": 0.85,  # Frame similarity threshold
    },
    
    # Blank/Black frame detection
    "blank_frames": {
        "black_threshold": 10,  # Mean pixel value for black frames
        "white_threshold": 245,  # Mean pixel value for white frames
        "low_variance_threshold": 5.0,  # Variance for frozen frames
    },
    
    # Resolution analysis
    "resolution": {
        "sharpness_threshold": 100.0,  # Laplacian variance for sharpness
        "blur_threshold": 0.35,  # Blur score threshold (0-1, higher = more blurry)
    },
}

# Streamlit UI configuration
UI_CONFIG = {
    "page_title": "Video Quality Analysis",
    "page_icon": "🎥",
    "layout": "wide",
    "max_upload_size_mb": 500,
    "supported_formats": ["mp4", "avi", "mov", "mkv"],
}

# Video processing configuration
VIDEO_CONFIG = {
    "sample_rate": 1,  # Process every Nth frame for temporal analysis
    "max_frames_for_analysis": 300,  # Limit frames for performance
}

# Groq API configuration
GROQ_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "model": "llama-3.3-70b-versatile",  # Using versatile model for better summaries
    "temperature": 0.7,
    "max_tokens": 1000,
}
