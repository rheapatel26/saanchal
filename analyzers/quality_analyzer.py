"""
Quality Analyzer using FAST-VQA Model
Performs video quality assessment using the FasterVQA model
"""
import sys
import os
import yaml
import torch
import numpy as np
from typing import Dict, Optional
from unittest.mock import MagicMock

# Add current directory and FAST-VQA to path
current_dir = os.path.dirname(os.path.abspath(__file__))
saanchal_dir = os.path.dirname(current_dir)
sys.path.insert(0, saanchal_dir)
sys.path.insert(0, os.path.join(saanchal_dir, 'FAST-VQA-and-FasterVQA'))

# Mock decord if not available
try:
    import decord
except ImportError:
    mock_decord = MagicMock()
    mock_decord.bridge = MagicMock()
    
    # We need a VideoReader that works
    from utils.video_processor import OpenCVVideoReaderCompatible
    
    class MockVideoReader(OpenCVVideoReaderCompatible):
        def __getitem__(self, idx):
            frame = super().__getitem__(idx)
            # In decord with torch bridge, it returns torch tensor [H,W,C]
            if hasattr(frame, 'asnumpy'):
                frame = frame.asnumpy()
            return torch.from_numpy(np.array(frame, dtype=np.uint8))
            
    mock_decord.VideoReader = MockVideoReader
    sys.modules["decord"] = mock_decord

from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import config


class QualityAnalyzer:
    """Analyzes video quality using FAST-VQA model"""
    
    def __init__(self, model_type: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize quality analyzer
        
        Args:
            model_type: Model type (FasterVQA, FAST-VQA, etc.)
            device: Device to use (cuda/cpu)
        """
        self.model_type = model_type or config.MODEL_CONFIG["model_type"]
        self.device = device or config.MODEL_CONFIG["device"]
        self.config_path = config.MODEL_CONFIG["config_path"]
        self.weights_path = config.MODEL_CONFIG["weights_path"]
        
        self.evaluator = None
        self.opt = None
        self._load_model()
    
    def _load_model(self):
        """Load FAST-VQA model"""
        try:
            # Load configuration
            with open(self.config_path, "r") as f:
                self.opt = yaml.safe_load(f)
            
            # Initialize model
            self.evaluator = DiViDeAddEvaluator(**self.opt["model"]["args"]).to(self.device)
            
            # Load pretrained weights
            state_dict = torch.load(self.weights_path, map_location=self.device)
            self.evaluator.load_state_dict(state_dict["state_dict"])
            self.evaluator.eval()
            
            print(f"✓ Loaded {self.model_type} model successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _sigmoid_rescale(self, score: float) -> float:
        """
        Rescale raw score to 0-1 range using sigmoid
        
        Args:
            score: Raw model output score
            
        Returns:
            Rescaled score in [0, 1]
        """
        mean_stds = {
            "FasterVQA": (0.14759505, 0.03613452),
            "FasterVQA-MS": (0.15218826, 0.03230298),
            "FasterVQA-MT": (0.14699507, 0.036453716),
            "FAST-VQA": (-0.110198185, 0.04178565),
            "FAST-VQA-M": (0.023889644, 0.030781006),
        }
        
        mean, std = mean_stds.get(self.model_type, mean_stds["FasterVQA"])
        x = (score - mean) / std
        rescaled = 1 / (1 + np.exp(-x))
        return rescaled
    
    def _prepare_video_samples(self, vreader) -> Dict:
        """
        Prepare video samples for model input
        
        Args:
            vreader: Decord-compatible VideoReader object
            
        Returns:
            Dictionary of video samples
        """
        vsamples = {}
        t_data_opt = self.opt["data"]["val-kv1k"]["args"]
        s_data_opt = self.opt["data"]["val-kv1k"]["args"]["sample_types"]
        
        for sample_type, sample_args in s_data_opt.items():
            # Sample temporally
            if t_data_opt.get("t_frag", 1) > 1:
                sampler = FragmentSampleFrames(
                    fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                    fragments_t=sample_args.get("t_frag", 1),
                    num_clips=sample_args.get("num_clips", 1),
                )
            else:
                sampler = SampleFrames(
                    clip_len=sample_args["clip_len"],
                    num_clips=sample_args["num_clips"]
                )
            
            num_clips = sample_args.get("num_clips", 1)
            frames = sampler(len(vreader))
            
            # Get frames
            frame_indices = np.unique(frames)
            frame_dict = {}
            for idx in frame_indices:
                frame = vreader[idx]
                if hasattr(frame, 'asnumpy'):
                    frame = frame.asnumpy()
                
                if not isinstance(frame, torch.Tensor):
                    # Explicitly convert to numpy array with numeric dtype
                    frame_np = np.array(frame, dtype=np.uint8)
                    frame = torch.from_numpy(frame_np)
                
                frame_dict[idx] = frame
            
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)
            
            # Sample spatially
            sampled_video = get_spatial_fragments(video, **sample_args)
            
            # Normalize
            mean = torch.FloatTensor([123.675, 116.28, 103.53])
            std = torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            
            sampled_video = sampled_video.reshape(
                sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
            ).transpose(0, 1)
            
            vsamples[sample_type] = sampled_video.to(self.device)
        
        return vsamples
    
    def analyze(self, video_reader) -> Dict:
        """
        Analyze video quality
        
        Args:
            video_reader: Decord VideoReader object
            
        Returns:
            Dictionary with quality analysis results
        """
        try:
            with torch.no_grad():
                # Prepare samples
                vsamples = self._prepare_video_samples(video_reader)
                
                # Run inference
                result = self.evaluator(vsamples)
                raw_score = result.mean().item()
                
                # Rescale to 0-1
                quality_score = self._sigmoid_rescale(raw_score)
                
                # Interpret score
                if quality_score >= 0.75:
                    interpretation = "Excellent"
                    description = "Very high quality video with minimal distortions"
                elif quality_score >= 0.5:
                    interpretation = "Good"
                    description = "Good quality video with minor distortions"
                elif quality_score >= 0.25:
                    interpretation = "Fair"
                    description = "Acceptable quality with noticeable distortions"
                else:
                    interpretation = "Poor"
                    description = "Low quality video with significant distortions"
                
                return {
                    "quality_score": quality_score,
                    "raw_score": raw_score,
                    "interpretation": interpretation,
                    "description": description,
                    "model_used": self.model_type,
                }
                
        except Exception as e:
            raise RuntimeError(f"Quality analysis failed: {str(e)}")
    
    def get_interpretation(self, score: float) -> str:
        """
        Get interpretation for a quality score
        
        Args:
            score: Quality score (0-1)
            
        Returns:
            Interpretation string
        """
        if score >= 0.75:
            return "Excellent"
        elif score >= 0.5:
            return "Good"
        elif score >= 0.25:
            return "Fair"
        else:
            return "Poor"
