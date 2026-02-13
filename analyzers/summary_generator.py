"""
Summary Generator using Groq Scout Model
Generates layman's language summaries of video quality analysis
"""
import json
from typing import Dict, Any, Optional
from groq import Groq
import numpy as np
import config


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


class SummaryGenerator:
    """Generate layman's language summaries using Groq Scout model"""
    
    def __init__(self):
        """Initialize the summary generator"""
        self.api_key = config.GROQ_CONFIG["api_key"]
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=self.api_key)
        self.model = config.GROQ_CONFIG["model"]
        self.temperature = config.GROQ_CONFIG["temperature"]
        self.max_tokens = config.GROQ_CONFIG["max_tokens"]
    
    def generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a layman's language summary of the video analysis
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            Dictionary containing the summary and metadata
        """
        try:
            # Create a simplified version of results for the prompt
            simplified_results = self._simplify_results(analysis_results)
            
            # Create the prompt
            prompt = self._create_prompt(simplified_results)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video quality expert who explains technical analysis results in simple, easy-to-understand language for non-technical users. Be concise, clear, and helpful."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            summary_text = response.choices[0].message.content
            
            return {
                "summary": summary_text,
                "model_used": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "model_used": self.model,
                "error": str(e)
            }
    
    def _simplify_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify results for the prompt"""
        simplified = {
            "metadata": {
                "resolution": f"{results['metadata']['width']}x{results['metadata']['height']}",
                "duration": f"{results['metadata']['duration']:.1f}s",
                "fps": f"{results['metadata']['fps']:.1f}",
                "frames": results['metadata']['frame_count']
            },
            "quality": {
                "score": results['quality']['quality_score'],
                "interpretation": results['quality']['interpretation'],
                "description": results['quality']['description']
            },
            "resolution_analysis": {
                "sharpness_quality": results['resolution']['sharpness_quality'],
                "average_sharpness": results['resolution']['average_sharpness'],
                "blur_quality": results['resolution']['blur_quality'],
                "average_blur": results['resolution']['average_blur'],
                "has_excessive_blur": results['resolution']['has_excessive_blur'],
                "potentially_upscaled": results['resolution']['potentially_upscaled']
            },
            "temporal": {
                "overall_quality": results['temporal']['overall_temporal_quality'],
                "has_flickering": results['temporal']['flickering']['has_flickering'],
                "flickering_severity": results['temporal']['flickering']['severity'],
                "has_jitter": results['temporal']['jitter']['has_jitter'],
                "jitter_severity": results['temporal']['jitter']['severity']
            },
            "blank_frames": {
                "quality": results['blank_frames']['quality'],
                "has_issues": results['blank_frames']['has_issues'],
                "black_frames": results['blank_frames']['num_black_frames'],
                "white_frames": results['blank_frames']['num_white_frames'],
                "frozen_frames": results['blank_frames']['num_frozen_frames']
            }
        }
        
        # Convert all NumPy types to native Python types
        return convert_to_native_types(simplified)
    
    def _create_prompt(self, simplified_results: Dict[str, Any]) -> str:
        """Create the prompt for the LLM"""
        results_json = json.dumps(simplified_results, indent=2)
        
        return f"""I have analyzed a video and got the following technical results:

{results_json}

Please provide a comprehensive but easy-to-understand summary of this video's quality in layman's language. Structure your response as follows:

1. **Overall Assessment**: Start with a one-sentence verdict about the video quality
2. **Key Findings**: Explain the main quality aspects (sharpness, stability, etc.) in simple terms
3. **Issues Found** (if any): Describe any problems detected (flickering, jitter, blank frames, etc.) and what they mean for viewing experience
4. **Bottom Line**: A final recommendation or conclusion

Keep it concise (4-6 paragraphs max), avoid technical jargon, and focus on what matters to someone watching the video."""
