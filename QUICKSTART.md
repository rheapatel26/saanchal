# Quick Start Guide

## Installation

```bash
cd /Users/rheapatel/saanchal

# Option 1: Run installation script
./install.sh

# Option 2: Install manually
pip install streamlit plotly pandas opencv-python scikit-image pillow torch torchvision timm einops pyyaml scipy

# Install decord (required for video processing)
# On Mac, use conda:
conda install -c conda-forge decord
```

## Launch the App

```bash
streamlit run video_analysis_app.py
```

The app will open at `http://localhost:8501`

## How to Use

1. **Upload** your video (MP4, AVI, MOV, MKV)
2. **Click** "🚀 Run Analysis"
3. **View** results in tabs:
   - Summary: Overall scores
   - Quality: FAST-VQA score
   - Resolution: Sharpness analysis
   - Temporal: Flickering & jitter
   - Blank Frames: Issue detection
4. **Export** results as JSON or text

## What Gets Analyzed

✅ **Video Quality** - FAST-VQA model score (0-1)  
✅ **Resolution** - Actual resolution, sharpness, edge density  
✅ **Flickering** - Brightness variance, periodic patterns  
✅ **Jitter** - Motion inconsistencies, frame stability  
✅ **Blank Frames** - Black, white, and frozen frames  

## Troubleshooting

**"ModuleNotFoundError: No module named 'decord'"**
→ Install decord: `conda install -c conda-forge decord`

**"CUDA out of memory"**
→ Edit `config.py`, set `device: "cpu"`

**Analysis takes too long**
→ Edit `config.py`, reduce `max_frames_for_analysis` to 100

## Files Created

- `video_analysis_app.py` - Main Streamlit app
- `config.py` - Configuration & thresholds
- `analyzers/` - Quality, resolution, temporal, blank frame analyzers
- `utils/` - Video processor & visualizer
- `README.md` - Full documentation
- `test_analysis.py` - Test script

Enjoy analyzing your videos! 🎥
