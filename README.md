# Video Quality Analysis App

A comprehensive Streamlit application for video quality analysis using **FAST-VQA** model and advanced temporal distortion detection.

## Features

### 🎯 Quality Assessment
- **FAST-VQA Model Integration**: Uses FasterVQA (FAST-VQA-B-3D) for end-to-end video quality assessment
- Quality score from 0-1 with interpretation (Excellent/Good/Fair/Poor)
- Pretrained model with state-of-the-art performance

### 📐 Resolution & Sharpness Analysis
- Actual resolution detection
- Sharpness measurement using Laplacian variance
- Edge density analysis
- Upscaling artifact detection

### ⏱️ Temporal Distortion Detection
- **Flickering Detection**:
  - Frame-to-frame brightness variance analysis
  - Frequency domain analysis (FFT) for periodic flickering
  - Brightness spike detection
- **Jitter Detection**:
  - Optical flow analysis for motion inconsistencies
  - Structural similarity (SSIM) for frame consistency
  - Sudden motion change detection

### 🎬 Blank & Frozen Frame Detection
- Black frame detection
- White/blank frame detection
- Frozen frame detection (consecutive identical frames)
- Timestamp tracking for all detected issues

### 📊 Interactive Visualizations
- Quality score gauge charts
- Temporal analysis line plots
- Issue timeline visualizations
- Frame-by-frame metrics
- Export results as JSON or text

## Installation

1. **Install dependencies**:
```bash
cd /Users/rheapatel/saanchal
pip install -r requirements_streamlit.txt
```

2. **Verify FAST-VQA setup**:
The app uses the existing FAST-VQA repository and pretrained weights already in your folder structure.

## Usage

### Run the Streamlit App

```bash
cd /Users/rheapatel/saanchal
streamlit run video_analysis_app.py
```

### Test the Analyzers

Run the test script to verify all components:

```bash
cd /Users/rheapatel/saanchal
python test_analysis.py
```

## Application Structure

```
saanchal/
├── video_analysis_app.py          # Main Streamlit application
├── config.py                       # Configuration and thresholds
├── requirements_streamlit.txt      # Python dependencies
├── test_analysis.py               # Test script
│
├── analyzers/                     # Analysis modules
│   ├── __init__.py
│   ├── quality_analyzer.py        # FAST-VQA integration
│   ├── resolution_analyzer.py     # Resolution & sharpness
│   ├── temporal_analyzer.py       # Flickering & jitter
│   └── blank_frame_detector.py    # Blank/frozen frames
│
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── video_processor.py         # Video loading & processing
│   └── visualizer.py              # Plotly visualizations
│
├── FAST-VQA-and-FasterVQA/       # FAST-VQA repository
└── pretrained_weights/            # Model weights
    └── FAST_VQA_3D_1_1_Scr.pth
```

## How It Works

### 1. Video Upload
- Upload video through Streamlit interface
- Supports MP4, AVI, MOV, MKV formats
- Video preview displayed

### 2. Analysis Pipeline
1. **Video Loading**: Extract frames using decord
2. **Quality Analysis**: Run FAST-VQA model inference
3. **Resolution Analysis**: Calculate sharpness and edge density
4. **Temporal Analysis**: Detect flickering and jitter
5. **Frame Analysis**: Identify blank/frozen frames

### 3. Results Display
- **Summary Tab**: Overall scores and video metadata
- **Quality Tab**: FAST-VQA score with gauge visualization
- **Resolution Tab**: Sharpness analysis over time
- **Temporal Tab**: Flickering and jitter detection with charts
- **Blank Frames Tab**: Timeline of detected issues

### 4. Export
- Download complete results as JSON
- Download summary report as text

## Configuration

Edit `config.py` to customize:
- Model selection (FasterVQA, FAST-VQA, FAST-VQA-M)
- Analysis thresholds
- Device (CPU/CUDA)
- UI settings

## Model Information

**FasterVQA (FAST-VQA-B-3D)**:
- 4x faster than FAST-VQA-B
- Similar performance to FAST-VQA-B
- Pretrained on Kinetics-400
- PLCC@LSVQ_test: 0.874
- PLCC@KoNViD: 0.864

## Thresholds

Default thresholds (configurable in `config.py`):

- **Quality**: 0.75 (Excellent), 0.5 (Good), 0.25 (Fair)
- **Flickering**: Brightness variance > 15.0, Spike > 30.0
- **Jitter**: Motion variance > 5.0, SSIM drop < 0.85
- **Black frames**: Mean pixel < 10
- **White frames**: Mean pixel > 245
- **Frozen frames**: Frame difference < 1.0

## Performance

- Analysis time depends on video length and resolution
- Recommended: Limit to 300 frames for faster processing (configurable)
- GPU acceleration supported if CUDA available

## Troubleshooting

### Model Loading Error
- Verify pretrained weights exist at `/Users/rheapatel/saanchal/pretrained_weights/FAST_VQA_3D_1_1_Scr.pth`
- Check FAST-VQA repository is at `/Users/rheapatel/saanchal/FAST-VQA-and-FasterVQA`

### Memory Issues
- Reduce `max_frames_for_analysis` in `config.py`
- Use CPU instead of CUDA for lower memory usage

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements_streamlit.txt`
- Check Python path includes current directory

## Credits

- **FAST-VQA**: [GitHub Repository](https://github.com/QualityAssessment/FAST-VQA-and-FasterVQA)
- **Paper**: "FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling" (ECCV 2022)
- **Extension**: "Neighbourhood Representative Sampling for Efficient End-to-end Video Quality Assessment" (TPAMI 2023)
