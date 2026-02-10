#!/bin/bash

# Installation script for Video Analysis App
# This script sets up all dependencies for the Streamlit video analysis application

echo "================================================"
echo "Video Analysis App - Installation Script"
echo "================================================"

# Check if we're in the correct directory
if [ ! -d "FAST-VQA-and-FasterVQA" ]; then
    echo "Error: FAST-VQA-and-FasterVQA directory not found"
    echo "Please run this script from /Users/rheapatel/saanchal"
    exit 1
fi

echo ""
echo "Step 1: Installing Python dependencies..."
echo "-------------------------------------------"

# Install basic dependencies
pip install streamlit plotly pandas pillow pyyaml scipy scikit-image -q

echo "✓ Basic dependencies installed"

# Install PyTorch (check if already installed)
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing PyTorch..."
    pip install torch torchvision -q
    echo "✓ PyTorch installed"
else
    echo "✓ PyTorch already installed"
fi

# Install OpenCV
pip install opencv-python -q
echo "✓ OpenCV installed"

# Install other ML dependencies
pip install timm einops -q
echo "✓ ML dependencies installed"

echo ""
echo "Step 2: Installing decord (video processing)..."
echo "------------------------------------------------"

# Try to install decord
# On Mac, decord might need to be built from source or installed via conda
python -c "import decord" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Attempting to install decord..."
    
    # Try pip install with --no-deps first
    pip install decord --no-deps 2>/dev/null
    
    # Check if successful
    python -c "import decord" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ decord installed successfully"
    else
        echo "⚠️  Could not install decord via pip"
        echo ""
        echo "Please install decord manually using one of these methods:"
        echo "1. Using conda: conda install -c conda-forge decord"
        echo "2. Build from source: https://github.com/dmlc/decord"
        echo ""
        echo "Or continue without decord (some features may not work)"
    fi
else
    echo "✓ decord already installed"
fi

echo ""
echo "Step 3: Verifying FAST-VQA setup..."
echo "------------------------------------"

# Check if pretrained weights exist
if [ -f "pretrained_weights/FAST_VQA_3D_1_1_Scr.pth" ]; then
    echo "✓ Pretrained weights found"
else
    echo "⚠️  Pretrained weights not found at pretrained_weights/FAST_VQA_3D_1_1_Scr.pth"
    echo "Please download from: https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.1/FAST_VQA_3D_1_1_Scr.pth"
fi

# Check FAST-VQA config
if [ -f "FAST-VQA-and-FasterVQA/options/fast/f3dvqa-b.yml" ]; then
    echo "✓ FAST-VQA config found"
else
    echo "⚠️  FAST-VQA config not found"
fi

echo ""
echo "================================================"
echo "Installation Summary"
echo "================================================"

# Test imports
echo ""
echo "Testing imports..."

python -c "import streamlit; print('✓ streamlit')" 2>/dev/null || echo "✗ streamlit"
python -c "import torch; print('✓ torch')" 2>/dev/null || echo "✗ torch"
python -c "import cv2; print('✓ opencv')" 2>/dev/null || echo "✗ opencv"
python -c "import plotly; print('✓ plotly')" 2>/dev/null || echo "✗ plotly"
python -c "import decord; print('✓ decord')" 2>/dev/null || echo "✗ decord (optional but recommended)"

echo ""
echo "================================================"
echo "Next Steps"
echo "================================================"
echo ""
echo "1. If decord failed to install, install it manually"
echo "2. Run the test script: python test_analysis.py"
echo "3. Launch the app: streamlit run video_analysis_app.py"
echo ""
