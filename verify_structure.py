"""
Simple verification script to check code structure and imports
"""
import sys
import os

print("=" * 60)
print("Video Analysis App - Code Structure Verification")
print("=" * 60)

# Check directory structure
print("\n1. Checking directory structure...")
print("-" * 60)

required_files = [
    "video_analysis_app.py",
    "config.py",
    "requirements_streamlit.txt",
    "test_analysis.py",
    "README.md",
    "analyzers/__init__.py",
    "analyzers/quality_analyzer.py",
    "analyzers/resolution_analyzer.py",
    "analyzers/temporal_analyzer.py",
    "analyzers/blank_frame_detector.py",
    "utils/__init__.py",
    "utils/video_processor.py",
    "utils/visualizer.py",
]

all_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"{status} {file}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n✅ All required files exist")
else:
    print("\n⚠️  Some files are missing")

# Check FAST-VQA setup
print("\n2. Checking FAST-VQA setup...")
print("-" * 60)

fast_vqa_files = [
    "FAST-VQA-and-FasterVQA/vqa.py",
    "FAST-VQA-and-FasterVQA/options/fast/f3dvqa-b.yml",
    "pretrained_weights/FAST_VQA_3D_1_1_Scr.pth",
]

for file in fast_vqa_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"{status} {file}")

# Check Python syntax
print("\n3. Checking Python syntax...")
print("-" * 60)

python_files = [
    "video_analysis_app.py",
    "config.py",
    "analyzers/quality_analyzer.py",
    "analyzers/resolution_analyzer.py",
    "analyzers/temporal_analyzer.py",
    "analyzers/blank_frame_detector.py",
    "utils/video_processor.py",
    "utils/visualizer.py",
]

syntax_ok = True
for file in python_files:
    try:
        with open(file, 'r') as f:
            compile(f.read(), file, 'exec')
        print(f"✓ {file}")
    except SyntaxError as e:
        print(f"✗ {file}: {e}")
        syntax_ok = False

if syntax_ok:
    print("\n✅ All Python files have valid syntax")
else:
    print("\n⚠️  Some files have syntax errors")

# Check imports (without actually importing to avoid dependency issues)
print("\n4. Checking import structure...")
print("-" * 60)

import_checks = {
    "config.py": ["import os", "import torch"],
    "utils/video_processor.py": ["import cv2", "import numpy", "import decord"],
    "utils/visualizer.py": ["import plotly", "import pandas"],
    "analyzers/quality_analyzer.py": ["import torch", "import yaml"],
    "analyzers/resolution_analyzer.py": ["import cv2", "import numpy"],
    "analyzers/temporal_analyzer.py": ["import cv2", "import numpy", "from scipy"],
    "analyzers/blank_frame_detector.py": ["import cv2", "import numpy"],
}

for file, expected_imports in import_checks.items():
    with open(file, 'r') as f:
        content = f.read()
    
    missing = []
    for imp in expected_imports:
        if imp not in content:
            missing.append(imp)
    
    if missing:
        print(f"⚠️  {file}: missing {', '.join(missing)}")
    else:
        print(f"✓ {file}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("\n✅ Code structure verification complete!")
print("\nNext steps:")
print("1. Run: ./install.sh (to install dependencies)")
print("2. Run: python test_analysis.py (to test with dependencies)")
print("3. Run: streamlit run video_analysis_app.py (to launch app)")
print("")
