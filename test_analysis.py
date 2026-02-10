"""
Test script to verify all analyzers work correctly
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from utils import VideoProcessor
from analyzers import QualityAnalyzer, ResolutionAnalyzer, TemporalAnalyzer, BlankFrameDetector


def test_video_processor():
    """Test video processor"""
    print("\n=== Testing Video Processor ===")
    video_path = "video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        vp = VideoProcessor(video_path)
        metadata = vp.get_metadata()
        print(f"✓ Video loaded successfully")
        print(f"  Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  Frames: {metadata['frame_count']}")
        print(f"  Duration: {metadata['duration']:.2f}s")
        print(f"  FPS: {metadata['fps']:.2f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_quality_analyzer():
    """Test quality analyzer"""
    print("\n=== Testing Quality Analyzer ===")
    
    try:
        qa = QualityAnalyzer()
        print("✓ Quality analyzer initialized")
        print(f"  Model: {qa.model_type}")
        print(f"  Device: {qa.device}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_resolution_analyzer():
    """Test resolution analyzer"""
    print("\n=== Testing Resolution Analyzer ===")
    
    try:
        ra = ResolutionAnalyzer()
        print("✓ Resolution analyzer initialized")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_temporal_analyzer():
    """Test temporal analyzer"""
    print("\n=== Testing Temporal Analyzer ===")
    
    try:
        ta = TemporalAnalyzer()
        print("✓ Temporal analyzer initialized")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_blank_frame_detector():
    """Test blank frame detector"""
    print("\n=== Testing Blank Frame Detector ===")
    
    try:
        bfd = BlankFrameDetector()
        print("✓ Blank frame detector initialized")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_full_analysis():
    """Test full analysis pipeline"""
    print("\n=== Testing Full Analysis Pipeline ===")
    video_path = "video.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠️ Skipping full analysis - video file not found: {video_path}")
        return True
    
    try:
        # Load video
        print("Loading video...")
        vp = VideoProcessor(video_path)
        metadata = vp.get_metadata()
        
        # Load frames (limited for testing)
        print("Extracting frames...")
        frames = vp.get_all_frames(max_frames=50)
        print(f"  Loaded {len(frames)} frames")
        
        # Quality analysis
        print("Running quality analysis...")
        qa = QualityAnalyzer()
        quality_results = qa.analyze(vp.video_reader)
        print(f"  ✓ Quality Score: {quality_results['quality_score']:.4f} ({quality_results['interpretation']})")
        
        # Resolution analysis
        print("Running resolution analysis...")
        ra = ResolutionAnalyzer()
        resolution_results = ra.analyze(frames, metadata)
        print(f"  ✓ Sharpness: {resolution_results['sharpness_quality']}")
        
        # Temporal analysis
        print("Running temporal analysis...")
        ta = TemporalAnalyzer()
        temporal_results = ta.analyze(frames)
        print(f"  ✓ Temporal Quality: {temporal_results['overall_temporal_quality']}")
        
        # Blank frame detection
        print("Running blank frame detection...")
        bfd = BlankFrameDetector()
        blank_results = bfd.analyze(frames, metadata['fps'])
        print(f"  ✓ Frame Quality: {blank_results['quality']}")
        
        print("\n✅ Full analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during full analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Video Analysis System Test")
    print("=" * 60)
    
    tests = [
        ("Video Processor", test_video_processor),
        ("Quality Analyzer", test_quality_analyzer),
        ("Resolution Analyzer", test_resolution_analyzer),
        ("Temporal Analyzer", test_temporal_analyzer),
        ("Blank Frame Detector", test_blank_frame_detector),
        ("Full Analysis", test_full_analysis),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
