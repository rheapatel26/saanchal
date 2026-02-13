"""
Video Quality Analysis Streamlit Application
Comprehensive video analysis using FAST-VQA and custom analyzers
"""
import streamlit as st
import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils import VideoProcessor, Visualizer, save_uploaded_video, get_video_library, save_analysis_results, load_analysis_results, delete_video
from analyzers import QualityAnalyzer, ResolutionAnalyzer, TemporalAnalyzer, BlankFrameDetector, SummaryGenerator


# Page configuration
st.set_page_config(
    page_title=config.UI_CONFIG["page_title"],
    page_icon=config.UI_CONFIG["page_icon"],
    layout=config.UI_CONFIG["layout"],
)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'selected_video' not in st.session_state:
        st.session_state.selected_video = None
    if 'video_source' not in st.session_state:
        st.session_state.video_source = None  # 'upload' or 'library'
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None


def run_analysis(video_path: str):
    """
    Run complete video analysis
    
    Args:
        video_path: Path to video file
    """
    try:
        # Initialize video processor
        with st.spinner("Loading video..."):
            video_processor = VideoProcessor(video_path)
            metadata = video_processor.get_metadata()
        
        st.success(f"✓ Video loaded: {metadata['width']}x{metadata['height']}, "
                   f"{metadata['frame_count']} frames, {metadata['duration']:.2f}s")
        
        # Load frames for analysis
        with st.spinner("Extracting frames..."):
            max_frames = config.VIDEO_CONFIG["max_frames_for_analysis"]
            frames = video_processor.get_all_frames(max_frames=max_frames)
            st.info(f"Analyzing {len(frames)} frames")
        
        results = {"metadata": metadata}
        
        # Quality Analysis
        with st.spinner("Running FAST-VQA quality analysis..."):
            quality_analyzer = QualityAnalyzer()
            quality_results = quality_analyzer.analyze(video_processor.video_reader)
            results["quality"] = quality_results
            st.success(f"✓ Quality Score: {quality_results['quality_score']:.4f} ({quality_results['interpretation']})")
        
        # Resolution Analysis
        with st.spinner("Analyzing resolution and sharpness..."):
            resolution_analyzer = ResolutionAnalyzer()
            resolution_results = resolution_analyzer.analyze(frames, metadata)
            results["resolution"] = resolution_results
            st.success(f"✓ Sharpness: {resolution_results['sharpness_quality']}")
        
        # Temporal Analysis
        with st.spinner("Detecting flickering and jitter..."):
            temporal_analyzer = TemporalAnalyzer()
            temporal_results = temporal_analyzer.analyze(frames)
            results["temporal"] = temporal_results
            st.success(f"✓ Temporal Quality: {temporal_results['overall_temporal_quality']}")
        
        # Blank Frame Detection
        with st.spinner("Detecting blank and frozen frames..."):
            blank_detector = BlankFrameDetector()
            blank_results = blank_detector.analyze(frames, metadata['fps'])
            results["blank_frames"] = blank_results
            st.success(f"✓ Frame Quality: {blank_results['quality']}")
        
        # Generate Layman's Summary
        with st.spinner("Generating summary with Groq Scout..."):
            try:
                summary_generator = SummaryGenerator()
                summary_results = summary_generator.generate_summary(results)
                results["layman_summary"] = summary_results
                st.success("✓ Summary generated")
            except Exception as e:
                st.warning(f"⚠️ Could not generate summary: {str(e)}")
                results["layman_summary"] = {
                    "summary": "Summary generation unavailable. Please ensure GROQ_API_KEY is set.",
                    "error": str(e)
                }
        
        return results
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def display_quality_tab(results):
    """Display quality analysis results"""
    st.header("🎯 Video Quality Assessment")
    
    quality = results["quality"]
    
    # Display gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = Visualizer.create_quality_gauge(quality["quality_score"])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Quality Score", f"{quality['quality_score']:.4f}")
        st.metric("Interpretation", quality["interpretation"])
        st.metric("Model Used", quality["model_used"])
        
        # Description
        st.info(quality["description"])
    
    # Additional metrics
    st.subheader("Detailed Metrics")
    metrics_df = Visualizer.create_metrics_table({
        "Quality Score (0-1)": quality["quality_score"],
        "Raw Model Score": quality["raw_score"],
        "Quality Level": quality["interpretation"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def display_resolution_tab(results):
    """Display resolution analysis results"""
    st.header("📐 Resolution, Sharpness & Blur Analysis")
    
    resolution = results["resolution"]
    metadata = results["metadata"]
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Resolution", resolution["nominal_resolution"])
    with col2:
        st.metric("Sharpness Quality", resolution["sharpness_quality"])
    with col3:
        st.metric("Blur Quality", resolution["blur_quality"])
    with col4:
        st.metric("Avg Sharpness", f"{resolution['average_sharpness']:.2f}")
    with col5:
        st.metric("Avg Blur", f"{resolution['average_blur']:.3f}")
    
    # Warnings
    if resolution["potentially_upscaled"]:
        st.warning("⚠️ Video may be upscaled - low edge density for resolution")
    
    if resolution["has_excessive_blur"]:
        st.warning("⚠️ Excessive blur detected - video may be out of focus or motion blurred")
    
    # Sharpness over time
    st.subheader("Sharpness Analysis Over Time")
    fig = Visualizer.create_temporal_plot(
        resolution["frame_indices"],
        resolution["sharpness_scores"],
        "Frame Sharpness",
        "Sharpness Score",
        threshold=config.THRESHOLDS["resolution"]["sharpness_threshold"]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Blur over time
    st.subheader("Blur Analysis Over Time")
    fig_blur = Visualizer.create_temporal_plot(
        resolution["frame_indices"],
        resolution["blur_scores"],
        "Frame Blur Score",
        "Blur Score (0-1)",
        threshold=config.THRESHOLDS["resolution"]["blur_threshold"]
    )
    st.plotly_chart(fig_blur, use_container_width=True)
    
    # Detailed metrics
    st.subheader("Detailed Metrics")
    metrics_df = Visualizer.create_metrics_table({
        "Nominal Resolution": resolution["nominal_resolution"],
        "Total Pixels": resolution["total_pixels"],
        "Average Sharpness": resolution["average_sharpness"],
        "Min Sharpness": resolution["min_sharpness"],
        "Max Sharpness": resolution["max_sharpness"],
        "Sharpness Std Dev": resolution["sharpness_std"],
        "Average Blur": resolution["average_blur"],
        "Min Blur": resolution["min_blur"],
        "Max Blur": resolution["max_blur"],
        "Blur Std Dev": resolution["blur_std"],
        "Average Edge Density": resolution["average_edge_density"],
        "Potentially Upscaled": resolution["potentially_upscaled"],
        "Has Excessive Blur": resolution["has_excessive_blur"],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def display_temporal_tab(results):
    """Display temporal analysis results"""
    st.header("⏱️ Temporal Distortion Analysis")
    
    temporal = results["temporal"]
    
    # Overall status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Temporal Quality", temporal["overall_temporal_quality"])
    with col2:
        flicker_status = "✓ No Issues" if not temporal["flickering"]["has_flickering"] else f"⚠️ {temporal['flickering']['severity']}"
        st.metric("Flickering", flicker_status)
    with col3:
        jitter_status = "✓ No Issues" if not temporal["jitter"]["has_jitter"] else f"⚠️ {temporal['jitter']['severity']}"
        st.metric("Jitter", jitter_status)
    
    # Flickering Analysis
    st.subheader("🔆 Flickering Detection")
    
    flicker = temporal["flickering"]
    
    if flicker["has_flickering"]:
        st.warning(f"Flickering detected - Severity: {flicker['severity']}")
        st.write(f"- Number of brightness spikes: {flicker['num_spikes']}")
        st.write(f"- Brightness variation (std): {flicker['brightness_std']:.2f}")
        st.write(f"- Dominant frequency: {flicker['dominant_frequency']:.4f}")
    else:
        st.success("✓ No flickering detected")
    
    # Brightness over time
    frame_indices = list(range(len(flicker["brightness_values"])))
    fig = Visualizer.create_temporal_plot(
        frame_indices,
        flicker["brightness_values"],
        "Brightness Over Time",
        "Average Brightness"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Spike timeline
    if flicker["num_spikes"] > 0:
        total_frames = len(flicker["brightness_values"])
        fig = Visualizer.create_issue_timeline(
            total_frames,
            flicker["spike_frames"],
            "Brightness Spikes"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Jitter Analysis
    st.subheader("📹 Jitter Detection")
    
    jitter = temporal["jitter"]
    
    if jitter["has_jitter"]:
        st.warning(f"Jitter detected - Severity: {jitter['severity']}")
        st.write(f"- Motion variation (std): {jitter['motion_std']:.2f}")
        st.write(f"- Average SSIM: {jitter['ssim_mean']:.4f}")
        st.write(f"- Sudden motion changes: {jitter['num_sudden_changes']}")
    else:
        st.success("✓ No jitter detected")
    
    # Motion magnitude over time
    fig = Visualizer.create_temporal_plot(
        jitter["sampled_indices"],
        jitter["motion_magnitudes"],
        "Motion Magnitude Over Time",
        "Average Motion"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # SSIM over time
    fig = Visualizer.create_temporal_plot(
        jitter["sampled_indices"],
        jitter["ssim_scores"],
        "Frame Similarity (SSIM) Over Time",
        "SSIM Score",
        threshold=config.THRESHOLDS["jitter"]["ssim_drop_threshold"]
    )
    st.plotly_chart(fig, use_container_width=True)


def display_blank_frames_tab(results):
    """Display blank frame detection results"""
    st.header("🎬 Blank & Frozen Frame Detection")
    
    blank = results["blank_frames"]
    
    # Overall status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Frame Quality", blank["quality"])
    with col2:
        st.metric("Black Frames", f"{blank['num_black_frames']} ({blank['black_percentage']:.2f}%)")
    with col3:
        st.metric("White Frames", f"{blank['num_white_frames']} ({blank['white_percentage']:.2f}%)")
    with col4:
        st.metric("Frozen Frames", f"{blank['num_frozen_frames']} ({blank['frozen_percentage']:.2f}%)")
    
    # Issues summary
    if blank["has_issues"]:
        st.warning("⚠️ Issues detected:")
        for issue in blank["issues"]:
            st.write(f"- {issue}")
    else:
        st.success("✓ No blank or frozen frames detected")
    
    # Black frames
    if blank["num_black_frames"] > 0:
        st.subheader("⬛ Black Frames")
        st.write(f"Detected {blank['num_black_frames']} black frames")
        
        # Show timestamps
        with st.expander("View Black Frame Timestamps"):
            timestamps_str = ", ".join([f"{t:.2f}s" for t in blank["black_timestamps"][:20]])
            if len(blank["black_timestamps"]) > 20:
                timestamps_str += f" ... and {len(blank['black_timestamps']) - 20} more"
            st.write(timestamps_str)
        
        # Timeline
        fig = Visualizer.create_issue_timeline(
            blank["total_frames"],
            blank["black_frames"],
            "Black Frames"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # White frames
    if blank["num_white_frames"] > 0:
        st.subheader("⬜ White Frames")
        st.write(f"Detected {blank['num_white_frames']} white frames")
        
        # Show timestamps
        with st.expander("View White Frame Timestamps"):
            timestamps_str = ", ".join([f"{t:.2f}s" for t in blank["white_timestamps"][:20]])
            if len(blank["white_timestamps"]) > 20:
                timestamps_str += f" ... and {len(blank['white_timestamps']) - 20} more"
            st.write(timestamps_str)
        
        # Timeline
        fig = Visualizer.create_issue_timeline(
            blank["total_frames"],
            blank["white_frames"],
            "White Frames"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Frozen frames
    if blank["num_frozen_frames"] > 0:
        st.subheader("❄️ Frozen Frames")
        st.write(f"Detected {blank['num_frozen_frames']} frozen frames")
        
        # Show timestamps
        with st.expander("View Frozen Frame Timestamps"):
            timestamps_str = ", ".join([f"{t:.2f}s" for t in blank["frozen_timestamps"][:20]])
            if len(blank["frozen_timestamps"]) > 20:
                timestamps_str += f" ... and {len(blank['frozen_timestamps']) - 20} more"
            st.write(timestamps_str)
        
        # Timeline
        fig = Visualizer.create_issue_timeline(
            blank["total_frames"],
            blank["frozen_frames"],
            "Frozen Frames"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_summary_tab(results):
    """Display comprehensive summary"""
    st.header("📊 Analysis Summary")
    
    # Layman's Summary Section
    if "layman_summary" in results and "summary" in results["layman_summary"]:
        st.subheader("🎯 Overall Assessment")
        
        summary_data = results["layman_summary"]
        
        # Display the summary in a nice container
        if "error" not in summary_data:
            st.markdown(summary_data["summary"])
            
            # Show model info in an expander
            with st.expander("ℹ️ Summary Details"):
                st.caption(f"Generated by: {summary_data.get('model_used', 'N/A')}")
                if summary_data.get('tokens_used'):
                    st.caption(f"Tokens used: {summary_data['tokens_used']}")
        else:
            st.info(summary_data["summary"])
        
        st.divider()
    
    # Overall scores
    st.subheader("Overall Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Quality", f"{results['quality']['quality_score']:.3f}")
        st.caption(results['quality']['interpretation'])
    
    with col2:
        st.metric("Sharpness", results['resolution']['sharpness_quality'])
        st.caption(f"Avg: {results['resolution']['average_sharpness']:.1f}")
    
    with col3:
        st.metric("Temporal", results['temporal']['overall_temporal_quality'])
        flicker_sev = results['temporal']['flickering']['severity']
        jitter_sev = results['temporal']['jitter']['severity']
        st.caption(f"F:{flicker_sev}, J:{jitter_sev}")
    
    with col4:
        st.metric("Frames", results['blank_frames']['quality'])
        st.caption(f"{results['blank_frames']['num_black_frames']}B, {results['blank_frames']['num_frozen_frames']}F")
    
    # Video metadata
    st.subheader("Video Information")
    metadata = results["metadata"]
    info_df = Visualizer.create_metrics_table({
        "Resolution": f"{metadata['width']}x{metadata['height']}",
        "Frame Count": metadata['frame_count'],
        "Duration (seconds)": f"{metadata['duration']:.2f}",
        "FPS": f"{metadata['fps']:.2f}",
        "Codec": metadata['codec'],
        "File Size (MB)": f"{metadata['file_size_mb']:.2f}",
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # Export results
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_str,
            file_name="video_analysis_results.json",
            mime="application/json"
        )
    
    with col2:
        # Summary text export
        summary_text = f"""Video Quality Analysis Report
================================

Quality Score: {results['quality']['quality_score']:.4f} ({results['quality']['interpretation']})
Resolution: {metadata['width']}x{metadata['height']}
Sharpness: {results['resolution']['sharpness_quality']} (Avg: {results['resolution']['average_sharpness']:.2f})
Temporal Quality: {results['temporal']['overall_temporal_quality']}
Frame Quality: {results['blank_frames']['quality']}

Flickering: {results['temporal']['flickering']['severity']}
Jitter: {results['temporal']['jitter']['severity']}
Black Frames: {results['blank_frames']['num_black_frames']}
White Frames: {results['blank_frames']['num_white_frames']}
Frozen Frames: {results['blank_frames']['num_frozen_frames']}
"""
        st.download_button(
            label="📥 Download Summary (TXT)",
            data=summary_text,
            file_name="video_analysis_summary.txt",
            mime="text/plain"
        )


def main():
    """Main application"""
    initialize_session_state()
    
    # Title
    st.title("🎥 Video Quality Analysis")
    st.markdown("Comprehensive video analysis using **FAST-VQA** and advanced temporal distortion detection")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        st.write(f"**Model:** {config.MODEL_CONFIG['model_type']}")
        st.write(f"**Device:** {config.MODEL_CONFIG['device']}")
        
        st.divider()
        
        # Video Library
        st.header("📚 Video Library")
        
        video_library = get_video_library(config.VIDEOS_DIR)
        
        if video_library:
            st.write(f"**{len(video_library)} video(s) saved**")
            
            for video in video_library:
                with st.expander(f"📹 {video['filename'][:30]}..."):
                    st.caption(f"📅 {video['upload_date']}")
                    st.caption(f"💾 {video['size_mb']:.2f} MB")
                    
                    if video['has_results']:
                        st.caption("✅ Analyzed")
                    else:
                        st.caption("⏳ Not analyzed")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Select", key=f"select_{video['filename']}"):
                            st.session_state.selected_video = video
                            st.session_state.video_path = video['path']
                            st.session_state.video_source = 'library'
                            
                            # Load existing results if available
                            if video['has_results']:
                                st.session_state.results = load_analysis_results(video['path'])
                                st.session_state.analysis_complete = True
                            else:
                                st.session_state.results = None
                                st.session_state.analysis_complete = False
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("Delete", key=f"delete_{video['filename']}"):
                            delete_video(video['path'])
                            if st.session_state.video_path == video['path']:
                                st.session_state.video_path = None
                                st.session_state.selected_video = None
                                st.session_state.results = None
                                st.session_state.analysis_complete = False
                            st.rerun()
        else:
            st.info("No videos saved yet")
        
        st.divider()
        
        st.header("About")
        st.markdown("""
        This app analyzes:
        - **Quality**: FAST-VQA model score
        - **Resolution**: Sharpness & blur
        - **Temporal**: Flickering & jitter
        - **Frames**: Blank/black/frozen detection
        """)
    
    # File upload
    st.header("Upload Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=config.UI_CONFIG["supported_formats"],
        help=f"Supported formats: {', '.join(config.UI_CONFIG['supported_formats'])}"
    )
    
    if uploaded_file is not None:
        # Check if this is a new upload (different file)
        is_new_upload = (
            st.session_state.video_source != 'upload' or 
            st.session_state.uploaded_filename != uploaded_file.name
        )
        
        # Save uploaded file permanently if it's a new upload
        if is_new_upload:
            with st.spinner("Saving video..."):
                video_info = save_uploaded_video(uploaded_file, config.VIDEOS_DIR)
                st.session_state.video_path = video_info['path']
                st.session_state.video_source = 'upload'
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.selected_video = None
                st.session_state.results = None
                st.session_state.analysis_complete = False
                st.success(f"✅ Video saved: {video_info['filename']}")
        
        video_path = st.session_state.video_path
        
        # Display video
        st.video(video_path)
        
        # Check if analysis results exist
        existing_results = load_analysis_results(video_path)
        
        if existing_results and not st.session_state.analysis_complete:
            st.info("💡 This video has been analyzed before. Loading cached results...")
            st.session_state.results = existing_results
            st.session_state.analysis_complete = True
        
        # Run analysis button
        button_label = "🔄 Re-analyze" if existing_results else "🚀 Run Analysis"
        if st.button(button_label, type="primary"):
            st.session_state.analysis_complete = False
            st.session_state.results = None
            
            results = run_analysis(video_path)
            
            if results is not None:
                # Save results
                save_analysis_results(video_path, results)
                
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.success("✅ Analysis complete and saved!")
                st.rerun()
        
        # Display results if available
        if st.session_state.analysis_complete and st.session_state.results is not None:
            st.divider()
            
            # Create tabs
            tabs = st.tabs([
                "📊 Summary",
                "🎯 Quality",
                "📐 Resolution",
                "⏱️ Temporal",
                "🎬 Blank Frames"
            ])
            
            with tabs[0]:
                display_summary_tab(st.session_state.results)
            
            with tabs[1]:
                display_quality_tab(st.session_state.results)
            
            with tabs[2]:
                display_resolution_tab(st.session_state.results)
            
            with tabs[3]:
                display_temporal_tab(st.session_state.results)
            
            with tabs[4]:
                display_blank_frames_tab(st.session_state.results)
    
    elif st.session_state.selected_video is not None:
        # Display selected video from library
        video_path = st.session_state.video_path
        
        st.success(f"📹 Selected: {st.session_state.selected_video['filename']}")
        
        # Display video
        st.video(video_path)
        
        # Run analysis button if not analyzed
        if not st.session_state.analysis_complete:
            if st.button("🚀 Run Analysis", type="primary"):
                st.session_state.analysis_complete = False
                st.session_state.results = None
                
                results = run_analysis(video_path)
                
                if results is not None:
                    # Save results
                    save_analysis_results(video_path, results)
                    
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                    st.success("✅ Analysis complete and saved!")
                    st.rerun()
        else:
            # Show re-analyze option
            if st.button("🔄 Re-analyze", type="secondary"):
                st.session_state.analysis_complete = False
                st.session_state.results = None
                
                results = run_analysis(video_path)
                
                if results is not None:
                    # Save results
                    save_analysis_results(video_path, results)
                    
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                    st.success("✅ Analysis complete and saved!")
                    st.rerun()
        
        # Display results if available
        if st.session_state.analysis_complete and st.session_state.results is not None:
            st.divider()
            
            # Create tabs
            tabs = st.tabs([
                "📊 Summary",
                "🎯 Quality",
                "📐 Resolution",
                "⏱️ Temporal",
                "🎬 Blank Frames"
            ])
            
            with tabs[0]:
                display_summary_tab(st.session_state.results)
            
            with tabs[1]:
                display_quality_tab(st.session_state.results)
            
            with tabs[2]:
                display_resolution_tab(st.session_state.results)
            
            with tabs[3]:
                display_temporal_tab(st.session_state.results)
            
            with tabs[4]:
                display_blank_frames_tab(st.session_state.results)
    
    else:
        st.info("👆 Please upload a video file or select from library to begin analysis")


if __name__ == "__main__":
    main()
