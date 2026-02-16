import os
import sys
import time
import json
import tempfile
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Video Analysis",
    page_icon="🤖",
    layout="wide"
)

class LLMVideoAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the Gemini API client"""
        if not api_key:
            raise ValueError("API Key is required. Set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def upload_file(self, path: str):
        """Uploads a file to the Gemini API and waits for it to be active"""
        with st.status("Uploading video to Gemini...", expanded=True) as status:
            try:
                video_file = genai.upload_file(path=path)
                st.write(f"Upload complete: `{video_file.name}`")
                status.update(label="Video uploaded!", state="complete", expanded=False)
                return video_file
            except Exception as e:
                status.update(label="Upload failed!", state="error")
                st.error(f"Error uploading file: {e}")
                raise

    def wait_for_files_active(self, files):
        """Waits for the uploaded files to be processed and active"""
        with st.status("Waiting for video processing...", expanded=True) as status:
            for name in (file.name for file in files):
                file = genai.get_file(name)
                while file.state.name == "PROCESSING":
                    time.sleep(2)
                    file = genai.get_file(name)
                
                if file.state.name != "ACTIVE":
                    status.update(label="Processing failed!", state="error")
                    raise Exception(f"File {file.name} failed to process")
            
            status.update(label="Video processed & ready!", state="complete", expanded=False)

    def analyze_video(self, video_path: str) -> dict:
        """
        Analyzes a video using Gemini and returns a JSON dict 
        matching the structure of the computer vision pipeline.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1. Upload Video
        video_file = self.upload_file(video_path)
        
        # 2. Wait for processing
        self.wait_for_files_active([video_file])

        # 3. Construct Prompt
        prompt = """
        Analyze this video file for technical quality issues. 
        Act as a professional video quality assessment tool.
        
        You must output ONLY a valid JSON object. Do not add markdown formatting or explanations outside the JSON.
        
        The JSON must strictly follow this structure and calculate these specific metrics based on your visual analysis:

        {
            "metadata": {
                "width": <int, estimate resolution width>,
                "height": <int, estimate resolution height>,
                "frame_count": <int, estimate total frames>,
                "duration": <float, duration in seconds>,
                "fps": <float, estimate fps>,
                "codec": "h264", 
                "file_size_mb": <float, estimate or 0.0 if unknown>
            },
            "quality": {
                "quality_score": <float, 0.0 to 1.0, where 1.0 is perfect>,
                "interpretation": <string, one of: "Excellent", "Good", "Fair", "Poor">,
                "raw_score": <float, 0.0 to 1.0>,
                "model_used": "Gemini 2.0 Flash"
            },
            "resolution": {
                "nominal_resolution": <string, e.g. "1920x1080">,
                "nominal_width": <int>,
                "nominal_height": <int>,
                "average_sharpness": <float, calibrated score ~0-2000, >100 is sharp>,
                "sharpness_quality": <string, "Excellent"/"Good"/"Fair"/"Poor">,
                "average_blur": <float, 0.0 to 1.0, higher is blurrier>,
                "blur_quality": <string, "Excellent" if low blur, else "Good"/"Fair"/"Poor">,
                "clarity_score": <float, 0.0 to 1.0>,
                "clarity_quality": <string, "Excellent"/"Good"/"Fair"/"Poor">,
                "is_oversharpened": <boolean, true if halos/artifacts present>,
                "has_excessive_blur": <boolean>,
                "potentially_upscaled": <boolean, true if soft edges despite high resolution>,
                "average_edge_density": <float, 0.0 to 1.0>,
                 "sharpness_scores": [], 
                 "blur_scores": [],
                 "frame_indices": []
            },
            "temporal": {
                "overall_temporal_quality": <string, "Excellent"/"Good"/"Fair"/"Poor">,
                "flickering": {
                    "has_flickering": <boolean>,
                    "severity": <string, "None"/"Low"/"Medium"/"High">,
                    "num_spikes": <int>,
                    "brightness_std": <float>
                },
                "jitter": {
                    "has_jitter": <boolean, true if camera shake/unstable>,
                    "severity": <string, "None"/"Low"/"Medium"/"High">,
                    "motion_std": <float>
                }
            },
            "blank_frames": {
                "quality": <string, "Excellent" if none, else "Good"/"Fair"/"Poor">,
                "has_issues": <boolean>,
                "num_black_frames": <int>,
                "black_percentage": <float>,
                "num_frozen_frames": <int>,
                "frozen_percentage": <float>,
                "num_white_frames": <int>,
                "white_percentage": <float>,
                "issues": <list of strings, e.g. ["Detected 5 black frames"]>
            },
            "layman_summary": {
                "summary": <string, a helpful paragraph summarizing the video quality for a non-technical user>
            }
        }
        
        IMPORTANT GUIDELINES:
        - "average_blur": 0.0 is very sharp, 1.0 is very blurry.
        - "clarity_score": Combined metric, 1.0 is best.
        - "sharpness_quality": >= 0.75 is Excellent.
        - Be critical. If the video has artifacts, compression blocks, or noise, reflect that in the scores.
        """

        # 4. Generate Content
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with st.spinner(f"Analyzing content with Gemini (Attempt {retry_count+1}/{max_retries})..."):
                    response = self.model.generate_content(
                        [video_file, prompt],
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json"
                        )
                    )
                    break # Success!
            except Exception as e:
                # Check for 429 Resource Exhausted
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    retry_count += 1
                    if retry_count >= max_retries:
                        st.error("❌ API Quota Exceeded. Please try again later or upgrade your plan.")
                        raise e
                    
                    # Extract suggested delay or default to 60s
                    wait_time = 60
                    st.warning(f"⚠️ Rate limit hit. Waiting {wait_time}s before retrying...")
                    
                    # Visual countdown
                    progress_bar = st.progress(0)
                    for i in range(wait_time):
                        time.sleep(1)
                        progress_bar.progress((i + 1) / wait_time)
                    progress_bar.empty()
                else:
                    raise e

        # 5. Parse Response
        try:
            if not response.parts:
                return {"error": "LLM response was blocked (safety filters)."}
            
            result_json = json.loads(response.text)
            return result_json
        except ValueError:
            st.error("Failed to get text from LLM response (Content Blocked?)")
            return {"error": "Invalid response (Safety Blocked)"}
        except json.JSONDecodeError:
            st.error("Failed to decode JSON from LLM response.")
            st.code(response.text)
            return {"error": "Invalid JSON response from LLM"}

def main():
    st.title("🤖 LLM Video Quality Analysis")
    st.markdown("Analyze video quality using **Gemini 2.0 Flash**'s multimodal capabilities.")

    # API Key Handling
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if not api_key:
            st.warning("⚠️ Please provide a Gemini API Key to continue.")
            st.stop()
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Analyze Video", type="primary"):
            try:
                analyzer = LLMVideoAnalyzer(api_key)
                results = analyzer.analyze_video(video_path)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Analysis Complete!")
                    
                    # Display Layman Summary
                    if "layman_summary" in results:
                        st.subheader("📝 Summary")
                        st.info(results["layman_summary"].get("summary", "No summary available."))
                    
                    # Key Metrics Columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        q_score = results.get("quality", {}).get("quality_score", 0)
                        st.metric("Overall Quality", f"{q_score*100:.1f}%", results.get("quality", {}).get("interpretation", ""))
                    
                    with col2:
                        tech_res = results.get("resolution", {})
                        st.metric("Sharpness", tech_res.get("sharpness_quality", "N/A"))
                    
                    with col3:
                        temp_res = results.get("temporal", {})
                        st.metric("Temporal Stability", temp_res.get("overall_temporal_quality", "N/A"))

                    # Full JSON Output
                    with st.expander("View Full Analysis JSON", expanded=True):
                        st.json(results)
                    
                    # Download Button
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_str,
                        file_name="llm_video_analysis.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")
            finally:
                # Cleanup temp file
                if os.path.exists(video_path):
                    os.unlink(video_path)

if __name__ == "__main__":
    main()
