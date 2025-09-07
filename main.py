import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tempfile
import os
from detect_rice import DetectRice
from detect_wheat import DetectWheat
from GenerateSolution import GenerateSolution

class StreamlitMain:
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.solutions = {}
            st.session_state.frame_count = 0
            st.session_state.camera_running = False
            
        if not st.session_state.initialized:
            self.initialize_models()
    
    def initialize_models(self):
        """Initialize the models with loading indicators"""
        with st.spinner("Loading AI models... This may take a moment."):
            try:
                st.session_state.wheat_detector = DetectWheat()
                st.session_state.rice_detector = DetectRice()
                st.session_state.solution_generator = GenerateSolution()
                st.session_state.solution_generator.process_documents()
                st.session_state.initialized = True
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.stop()
    
    def detect_disease_from_frame(self, frame, crop_type):
        """Detect disease from camera frame"""
        try:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, frame)
            
            # Detect disease based on crop type
            if crop_type == "wheat":
                disease = st.session_state.wheat_detector.detect_wheat_disease(temp_path)
            else:
                disease = st.session_state.rice_detector.detect_rice_disease(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return disease
        except Exception as e:
            st.error(f"Error detecting disease: {str(e)}")
            return None
    
    def get_solution(self, crop_type, disease):
        """Get solution for detected disease"""
        solution_key = f"{crop_type}_{disease}"
        
        if solution_key not in st.session_state.solutions:
            try:
                st.session_state.solution_generator.generate_with_context(crop_type, disease)
                st.session_state.solutions[solution_key] = {
                    'answer': st.session_state.solution_generator.response["answer"],
                    'processing_time': st.session_state.solution_generator.processing_time
                }
            except Exception as e:
                st.error(f"Error generating solution: {str(e)}")
                return None
        
        return st.session_state.solutions[solution_key]
    
    def draw_detection_box(self, frame):
        """Draw detection box on frame"""
        h, w = frame.shape[:2]
        box_size = 200
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Draw red rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        return frame
    
    def run_camera_detection(self, crop_type):
        """Run live camera detection"""
        
        # Create placeholders for dynamic content
        camera_placeholder = st.empty()
        detection_placeholder = st.empty()
        solutions_placeholder = st.empty()
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_button = st.button("🎥 Start Camera", key="start_cam")
        with col2:
            stop_button = st.button("⏹️ Stop Camera", key="stop_cam")
        with col3:
            analyze_button = st.button("🔍 Analyze Current Frame", key="analyze_frame")
        
        if start_button:
            st.session_state.camera_running = True
        
        if stop_button:
            st.session_state.camera_running = False
        
        # Initialize camera
        cap = None
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check your camera connection.")
                return
            
            st.success("📹 Camera is running...")
        
        # Main camera loop
        current_frame = None
        while st.session_state.camera_running and cap is not None:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            st.session_state.frame_count += 1
            current_frame = frame.copy()
            
            # Draw detection box
            display_frame = self.draw_detection_box(frame.copy())
            
            # Convert BGR to RGB for display
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display current frame
            with camera_placeholder.container():
                st.image(display_frame_rgb, channels="RGB", caption=f"Live Feed - Scanning {crop_type.title()}")
            
            # Auto-detect every 30 frames (reduce processing load)
            if st.session_state.frame_count % 30 == 0:
                with detection_placeholder.container():
                    with st.spinner("🔍 Analyzing frame..."):
                        disease = self.detect_disease_from_frame(frame, crop_type)
                        if disease:
                            st.success(f"🦠 Detected: **{disease}**")
                            
                            # Get solution
                            solution_data = self.get_solution(crop_type, disease)
                            if solution_data:
                                with solutions_placeholder.container():
                                    st.markdown("### 💡 Treatment Solution")
                                    st.markdown(f"**Disease:** {disease}")
                                    st.markdown(f"**Crop:** {crop_type.title()}")
                                    st.markdown("**Recommended Treatment:**")
                                    st.markdown(solution_data['answer'])
                                    st.caption(f"Processing time: {solution_data['processing_time']:.2f} seconds")
            
            # Manual analysis button
            if analyze_button and current_frame is not None:
                with detection_placeholder.container():
                    with st.spinner("🔍 Analyzing current frame..."):
                        disease = self.detect_disease_from_frame(current_frame, crop_type)
                        if disease:
                            st.success(f"🦠 Detected: **{disease}**")
                            
                            # Get solution
                            solution_data = self.get_solution(crop_type, disease)
                            if solution_data:
                                with solutions_placeholder.container():
                                    st.markdown("### 💡 Treatment Solution")
                                    st.markdown(f"**Disease:** {disease}")
                                    st.markdown(f"**Crop:** {crop_type.title()}")
                                    st.markdown("**Recommended Treatment:**")
                                    st.markdown(solution_data['answer'])
                                    st.caption(f"Processing time: {solution_data['processing_time']:.2f} seconds")
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.1)
        
        # Cleanup
        if cap is not None:
            cap.release()
    
    def display_previous_solutions(self):
        """Display all previously detected solutions"""
        if st.session_state.solutions:
            st.markdown("### 📋 Previous Detections")
            
            for solution_key, solution_data in st.session_state.solutions.items():
                crop, disease = solution_key.split('_', 1)
                
                with st.expander(f"🦠 {disease} in {crop.title()}", expanded=False):
                    st.markdown("**Recommended Treatment:**")
                    st.markdown(solution_data['answer'])
                    st.caption(f"Processing time: {solution_data['processing_time']:.2f} seconds")
    
    def main(self):
        st.set_page_config(
            page_title="🌾 Minori AI - Crop Disease Detection",
            page_icon="🌾",
            layout="wide"
        )
        
        st.title("🌾 Minori AI - Crop Disease Detection System")
        st.markdown("---")
        
        # Sidebar for crop selection and controls
        with st.sidebar:
            st.header("🔧 Settings")
            
            crop_type = st.selectbox(
                "🌱 Select Crop Type",
                ["wheat", "rice"],
                index=0
            ).lower()
            
            st.markdown("---")
            st.markdown("### 📝 Instructions")
            st.markdown("""
            1. Select your crop type
            2. Click 'Start Camera' to begin live detection
            3. Position the crop in the red detection box
            4. The system will automatically analyze frames
            5. Use 'Analyze Current Frame' for manual detection
            6. Click 'Stop Camera' when done
            """)
            
            st.markdown("---")
            if st.button("🗑️ Clear All Solutions"):
                st.session_state.solutions = {}
                st.success("Solutions cleared!")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 📸 Image Detection - {crop_type.title()}")
            
            # Check if running locally or on cloud
            try:
                # Try to access camera (will fail on Streamlit Cloud)
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.release()
                    # If camera is available, show camera option
                    detection_mode = st.radio(
                        "Choose detection mode:",
                        ["📷 Upload Image", "🎥 Live Camera"],
                        index=0
                    )
                    
                    if detection_mode == "🎥 Live Camera":
                        self.run_camera_detection_local(crop_type)
                    else:
                        self.run_image_upload_detection(crop_type)
                else:
                    # Camera not available, use image upload
                    self.run_image_upload_detection(crop_type)
            except:
                # On Streamlit Cloud or camera not accessible
                self.run_image_upload_detection(crop_type)
    
    def run_camera_detection_local(self, crop_type):
        """Run local camera detection (for local development)"""
        
        # Create placeholders for dynamic content
        camera_placeholder = st.empty()
        detection_placeholder = st.empty()
        solutions_placeholder = st.empty()
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_button = st.button("🎥 Start Camera", key="start_cam")
        with col2:
            stop_button = st.button("⏹️ Stop Camera", key="stop_cam")
        with col3:
            analyze_button = st.button("🔍 Analyze Current Frame", key="analyze_frame")
        
        if start_button:
            st.session_state.camera_running = True
        
        if stop_button:
            st.session_state.camera_running = False
        
        # Initialize camera
        cap = None
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open camera. Please check your camera connection.")
                return
            
            st.success("📹 Camera is running...")
        
        # Main camera loop
        current_frame = None
        while st.session_state.camera_running and cap is not None:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            st.session_state.frame_count += 1
            current_frame = frame.copy()
            
            # Draw detection box
            display_frame = self.draw_detection_box(frame.copy())
            
            # Convert BGR to RGB for display
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display current frame
            with camera_placeholder.container():
                st.image(display_frame_rgb, channels="RGB", caption=f"Live Feed - Scanning {crop_type.title()}")
            
            # Auto-detect every 30 frames (reduce processing load)
            if st.session_state.frame_count % 30 == 0:
                with detection_placeholder.container():
                    with st.spinner("🔍 Analyzing frame..."):
                        disease = self.detect_disease_from_frame(frame, crop_type)
                        if disease:
                            st.success(f"🦠 Detected: **{disease}**")
                            
                            # Get solution
                            solution_data = self.get_solution(crop_type, disease)
                            if solution_data:
                                with solutions_placeholder.container():
                                    st.markdown("### 💡 Treatment Solution")
                                    st.markdown(f"**Disease:** {disease}")
                                    st.markdown(f"**Crop:** {crop_type.title()}")
                                    st.markdown("**Recommended Treatment:**")
                                    st.markdown(solution_data['answer'])
                                    st.caption(f"Processing time: {solution_data['processing_time']:.2f} seconds")
            
            # Manual analysis button
            if analyze_button and current_frame is not None:
                with detection_placeholder.container():
                    with st.spinner("🔍 Analyzing current frame..."):
                        disease = self.detect_disease_from_frame(current_frame, crop_type)
                        if disease:
                            st.success(f"🦠 Detected: **{disease}**")
                            
                            # Get solution
                            solution_data = self.get_solution(crop_type, disease)
                            if solution_data:
                                with solutions_placeholder.container():
                                    st.markdown("### 💡 Treatment Solution")
                                    st.markdown(f"**Disease:** {disease}")
                                    st.markdown(f"**Crop:** {crop_type.title()}")
                                    st.markdown("**Recommended Treatment:**")
                                    st.markdown(solution_data['answer'])
                                    st.caption(f"Processing time: {solution_data['processing_time']:.2f} seconds")
            
            # Small delay to prevent overwhelming the browser
            time.sleep(0.1)
        
        # Cleanup
        if cap is not None:
            cap.release()
        
        with col2:
            self.display_previous_solutions()

if __name__ == "__main__":
    app = StreamlitMain()
    app.main()