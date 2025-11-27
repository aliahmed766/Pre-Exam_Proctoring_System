import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import time
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import queue
import threading
import base64
import hashlib

# Configure the page
st.set_page_config(
    page_title="Object Detection with AI Voice Feedback",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ObjectDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.detection_queue = queue.Queue()
        self.last_audio_time = 0
        self.audio_cooldown = 5  # seconds between audio messages
        self.gemini_api_key = None
        self.detected_objects = set()
        self.confidence_threshold = 0.5

    def load_model(self, model_path):
        """best.pt"""
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False

    def set_gemini_api_key(self, api_key):
        """AIzaSyDKM0EyDAx0CwFi7pGlcNXw0nCS4fHrxWo"""
        self.gemini_api_key = api_key
        if api_key:
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                st.error(f"Error configuring Gemini API: {e}")

    def set_confidence_threshold(self, confidence):
        """Set confidence threshold for detection"""
        self.confidence_threshold = confidence

    def generate_voice_message(self, detected_objects):
        """Generate contextual message using Gemini API and convert to speech"""
        if not self.gemini_api_key or not detected_objects:
            return None

        try:
            # Create prompt for Gemini
            object_list = ", ".join(detected_objects)
            prompt = f"""
            You are an assistant that helps maintain a clean environment. 
            The following objects have been detected: {object_list}.

            Generate a brief, polite spoken message (max 15 words) asking to remove these objects.
            Be natural and conversational. For example:
            "I see a mobile phone and books. Please remove them from the area."
            "Detected a calculator and notebook. These should be put away."

            Response:
            """

            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)

            if response.text:
                return response.text.strip()
            else:
                return f"Please remove the detected objects: {object_list}"

        except Exception as e:
            st.error(f"Error generating Gemini message: {e}")
            return f"Detected {object_list}. Please remove them."

    def text_to_speech(self, text):
        """Convert text to speech and return audio file path"""
        if not text:
            return None

        try:
            # Create gTTS audio
            tts = gTTS(text=text, lang='en', slow=False)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None

    def play_audio(self, audio_file):
        """Play audio file in Streamlit"""
        if audio_file and os.path.exists(audio_file):
            try:
                # Read audio file and encode to base64
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()

                # Create audio player
                audio_base64 = base64.b64encode(audio_bytes).decode()
                audio_html = f'''
                    <audio autoplay controls style="width: 100%">
                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    </audio>
                '''
                st.components.v1.html(audio_html, height=100)

                # Clean up temporary file
                os.unlink(audio_file)

            except Exception as e:
                st.error(f"Error playing audio: {e}")

    def recv(self, frame):
        """Process each video frame"""
        img = frame.to_ndarray(format="bgr24")

        if self.model:
            # Run YOLO detection with current confidence threshold
            results = self.model(img, conf=self.confidence_threshold)

            # Process detections
            detected_now = set()
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.model.names[class_id]

                        # Add to current detection set
                        detected_now.add(class_name)

                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Add label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(img, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if we need to generate new audio
            current_time = time.time()
            if (detected_now and
                    current_time - self.last_audio_time > self.audio_cooldown and
                    detected_now != self.detected_objects):

                # Generate voice message
                message = self.generate_voice_message(detected_now)
                if message:
                    # Convert to speech
                    audio_file = self.text_to_speech(message)
                    if audio_file:
                        # Put in queue for main thread to handle
                        self.detection_queue.put({
                            'message': message,
                            'audio_file': audio_file,
                            'objects': detected_now
                        })

                self.last_audio_time = current_time
                self.detected_objects = detected_now.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .api-key-set {
        color: #28a745;
        font-weight: bold;
    }
    .api-key-not-set {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🎯 Smart Object Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time detection with AI voice feedback")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = ObjectDetectionProcessor()
        # Load YOLO model
        model_loaded = st.session_state.processor.load_model("best.pt")
        if not model_loaded:
            st.error("Failed to load YOLO model. Please ensure 'best.pt' is in the correct directory.")
            return

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key Input with secure handling
        st.subheader("🔑 API Configuration")

        # Option 1: File upload for API key (more secure)
        st.write("**Upload API Key File:**")
        api_file = st.file_uploader(
            "Upload a text file containing your Gemini API key",
            type=['txt'],
            help="Create a text file with your API key as the only content",
            label_visibility="collapsed"
        )

        # Option 2: Environment variable style input (hidden)
        st.write("**Or enter API key:**")
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your API key securely",
            help="Your API key is not displayed and is stored only for this session",
            label_visibility="collapsed"
        )

        # Determine which API key to use
        gemini_api_key = None
        if api_file is not None:
            try:
                gemini_api_key = api_file.getvalue().decode('utf-8').strip()
                st.success("✅ API key loaded from file")
            except Exception as e:
                st.error("Error reading API key file")
        elif api_key_input:
            gemini_api_key = api_key_input.strip()
            st.success("✅ API key set securely")

        # Set API key in processor
        if gemini_api_key:
            st.session_state.processor.set_gemini_api_key(gemini_api_key)
            st.markdown('<p class="api-key-set">🔐 API Key: Configured</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="api-key-not-set">🔐 API Key: Not Set</p>', unsafe_allow_html=True)
            st.info("API key is required for voice feedback")

        st.markdown("---")

        # Detection Settings
        st.subheader("🎯 Detection Settings")

        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Higher values = more confident detections, but might miss some objects"
        )

        # Audio cooldown setting
        audio_cooldown = st.slider(
            "Audio Cooldown (seconds)",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Minimum time between voice messages"
        )

        # Apply settings
        st.session_state.processor.set_confidence_threshold(confidence_threshold)
        st.session_state.processor.audio_cooldown = audio_cooldown

        st.markdown("---")
        st.header("📋 Detected Objects")
        st.write("The system detects these 7 object types:")
        objects_list = ["📱 Mobile", "🧮 Calculator", "⌚ Watch", "🎒 Bag",
                        "📚 Books", "📓 Notebooks", "📄 Paper"]
        for obj in objects_list:
            st.write(f"- {obj}")

        st.markdown("---")
        st.header("ℹ️ Instructions")
        st.write("""
        1. Set your Gemini API key (file upload or secure input)
        2. Adjust detection sensitivity
        3. Click 'Start Detection' 
        4. Allow camera access
        5. System provides automatic voice feedback
        """)

        # Current settings display
        st.markdown("---")
        st.header("📊 Current Settings")
        st.write(f"**Confidence:** {confidence_threshold}")
        st.write(f"**Audio Delay:** {audio_cooldown}s")
        st.write(f"**API Status:** {'✅ Set' if gemini_api_key else '❌ Not Set'}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎥 Live Camera Feed")

        # Status information
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            if gemini_api_key:
                st.success("✅ API Key: Configured")
            else:
                st.warning("⚠️ API Key: Not Set")

        with status_col2:
            st.info(f"🎯 Confidence: {confidence_threshold}")

        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: st.session_state.processor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Status information
        if webrtc_ctx.state.playing:
            st.success("✅ Camera is active - Detection running")

            # Display current detected objects
            if st.session_state.processor.detected_objects:
                st.subheader("🔍 Currently Detected Objects")
                for obj in st.session_state.processor.detected_objects:
                    st.write(f"- {obj}")
            else:
                st.info("👀 No objects currently detected")

    with col2:
        st.subheader("🔊 Voice Feedback")

        # Audio feedback section
        if webrtc_ctx.state.playing:
            # Check for new detection messages
            try:
                while True:
                    detection_data = st.session_state.processor.detection_queue.get_nowait()

                    # Display the generated message
                    st.info(f"**AI Message:** {detection_data['message']}")

                    # Play audio
                    st.session_state.processor.play_audio(detection_data['audio_file'])

                    # Show detected objects
                    st.write("**Objects detected:**")
                    for obj in detection_data['objects']:
                        st.write(f"- {obj}")

            except queue.Empty:
                pass

            st.markdown("---")
            st.subheader("📊 Detection Info")
            st.write(f"**Confidence Threshold:** {confidence_threshold}")
            st.write(f"**Audio Cooldown:** {audio_cooldown} seconds")
            st.write("**Model:** YOLOv8 Custom")

        else:
            st.warning("⏸️ Start the camera to begin detection")

            # Placeholder for demonstration
            st.markdown("""
            <div class="info-box">
            <h4>Voice Feedback Examples:</h4>
            <p>• "I see a mobile phone. Please remove it from the area."</p>
            <p>• "Detected books and notebooks. These should be put away."</p>
            <p>• "There's a calculator on the desk. Please store it properly."</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Security Note:** Your API key is handled securely and never displayed. "
        "Camera data is processed locally and not stored. API key is used only for Gemini text generation."
    )


if __name__ == "__main__":
    main()