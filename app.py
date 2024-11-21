import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image, ImageSequence
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip
import onnxruntime as ort
from typing import Tuple, List, Optional
import time
from io import BytesIO
import gc

# Constants
AI_LIST_SEPARATOR = ["----"]
SRVGGNetCompact_models_list = ['RealESR_Gx4', 'RealSRx4_Anime']
RRDB_models_list = ['RealESRGANx4', 'RealESRNetx4']
AI_models_list = SRVGGNetCompact_models_list + AI_LIST_SEPARATOR + RRDB_models_list

SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
SUPPORTED_VIDEO_TYPES = [".mp4", ".webm", ".mkv", ".avi", ".mov", ".flv"]
SUPPORTED_GIF_TYPE = [".gif"]

AI_MODELS = {
    'RealESRGANx4': {
        'name': 'RealESRGAN 4x Pro',
        'description': 'Professional-grade upscaling with superior detail reconstruction and noise handling.',
        'features': [
            'Best for: High-quality professional work',
            'Speed: Balanced processing time',
            'Upscale: 4x resolution increase',
            'Special: Advanced texture preservation'
        ],
        'category': 'High Quality'
    },
    'RealESR_Gx4': {
        'name': 'RealESR-General 4x',
        'description': 'A versatile model optimized for general-purpose upscaling. Excellent for photographs, digital art, and mixed content.',
        'features': [
            'Best for: Natural images, photographs, digital artwork',
            'Speed: Fast processing with good quality',
            'Upscale: 4x resolution increase',
            'Special: Enhanced detail preservation'
        ],
        'category': 'Fast & Efficient'
    },
    'RealSRx4_Anime': {
        'name': 'RealSR-Anime 4x',
        'description': 'Specialized for anime and cartoon content with focus on clean lines and color preservation.',
        'features': [
            'Best for: Anime, cartoons, illustrations',
            'Speed: Very fast processing',
            'Upscale: 4x resolution increase',
            'Special: Line art enhancement'
        ],
        'category': 'Fast & Efficient'
    },
}

class AIUpscaler:
    def __init__(self, model_name: str, device: str = "GPU"):
        self.model_name = model_name
        self.device = device
        self.model_path = os.path.join("AI-onnx", f"{self.model_name}_fp16.onnx")
        self.session = self._load_model()
        self.upscale_factor = self._get_upscale_factor()

    def _load_model(self):
        providers = [('DmlExecutionProvider', {'device_id': '0'})] if self.device == "GPU" else ['CPUExecutionProvider']
        return ort.InferenceSession(self.model_path, providers=providers)

    def _get_upscale_factor(self) -> int:
        if "x4" in self.model_name:
            return 4
        elif "x2" in self.model_name:
            return 2
        return 1

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess_image(self, output: np.ndarray) -> np.ndarray:
        output = np.squeeze(output)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output

    def upscale(self, img: np.ndarray) -> np.ndarray:
        input_data = self.preprocess_image(img)
        output = self.session.run(None, {self.session.get_inputs()[0].name: input_data})[0]
        return self.postprocess_image(output)
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'session'):
            del self.session
        gc.collect()


def initialize_session_state():
    """Initialize session state variables"""
    if 'upscaler' not in st.session_state:
        st.session_state.upscaler = None
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ""
    if 'processed_gif' not in st.session_state:
        st.session_state.processed_gif = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

def create_sidebar() -> dict:
    """Create and return sidebar settings"""
    with st.sidebar:
        st.title("Image Enhancer Settings")

        # Device Selection
        device = st.radio(
            "Processing Device",
            ["CPU"],
            help="Select processing device (GPU recommended)"
        )

        # Resize Factor
        resize_factor = st.slider(
            "Input Resolution %",
            min_value=1,
            max_value=100,
            value=100,
            help="Adjust input resolution before upscaling"
        )

        # Quality Settings
        quality = st.select_slider(
            "Output Quality",
            options=["Low", "Medium", "High"],
            value="High",
            help="Higher quality may increase processing time"
        )
        
        # AI Model Selection
        selected_model = st.selectbox(
            "Select AI Model",
            options=list(AI_MODELS.keys()),
            format_func=lambda x: AI_MODELS[x]['name']
        )
        
        # Show model details
        with st.expander("Model Details", expanded=True):
            st.markdown(f"### {AI_MODELS[selected_model]['name']}")
            st.markdown(AI_MODELS[selected_model]['description'])
            st.markdown("#### Features:")
            for feature in AI_MODELS[selected_model]['features']:
                st.markdown(f"- {feature}")

        return {
            "model": selected_model,
            "device": device,
            "resize_factor": resize_factor,
            "quality": quality
        }

def process_gif(gif_path: str, settings: dict) -> BytesIO:
    """Process a GIF file frame by frame and maintain animation"""
    try:
        # Initialize upscaler if not already done
        if st.session_state.upscaler is None:
            st.session_state.upscaler = AIUpscaler(settings["model"], settings["device"])
        
        # Open GIF
        with Image.open(gif_path) as gif:
            # Get GIF metadata
            duration = gif.info.get('duration', 100)
            loop = gif.info.get('loop', 0)
            
            # Create progress bar for frames
            frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
            total_frames = len(frames)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each frame
            processed_frames = []
            for i, frame in enumerate(frames):
                # Convert frame to RGB (remove transparency if present)
                if frame.mode == 'RGBA':
                    background = Image.new('RGBA', frame.size, (255, 255, 255))
                    frame = Image.alpha_composite(background, frame)
                rgb_frame = frame.convert('RGB')
                
                # Convert to numpy array
                np_frame = np.array(rgb_frame)
                
                # Resize input based on resize factor
                if settings["resize_factor"] != 100:
                    scale = settings["resize_factor"] / 100
                    new_size = (int(np_frame.shape[1] * scale), int(np_frame.shape[0] * scale))
                    np_frame = cv2.resize(np_frame, new_size, interpolation=cv2.INTER_AREA)
                
                # Process frame
                processed_np_frame = st.session_state.upscaler.upscale(np_frame)
                
                # Convert back to PIL Image
                processed_frame = Image.fromarray(processed_np_frame)
                processed_frames.append(processed_frame)
                
                # Update progress
                progress = (i + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {i + 1}/{total_frames}")
            
            # Save processed GIF
            output_buffer = BytesIO()
            processed_frames[0].save(
                output_buffer,
                format='GIF',
                save_all=True,
                append_images=processed_frames[1:],
                duration=duration,
                loop=loop,
                optimize=False
            )
            output_buffer.seek(0)
            
            return output_buffer
    finally:
        # Cleanup after processing
        if st.session_state.upscaler:
            st.session_state.upscaler.cleanup()
            st.session_state.upscaler = None
            gc.collect()

def set_page_config():
    st.set_page_config(
        page_title="Image Enhancer Prototype",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        /* Add button text color style */
        .stDownloadButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.6em 1.2em;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        /* Hide deployment bar */
        header[data-testid="stHeader"] {
            display: none;
        }
        .main {
            background-color: #f7f7f7;
            color: #000000;
        }
        .stApp {
            background-color: #FFFFFF;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }
        
        p {
            color: #000000;
        }
        
        .stMarkdown {
            color: #000000;
        }
        
        /* Keep sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
    """, unsafe_allow_html=True)
    
def create_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Replace with actual Animately logo path
        st.image("images/animately.jpeg", width=150)
    
    with col2:
        st.header("Image Enhancer Prototype")
        st.markdown("*Powered by Advanced AI Upscaling Technology*")
    
    with col3:
        # Replace with actual Ampcome logo path
        st.image("images/ampcome.jpeg", width=150)
    
    st.markdown("---")
    
def main():
    set_page_config()
    create_header()
    initialize_session_state()
    settings = create_sidebar()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a GIF to upscale",
        type=SUPPORTED_GIF_TYPE,
        help="Select a GIF file to upscale (max 500KB)", 
    )
    
    if uploaded_file:
        
        file_size = len(uploaded_file.getvalue()) / 1024  # Size in KB
        if file_size > 500:
            st.error("File size exceeds 500KB limit. Please upload a smaller file.")
            return
        
        # Check if we need to process a new file
        new_file = (st.session_state.current_file != uploaded_file.name)
        
        if new_file:
            st.session_state.current_file = uploaded_file.name
            st.session_state.processed_gif = None
        
        # Create columns for before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original")
            st.image(uploaded_file, use_column_width=True)
        
        with col2:
            st.markdown("### Upscaled")
            if new_file:
                with st.spinner("Processing GIF..."):
                    try:
                        # Save uploaded GIF temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name
                        
                        # Process GIF
                        processed_gif_buffer = process_gif(temp_path, settings)
                        st.session_state.processed_gif = processed_gif_buffer
                        
                        # # Display processed GIF
                        # st.image(processed_gif_buffer, use_column_width=True)
                        
                        # # Add download button
                        # st.download_button(
                        #     label="Download upscaled GIF",
                        #     data=processed_gif_buffer,
                        #     file_name=f"upscaled_{uploaded_file.name}",
                        #     mime="image/gif"
                        # )
                        
                    except Exception as e:
                        st.error(f"Error processing GIF: {str(e)}")
                    finally:
                        # Cleanup
                        if 'temp_path' in locals():
                            os.unlink(temp_path)
                        if st.session_state.upscaler:
                            st.session_state.upscaler.cleanup()
                            st.session_state.upscaler = None
                        gc.collect()
                        
            if st.session_state.processed_gif:
                st.image(st.session_state.processed_gif, use_column_width=True)
                st.download_button(
                    label="Download upscaled GIF",
                    data=st.session_state.processed_gif,
                    file_name=f"upscaled_{uploaded_file.name}",
                    mime="image/gif"
                )

    st.markdown("---")
    st.markdown("""
    ### About Image Enhancer Prototype
    
    Image Enhancer is a professional-grade image upscaling solution developed by Ampcome for Animately. 
    It uses state-of-the-art AI models to enhance image resolution while maintaining quality.
    
    *Built with ❤️ by [Ampcome](https://ampcome.com) for [Animately](https://animately.co)*
    """)

if __name__ == "__main__":
    main()