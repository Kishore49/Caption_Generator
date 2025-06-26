import streamlit as st
import torch
from PIL import Image
import io
import base64
import requests
from io import BytesIO

# Try importing BLIP classes, handle ImportError gracefully
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    BlipProcessor = None
    BlipForConditionalGeneration = None

# Configure page
st.set_page_config(
    page_title="AI Image Captioning Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Available Models with their capabilities
MODELS = {
    "BLIP Large - High Accuracy": {
        "processor": "Salesforce/blip-image-captioning-large",
        "model": "Salesforce/blip-image-captioning-large", 
        "type": "blip",
        "description": "Larger BLIP model with improved accuracy"
    },
    "BLIP Base - Fast & Balanced": {
        "processor": "Salesforce/blip-image-captioning-base",
        "model": "Salesforce/blip-image-captioning-base",
        "type": "blip", 
        "description": "Original BLIP model, good balance of speed and accuracy"
    }
}

# Initialize the model
@st.cache_resource
def load_model(model_name):
    """Load the selected model for image captioning"""
    model_info = MODELS[model_name]
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        raise ImportError(
            "BLIP classes are not available in your transformers installation. "
            "Please upgrade transformers: pip install --upgrade transformers"
        )
    processor = BlipProcessor.from_pretrained(model_info["processor"])
    model = BlipForConditionalGeneration.from_pretrained(model_info["model"])
    return processor, model, model_info["type"]

def generate_caption(image, processor, model, model_type, caption_style="detailed"):
    """Generate caption for the uploaded image with different styles"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_length=75,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.8,
            do_sample=True
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Custom CSS for modern, trendy design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 400;
    }
    .upload-area {
        border: 2px dashed #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .upload-area:hover {
        border-color: #667eea;
        background: #f1f5f9;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    .result-card {
        background: #000 !important;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    .caption-text {
        font-size: 1.1rem;
        color: #fff !important;  /* White text */
        line-height: 1.6;
        font-weight: 500;
    }
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("""
    <div class="main-container">
        <div class="header">
            <h1 class="title">🎨 AI Image Captioning Studio</h1>
            <p class="subtitle">Transform your images into compelling captions with cutting-edge AI technology</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 AI Model Selection")
    selected_model = st.selectbox(
        "Choose AI Model:",
        list(MODELS.keys()),
        index=0,
        help="Select the AI model for caption generation."
    )
    model_info = MODELS[selected_model]
    st.markdown(f"*{model_info['description']}*")
    
    st.markdown("### 🎨 Caption Style")
    caption_style = st.radio(
        "Choose caption style:",
        ["detailed", "creative", "technical", "simple"],
        index=0,
        horizontal=True,
        help="Different styles produce different types of captions"
    )
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("### 📤 Upload Your Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help="Upload an image to generate a caption"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True) 
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("🎯 Generate Caption", use_container_width=True):
                with st.spinner(f"🔮 {selected_model.split(' -')[0]} is analyzing your image..."):
                    try:
                        processor, model, model_type = load_model(selected_model)
                        caption = generate_caption(image, processor, model, model_type, caption_style)
                        st.session_state.caption = caption
                        st.session_state.processed_image = image
                        st.session_state.model_used = selected_model
                        st.session_state.style_used = caption_style
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.info("💡 Try selecting a different model or check your internet connection for first-time model downloads.")
    with col2:
        st.markdown("### 🎭 Generated Caption")
        if hasattr(st.session_state, 'caption') and st.session_state.caption:
            st.markdown(f"""
            <div class="result-card">
                <h4>✨ AI Generated Caption:</h4>
                <p class="caption-text">"{st.session_state.caption}"</p>
                <small style="color: #64748b; font-style: italic;">
                    Generated using: {st.session_state.get('model_used', 'Unknown')} | 
                    Style: {st.session_state.get('style_used', 'Unknown')}
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card">
                <p style="text-align: center; color: #64748b; font-style: italic;">
                    Upload an image and click "Generate Caption" to see the AI-generated description here.
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    st.markdown("</div>", unsafe_allow_html=True) 
