import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import os
from datetime import datetime
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import shutil
import uuid

st.set_page_config(
    page_title="Face Recognition System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
    def get_config(self):
        config = super().get_config()
        return config

SESSION_REF_DIR = 'reference_images'

if 'initialized' not in st.session_state:
    if os.path.exists(SESSION_REF_DIR):
        shutil.rmtree(SESSION_REF_DIR)
    os.makedirs(SESSION_REF_DIR, exist_ok=True)
    
    st.session_state.initialized = True
    st.session_state.reference_images = []
    st.session_state.verification_results = None

os.makedirs(SESSION_REF_DIR, exist_ok=True)

st.markdown("""
<style>
    /* Hide default elements */
    footer {visibility: hidden;}
    
    /* Main styling */
    .main {
        padding: 0;
    }
    
    /* Custom header */
    .header {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg,
        color: white;
        border-radius: 0;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color:
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color:
        border-bottom-color:
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_siamese_model():
    st.sidebar.markdown(f"TF Version: `{tf.__version__}`")
    model_path = 'siamesemodelv2_1.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        return None

    try:
        model = load_model(model_path, custom_objects={'L1Dist': L1Dist})
        st.sidebar.success("Siamese Model loaded")
        return model
    except Exception as e1:
        st.error(f"Error loading model: {e1}")
        return None

if 'model' not in st.session_state:
    st.session_state.model = load_siamese_model()


CROP_BOX_SIZE = 400 

def get_centered_crop_coords(height, width, box_size):
    size = min(box_size, height, width)
    x_start = (width - size) // 2
    y_start = (height - size) // 2
    x_end = x_start + size
    y_end = y_start + size
    return x_start, y_start, x_end, y_end

def preprocess_tf(file_path):
    """
    STRICT TensorFlow Preprocessing (Matching faceid.py logic)
    """
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

def get_processing_stages(frame):
    h, w = frame.shape[:2]
    raw_viz = frame.copy()
    x1, y1, x2, y2 = get_centered_crop_coords(h, w, CROP_BOX_SIZE)
    cv2.rectangle(raw_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
    return raw_viz, crop, resized

def load_reference_images():
    if os.path.exists(SESSION_REF_DIR):
        files = [f for f in os.listdir(SESSION_REF_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        return sorted(files)
    return []


class FaceDetectorVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        with self.frame_lock:
            self.out_image = img.copy()

        height, width = img.shape[:2]
        
        cv2.putText(
            img, 
            f"Resolution: {width}x{height}", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 0, 255),
            2
        )

        if width < 1200: 
            status_text = "Stabilising resolution"
            color = (0, 255, 255)
            
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            cv2.putText(img, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
        else:
            status_text = "Resolution stabilised"
            color = (0, 255, 0)
            
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50
            
            cv2.putText(img, status_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            x1, y1, x2, y2 = get_centered_crop_coords(height, width, CROP_BOX_SIZE)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            cv2.putText(img, f"Region: {CROP_BOX_SIZE}x{CROP_BOX_SIZE}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("""
<div class="header">
    <h1>🔐 Face Recognition System</h1>
    <p>Using <code>siamesemodelv2_1.h5</code> (Optimized)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📸 Part 1: Collect References", "🔍 Part 2: Verify Face"])

VIDEO_CONSTRAINTS = {
    "video": {
        "width": {"min": 640, "ideal": 1280, "max": 1920},
        "height": {"min": 480, "ideal": 720, "max": 1080},
    },
    "audio": False
}

with tab1:
    st.markdown("## Collect Reference Images")
    
    col_camera, col_gallery = st.columns([1.5, 1])
    
    with col_camera:
        st.subheader("📷 Live Camera Feed")
        
        # WebRTC Configuration (STUN Servers for Cloud Deployment)
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        })
        
        ctx_ref = webrtc_streamer(
            key="ref-video",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=FaceDetectorVideoProcessor,
            media_stream_constraints=VIDEO_CONSTRAINTS,
            async_processing=True
        )
        
        if st.button("📸 Capture Image", key="btn_capture_ref"):
            if ctx_ref.video_processor:
                with ctx_ref.video_processor.frame_lock:
                     out_image = ctx_ref.video_processor.out_image
                
                if out_image is not None:
                    h, w = out_image.shape[:2]
                    if w < 1200:
                        st.warning(f"⚠️ Stabilization Wait...")
                    else:
                        current_count = len(load_reference_images())
                        if current_count >= 50:
                            st.warning("⚠️ Max images reached")
                        else:
                            x1, y1, x2, y2 = get_centered_crop_coords(h, w, CROP_BOX_SIZE)
                            crop_bgr = out_image[y1:y2, x1:x2]
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            path = os.path.join(SESSION_REF_DIR, f"ref_{current_count + 1}_{timestamp}.jpg")
                            cv2.imwrite(path, crop_bgr) 
                            
                            st.success(f"✓ Capture {current_count+1}")
                            st.session_state.reference_images = load_reference_images()
                else:
                    st.warning("⚠️ No frame.")
        
    with col_gallery:
        st.subheader("🖼️ Gallery")
        references = load_reference_images()
        count = len(references)
        col_count, col_prog = st.columns([1, 2])
        col_count.metric("Total", f"{count}/50")
        col_prog.progress(min(count / 50.0, 1.0))
        
        if count > 0:
            if st.button("🔄 Refresh"):
                 st.session_state.reference_images = load_reference_images()
                 st.rerun()

            cols = st.columns(3)
            for idx, ref_file in enumerate(references):
                col1 = idx % 3
                with cols[col1]:
                    img_path = os.path.join(SESSION_REF_DIR, ref_file)
                    st.image(img_path, caption=f"#{idx+1}", width=80)
        
        if count > 0 and st.button("🗑️ Clear All", key="clear_all"):
            if os.path.exists(SESSION_REF_DIR):
                shutil.rmtree(SESSION_REF_DIR)
            os.makedirs(SESSION_REF_DIR, exist_ok=True)
            st.session_state.reference_images = []
            st.rerun()

with tab2:
    st.markdown("## Verify Your Face")
    
    references = load_reference_images()
    ref_count = len(references)
    
    if ref_count < 10:
        st.warning(f"⚠️ Need more reference images ({ref_count}). Collect at least 10.")
    else:
        st.markdown(f"📊 Using **{ref_count} reference images**")
        
        col_camera, col_results = st.columns([1.5, 1])
        
        with col_camera:
            st.subheader("📷 Live Camera Feed")
            
            ctx_test = webrtc_streamer(
                key="test-video",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_processor_factory=FaceDetectorVideoProcessor,
                media_stream_constraints=VIDEO_CONSTRAINTS,
                async_processing=True
            )
            
            if st.button("📸 Capture & Verify", key="btn_capture_verify"):
                if ctx_test.video_processor:
                     with ctx_test.video_processor.frame_lock:
                         out_image = ctx_test.video_processor.out_image
                     
                     if out_image is not None:
                        h, w = out_image.shape[:2]
                        if w < 1200:
                            st.warning(f"⚠️ Stabilization Wait...")
                        else:
                            st.info("Processing...")
                            
                            x1, y1, x2, y2 = get_centered_crop_coords(h, w, CROP_BOX_SIZE)
                            query_crop_bgr = out_image[y1:y2, x1:x2]
                            
                            temp_query_path = os.path.join(SESSION_REF_DIR, "temp_query.jpg")
                            cv2.imwrite(temp_query_path, query_crop_bgr)
                            
                            if st.session_state.model is not None:
                                try:
                                    query_tensor = preprocess_tf(temp_query_path)
                                    
                                    results = []
                                    for ref_file in references:
                                        ref_path = os.path.join(SESSION_REF_DIR, ref_file)
                                        
                                        ref_tensor = preprocess_tf(ref_path)
                                        
                                        prediction = st.session_state.model.predict(
                                            [np.expand_dims(query_tensor, axis=0),
                                             np.expand_dims(ref_tensor, axis=0)],
                                            verbose=0
                                        )
                                        results.append(float(prediction[0][0]))
                                    
                                    if results:
                                        results_arr = np.array(results)
                                        
                                        detection_threshold = 0.5
                                        verification_threshold = 0.7 
                                        
                                        matches = np.sum(results_arr > detection_threshold)
                                        score = matches / len(references)
                                        verified = score > verification_threshold
                                        
                                        st.session_state.verification_results = {
                                            "match_count": int(matches),
                                            "total_references": len(references),
                                            "verification_score": float(score),
                                            "verified": bool(verified),
                                            "captured_image": query_crop_bgr
                                        }
                                    
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                finally:
                                    try: os.remove(temp_query_path) 
                                    except: pass
                            else:
                                st.error("Model not loaded.")
                     else:
                        st.warning("⚠️ No frame available.")
        
        with col_results:
            st.subheader("📊 Results")
            if st.session_state.verification_results:
                res = st.session_state.verification_results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Matches", f"{res['match_count']}/{res['total_references']}")
                with col2:
                    st.metric("Score", f"{res['verification_score']*100:.1f}%")
                
                if res['verified']:
                    st.success(f"✅ VERIFIED")
                else:
                    st.error(f"❌ UNVERIFIED")
                
                st.image(cv2.cvtColor(res["captured_image"], cv2.COLOR_BGR2RGB), caption="Input (400x400)", width=150)
                
                if st.button("🔄 Reset"):
                    st.session_state.verification_results = None
                    st.rerun()

with st.sidebar:
    st.markdown("### ℹ️ Config")
    st.markdown(f"Model: **Siamese V2.1**")
    st.markdown(f"Crop: **{CROP_BOX_SIZE}x{CROP_BOX_SIZE}**")
    st.markdown("Status: **Optimized Pipeline**")
