import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Emosense Real-Time AI",
    page_icon="🧠",
    layout="wide"
)

# Konfigurasi server untuk koneksi WebRTC (Penting untuk Cloud Hosting)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 2. LOAD ASSETS (Optimasi Cache) ---
@st.cache_resource
def load_assets():
    try:
        # Load model tanpa compile untuk menghemat memori
        model = load_model('model_file_30epochs.h5', compile=False)
        cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return model, cascade
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

emotion_model, face_cascade = load_assets()

# Definisi Emosi
EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

# --- 3. LOGIKA PEMROSES VIDEO (Real-Time) ---
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Ubah frame ke format ndarray (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Mirroring untuk kenyamanan user
        img = cv2.flip(img, 1)
        
        if emotion_model is None:
            return frame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah (Scale factor 1.3 untuk performa lebih ringan)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Crop area wajah (ROI)
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi_reshaped = np.reshape(roi_resized, (1, 48, 48, 1))
            
            # Prediksi Emosi
            preds = emotion_model.predict(roi_reshaped, verbose=0)[0]
            idx_max = np.argmax(preds)
            label_eng = EMOTIONS_ENG[idx_max]
            emoji = EMOJI_MAP[label_eng][1]
            accuracy = preds[idx_max] * 100

            # Gambar UI di Frame Video
            color = (30, 136, 229) # Biru Emosense
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{emoji} {label_eng} ({accuracy:.1f}%)", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Kembalikan frame yang sudah diproses
        return frame.from_ndarray(img, format="bgr24")

# --- 4. TAMPILAN USER INTERFACE (UI) ---
st.title("🧠 Emosense Ultimate: Real-Time AI")
st.markdown("""
Aplikasi ini mendeteksi emosi wajah secara langsung menggunakan **Deep Learning (CNN)**. 
Dideploy melalui **Hugging Face Spaces** untuk performa yang lebih stabil.
""")

st.write("---")

col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("🎥 Live Feed")
    # Inisialisasi Streamer
    webrtc_streamer(
        key="emotion-realtime",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("ℹ️ Informasi")
    st.info("""
    **Cara Penggunaan:**
    1. Klik tombol **START** di layar video.
    2. Berikan izin akses kamera pada browser Anda.
    3. AI akan secara otomatis mendeteksi wajah dan emosi Anda.
    """)
    
    st.success("""
    **Keunggulan Versi HF:**
    - Memori lebih lega (16GB RAM).
    - Deteksi per frame yang lebih mulus.
    - Minim risiko 'Segmentation Fault'.
    """)

st.divider()
st.caption("Emosense Real-Time v5.1 | Powered by Hugging Face & Streamlit")