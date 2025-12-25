import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Emosense Ultimate - Real-Time AI",
    page_icon="🧠",
    layout="wide"
)

# RTC Configuration (Penting agar kamera bisa jalan setelah di-deploy ke internet/cloud)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Custom CSS untuk tampilan premium
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1e88e5 0%, #00d4ff 100%);
    }
    .status-card {
        background-color: #ffffff; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1); border-left: 5px solid #1e88e5;
        color: #1a1a1a; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_assets()

EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

# --- LOGIKA PEMROSES VIDEO REAL-TIME ---
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip Horizontal (Mirroring)
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Preprocessing
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
            
            # Prediksi
            preds = emotion_model.predict(roi_resized.reshape(1, 48, 48, 1), verbose=0)[0]
            idx_max = np.argmax(preds)
            label = EMOTIONS_ENG[idx_max]
            emoji = EMOJI_MAP[label][1]
            prob = preds[idx_max] * 100

            # Gambar di Frame Video
            color = (229, 136, 30) # Biru
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{emoji} {label} ({prob:.1f}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# --- UI UTAMA ---
st.title("🧠 Emosense Ultimate: Real-Time AI")
st.markdown("Deteksi emosi wajah secara langsung melalui aliran video.")
st.write("---")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("🎥 Live Video Feed")
    webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=EmotionTransformer,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("Klik 'Start' untuk mengaktifkan kamera dan deteksi.")

with col_right:
    st.subheader("📝 Petunjuk Penggunaan")
    st.markdown("""
    <div class="status-card">
        <b>Cara Kerja:</b><br>
        1. Izinkan akses kamera pada browser.<br>
        2. Klik tombol <b>Start</b> di samping.<br>
        3. AI akan mendeteksi wajah dan emosi Anda secara real-time.<br>
        4. Hasil akan muncul langsung di atas kotak wajah.
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Tips:** Pastikan wajah Anda mendapat cahaya yang cukup untuk akurasi terbaik.")

st.divider()
st.caption("Emosense Real-Time Edition v5.0 | 2025")