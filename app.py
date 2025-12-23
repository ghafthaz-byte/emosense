import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Konfigurasi Halaman
st.set_page_config(page_title="Emosense Live", layout="wide")
st.title("🎥 Real-Time Emotion Detector")

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {
    "angry": "😡", "disgust": "🤢", "fear": "😨", 
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

# Konfigurasi Server STUN untuk koneksi yang lebih stabil
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = None # Default kosong agar tidak error

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            preds = emotion_model.predict(roi, verbose=0)[0]
            self.last_emotion = EMOTIONS[preds.argmax()]
            
            # Gambar visual di frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, self.last_emotion, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# --- TAMPILAN UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Video Feed")
    ctx = webrtc_streamer(
        key="emotion-det",
        video_transformer_factory=EmotionTransformer,
        rtc_configuration=RTC_CONFIG, # Menggunakan config STUN yang lebih lengkap
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.write("### Status Emosi")
    # Perbaikan: Cek apakah ctx aktif dan video_transformer sudah mengirim data
    if ctx.video_transformer and ctx.video_transformer.last_emotion is not None:
        current_emo = ctx.video_transformer.last_emotion
        st.markdown(f"<h1 style='text-align: center; font-size: 150px; margin-bottom: 0;'>{EMOJI_MAP[current_emo]}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{current_emo.upper()}</h2>", unsafe_allow_html=True)
        
        # Tambahan: Tips singkat berdasarkan emosi
        tips = {
            "happy": "Pertahankan senyummu! 😊",
            "sad": "Jangan bersedih, semua akan baik-baik saja. 💙",
            "angry": "Tarik napas dalam-dalam, mari rileks sejenak. 🧘",
            "neutral": "Kamu tampak tenang hari ini. ✨"
        }
        st.info(tips.get(current_emo, "Sedang menganalisis ekspresimu..."))
    else:
        st.warning("Menunggu kamera aktif atau wajah terdeteksi...")
        st.write("Silakan klik **Start** dan pastikan wajah terlihat jelas di kamera.")