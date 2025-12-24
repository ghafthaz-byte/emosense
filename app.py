import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import av

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Emosense Live Pro", layout="wide")
st.title("🎥 Real-Time Emotion Detector (Optimized)")

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {"angry": "😡", "disgust": "🤢", "fear": "😨", "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"}

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]}]}
)

# --- LOGIKA PEMROSESAN VIDEO (DENGAN SKIP FRAME) ---
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = None
        self.frame_count = 0 # Counter untuk skip frame

    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # HANYA PROSES SETIAP 5 FRAME (Mencegah Server Crash)
        if self.frame_count % 5 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = np.reshape(roi, (1, 48, 48, 1))

                preds = emotion_model.predict(roi, verbose=0)[0]
                self.last_emotion = EMOTIONS[preds.argmax()]
                
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
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Mencegah UI membeku
    )

with col2:
    st.write("### Status Emosi")
    if ctx.video_transformer and ctx.video_transformer.last_emotion is not None:
        current_emo = ctx.video_transformer.last_emotion
        st.markdown(f"<h1 style='text-align: center; font-size: 150px;'>{EMOJI_MAP[current_emo]}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{current_emo.upper()}</h2>", unsafe_allow_html=True)
    else:
        st.warning("Menunggu wajah terdeteksi...")