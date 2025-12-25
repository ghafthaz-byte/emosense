import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# --- OPTIMASI MEMORI ---
@st.cache_resource
def load_assets():
    # Gunakan compile=False agar loading lebih ringan dan cepat
    model = load_model('model_file_30epochs.h5', compile=False)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_assets()

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Perkecil skala deteksi untuk meringankan CPU
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
            
            # Tambahkan batch dimension
            roi_reshaped = np.reshape(roi_resized, (1, 48, 48, 1))
            
            # Prediksi
            preds = emotion_model.predict(roi_reshaped, verbose=0)[0]
            label = EMOTIONS[np.argmax(preds)]
            
            # Drawing
            cv2.rectangle(img, (x, y), (x+w, y+h), (30, 136, 229), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 136, 229), 2)

        return frame.from_ndarray(img, format="bgr24")

st.title("🧠 Emosense Live")
st.write("Aplikasi sedang berjalan dalam mode Real-Time.")

webrtc_streamer(
    key="emosense-live",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

