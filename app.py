import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Emosense Live", layout="wide")
st.title("🎥 Real-Time Emotion Detector")

# --- LOAD MODEL (Cached agar tidak berat saat refresh) ---
@st.cache_resource
def load_my_model():
    # Pastikan file .h5 dan .xml ada di folder yang sama di GitHub
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()

# Daftar Label Emosi
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Pemetaan Emoji Raksasa
EMOJI_MAP = {
    "angry": "😡", 
    "disgust": "🤢", 
    "fear": "😨", 
    "happy": "😊", 
    "neutral": "😐", 
    "sad": "😢", 
    "surprise": "😲"
}

# --- KONFIGURASI STUN SERVER (PENTING UNTUK KONEKSI REAL-TIME) ---
# Daftar ini membantu menembus firewall ISP agar video bisa muncul
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    }
)

# --- LOGIKA PEMROSESAN FRAME VIDEO ---
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_emotion = None 

    def transform(self, frame):
        # Ubah frame ke format array OpenCV
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            # Preprocessing untuk Model CNN
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            # Prediksi Emosi
            preds = emotion_model.predict(roi, verbose=0)[0]
            self.last_emotion = EMOTIONS[preds.argmax()]
            
            # Gambar visual di frame video (Kotak dan Teks)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, self.last_emotion.upper(), (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# --- TAMPILAN ANTARMUKA (UI) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 📹 Video Feed")
    # Komponen Streaming Video
    ctx = webrtc_streamer(
        key="emotion-det",
        video_processor_factory=EmotionTransformer,
        rtc_configuration=RTC_CONFIG, 
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Menjalankan pemrosesan secara asinkron agar UI tidak freeze
    )

with col2:
    st.write("### 🎭 Status Emosi")
    
    # Cek apakah kamera sudah aktif dan mengirim data emosi
    if ctx.video_processor and ctx.video_processor.last_emotion is not None:
        current_emo = ctx.video_processor.last_emotion
        
        # Tampilkan Emoji Raksasa
        st.markdown(
            f"<h1 style='text-align: center; font-size: 150px; margin-top: 20px; margin-bottom: 0;'>{EMOJI_MAP[current_emo]}</h1>", 
            unsafe_allow_html=True
        )
        
        # Tampilkan Nama Emosi dengan warna hijau
        st.markdown(
            f"<h2 style='text-align: center; color: #4CAF50;'>{current_emo.upper()}</h2>", 
            unsafe_allow_html=True
        )
        
        # Tips singkat (User Experience)
        tips = {
            "happy": "Pertahankan senyummu! 😊",
            "sad": "Jangan bersedih, ceritakan pada temanmu. 💙",
            "angry": "Tarik napas dalam-dalam, mari rileks. 🧘",
            "neutral": "Kamu tampak tenang hari ini. ✨",
            "surprise": "Wow, ada apa itu? 😲",
            "fear": "Tenang, kamu aman di sini. 🛡️",
            "disgust": "Ada sesuatu yang tidak beres? 🤢"
        }
        st.info(tips.get(current_emo, "Sedang menganalisis ekspresimu..."))
    else:
        # Tampilan sebelum Start ditekan
        st.warning("Menunggu kamera aktif...")
        st.markdown(
            """
            1. Klik tombol **START** di sebelah kiri.
            2. Berikan izin (Allow) kamera pada browser.
            3. Pastikan wajah terlihat jelas di layar.
            """
        )

# --- FOOTER ---
st.markdown("---")
st.caption("Emosense App | Dibuat dengan Streamlit, OpenCV, & CNN")