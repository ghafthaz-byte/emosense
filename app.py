import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Emosense Pro", layout="wide")
st.title("😊 Emosense: Emotion Detector")
st.markdown("Ambil foto wajahmu untuk menganalisis emosi secara akurat.")

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    # Pastikan file ini ada di root folder GitHub kamu
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {
    "angry": "😡", "disgust": "🤢", "fear": "😨", 
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprise": "😲"
}

# --- INPUT KAMERA ---
img_file = st.camera_input("Klik tombol di bawah untuk mengambil foto")

if img_file:
    # Konversi file gambar ke format OpenCV
    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi Wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        st.divider()
        col1, col2 = st.columns([1, 1])
        
        # Ambil wajah pertama yang terdeteksi
        (x, y, w, h) = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float') / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        # Prediksi
        preds = emotion_model.predict(roi, verbose=0)[0]
        label = EMOTIONS[preds.argmax()]
        prob = max(preds) * 100

        with col1:
            st.image(img, channels="BGR", caption="Wajah Terdeteksi", use_container_width=True)
        
        with col2:
            st.markdown(f"<h1 style='text-align: center; font-size: 150px; margin-bottom: 0;'>{EMOJI_MAP[label]}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{label.upper()} ({prob:.2f}%)</h2>", unsafe_allow_html=True)
            
            # Tampilkan Grafik Probabilitas
            st.write("### Analisis Detail:")
            chart_data = {emo: float(p) for emo, p in zip(EMOTIONS, preds)}
            st.bar_chart(chart_data)
            
            # Tips Berdasarkan Emosi
            tips = {
                "happy": "Teruslah tersenyum! Hari ini milikmu. ✨",
                "sad": "Tidak apa-apa merasa sedih. Semuanya akan membaik. 💙",
                "angry": "Tarik napas dalam-dalam, mari rileks sejenak. 🧘",
                "neutral": "Tetap tenang dan fokus. Kamu luar biasa! 🕊️"
            }
            st.info(tips.get(label, "Analisis ekspresi berhasil diselesaikan."))
    else:
        st.error("Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas.")

st.markdown("---")
st.caption("Dibuat dengan Streamlit & CNN Model")