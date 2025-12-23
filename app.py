import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Judul Aplikasi
st.set_page_config(page_title="Emosense - Deteksi Emosi Wajah", layout="wide")
st.title("😊 Deteksi Emosi Wajah Real-Time")

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- UI SIDEBAR ---
st.sidebar.info("Aplikasi ini mendeteksi emosi wajah menggunakan CNN dan OpenCV.")

# --- KAMERA INPUT ---
img_file_buffer = st.camera_input("Ambil foto untuk cek emosi")

if img_file_buffer is not None:
    # Ubah buffer gambar ke format OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Proses Deteksi
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            preds = emotion_model.predict(roi, verbose=0)[0]
            label = EMOTIONS[preds.argmax()]
            prob = max(preds) * 100

            # Tampilkan Hasil
            st.success(f"Terdeteksi: **{label}** ({prob:.2f}%)")
            
            # Buat Grafik Bar untuk semua emosi
            chart_data = {emo: float(p) for emo, p in zip(EMOTIONS, preds)}
            st.bar_chart(chart_data)
    else:
        st.warning("Wajah tidak terdeteksi. Coba posisi wajah yang lebih jelas.")