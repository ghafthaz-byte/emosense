import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- KONFIGURASI TAMPILAN ---
st.set_page_config(
    page_title="Emosense - AI Emotion Detector",
    page_icon="😊",
    layout="wide"
)

# Custom CSS untuk UI Premium
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stCamera > div { border-radius: 20px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .emotion-card {
        background-color: white; padding: 30px; border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05); text-align: center; border: 1px solid #e1e4e8;
    }
    .confidence-badge {
        background-color: #e3f2fd; display: inline-block; padding: 5px 20px;
        border-radius: 50px; color: #1565c0; font-weight: bold; margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_my_model():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()

EMOTIONS = ["Marah", "Jijik", "Takut", "Bahagia", "Netral", "Sedih", "Terkejut"]
EMOJI_MAP = {
    "Marah": "😡", "Jijik": "🤢", "Takut": "😨", 
    "Bahagia": "😊", "Netral": "😐", "Sedih": "😢", "Terkejut": "😲"
}

# --- HEADER ---
st.title("🧠 Emosense: Deteksi Emosi Berbasis AI")
st.markdown("### Analisis ekspresi wajah Anda secara instan dan akurat.")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Ambil Foto")
    img_file = st.camera_input("Posisikan wajah tepat di tengah kamera")

with col_right:
    st.subheader("📊 Hasil Analisis Persentase")
    
    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            preds = emotion_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
            idx = np.argmax(preds)
            label = EMOTIONS[idx]
            prob_max = preds[idx] * 100

            # 1. Card Hasil Utama
            st.markdown(f"""
                <div class="emotion-card">
                    <h1 style='font-size: 130px; margin: 0;'>{EMOJI_MAP[label]}</h1>
                    <h2 style='color: #1e88e5; margin-bottom: 5px;'>{label}</h2>
                    <div class="confidence-badge">Dominan: {prob_max:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            
            # 2. Persentase Real-Time Per Kategori (Progress Bar)
            st.markdown("#### Detail Persentase Emosi:")
            for i in range(len(EMOTIONS)):
                score = float(preds[i])
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.write(f"{EMOJI_MAP[EMOTIONS[i]]} {EMOTIONS[i]}")
                with col_bar:
                    # Menampilkan progress bar dan persentase di sampingnya
                    st.progress(score)
                    st.caption(f"{score*100:.2f}%")

        else:
            st.error("⚠️ Wajah tidak terdeteksi. Coba ambil foto lagi.")
    else:
        st.info("Silakan ambil foto untuk melihat rincian persentase emosi.")

st.divider()
st.caption("Powered by Streamlit | OpenCV | TensorFlow")