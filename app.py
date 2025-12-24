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

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
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
st.markdown("### Temukan perasaanmu hanya dalam satu jepretan foto!")
st.divider()

# --- LAYOUT UTAMA ---
col_cam, col_res = st.columns([1, 1], gap="large")

with col_cam:
    st.subheader("📸 Ambil Foto")
    img_file = st.camera_input("Pastikan wajah terlihat jelas dan pencahayaan cukup")

with col_res:
    st.subheader("📊 Hasil Analisis")
    
    if img_file:
        # Proses Gambar
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # Prediksi
            (x, y, w, h) = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            preds = emotion_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
            idx = np.argmax(preds)
            label = EMOTIONS[idx]
            prob = preds[idx] * 100

            # Tampilan Emoji dan Label Besar
            st.markdown(f"""
                <div class="emotion-card">
                    <h1 style='font-size: 120px; margin: 0;'>{EMOJI_MAP[label]}</h1>
                    <h2 style='color: #2E86C1;'>{label}</h2>
                    <p style='font-size: 18px; color: gray;'>Tingkat Keyakinan: {prob:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("#### Skor Probabilitas Lengkap:")
            # Grafik Bar yang cantik
            chart_dict = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
            st.bar_chart(chart_dict)
            
        else:
            st.warning("⚠️ Wajah tidak terdeteksi. Silakan coba ambil foto lagi dengan posisi wajah tegak ke kamera.")
    else:
        st.info("Menunggu foto diambil... Gunakan modul kamera di sebelah kiri.")

# --- FOOTER ---
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.caption("Dibuat dengan ❤️ menggunakan Streamlit, OpenCV, dan TensorFlow.")
with footer_col2:
    st.markdown("<p style='text-align: right; color: gray;'>© 2025 Emosense Project</p>", unsafe_allow_html=True)