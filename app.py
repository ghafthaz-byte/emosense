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

# Custom CSS untuk mempercantik progress bar dan card
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
    }
    .emotion-label {
        font-weight: bold;
        margin-bottom: -15px;
    }
    .confidence-text {
        font-size: 0.85rem;
        color: #6c757d;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()

# Daftar Label Asli (Sesuai model)
EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# Pemetaan ke Bahasa Indonesia untuk Tampilan
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

st.title("🧠 Emosense: Deteksi Emosi Berbasis AI")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Ambil Foto")
    img_file = st.camera_input("Posisikan wajah tepat di tengah kamera")

with col_right:
    st.subheader("📊 Analisis Probabilitas")
    
    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            preds = emotion_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
            
            # Menampilkan Emosi Dominan dalam Card
            idx_max = np.argmax(preds)
            label_max, emoji_max = EMOJI_MAP[EMOTIONS_ENG[idx_max]]
            
            st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center; border: 1px solid #eee;">
                    <h1 style='font-size: 80px; margin: 0;'>{emoji_max}</h1>
                    <h2 style='color: #1e88e5; margin: 0;'>{label_max}</h2>
                    <p style='color: gray;'>Keyakinan: {preds[idx_max]*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("### Detail Probabilitas:")
            
            # --- BAGIAN PROGRESS BAR (Seperti di Gambar) ---
            for i, score in enumerate(preds):
                label_indo, emoji = EMOJI_MAP[EMOTIONS_ENG[i]]
                
                # Baris Label dan Persentase
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"<p class='emotion-label'>{emoji} {label_indo}</p>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<p class='confidence-text'>{score*100:.1f}%</p>", unsafe_allow_html=True)
                
                # Progress Bar Streamlit
                st.progress(float(score))
        else:
            st.warning("⚠️ Wajah tidak terdeteksi. Silakan coba lagi.")
    else:
        st.info("Menunggu foto diambil untuk memulai analisis...")

st.divider()
st.caption("Emosense Stable Build v2.0")