import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Emosense Ultimate", page_icon="🧠", layout="wide")

# Custom CSS untuk Kontras Teks dan Progress Bar
st.markdown("""
    <style>
    .face-card {
        background-color: #ffffff !important; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15); border-left: 5px solid #1e88e5;
        margin-bottom: 20px; color: #1a1a1a !important;
    }
    .face-card h3, .face-card p, .face-card b { color: #1a1a1a !important; }
    .group-card {
        background-color: #f0f7ff !important; padding: 20px; border-radius: 15px;
        border: 2px solid #bbdefb; text-align: center; color: #1a1a1a !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1e88e5 0%, #00d4ff 100%);
    }
    video { transform: scaleX(-1); } /* Mirror Preview */
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS (Optimasi Memori) ---
@st.cache_resource
def load_assets():
    model = load_model('model_file_30epochs.h5', compile=False)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_assets()

EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

# --- HEADER ---
st.title("🧠 Emosense Ultimate: Multi-Face AI")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Input Kamera")
    img_file = st.camera_input("Ambil foto untuk analisis")
    
with col_right:
    st.subheader("📊 Analisis Hasil")
    
    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.flip(img, 1) # Sinkronisasi Mirror
        
        img_draw = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            all_preds = []
            tabs = st.tabs([f"Wajah #{i+1}" for i in range(len(faces))])

            for i, (x, y, w, h) in enumerate(faces):
                roi_gray = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (48, 48)) / 255.0
                
                preds = emotion_model.predict(roi_resized.reshape(1, 48, 48, 1), verbose=0)[0]
                all_preds.append(preds)
                
                idx_max = np.argmax(preds)
                label, emoji = EMOJI_MAP[EMOTIONS_ENG[idx_max]]

                cv2.rectangle(img_draw, (x, y), (x+w, y+h), (30, 136, 229), 3)
                cv2.putText(img_draw, f"#{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 136, 229), 2)

                with tabs[i]:
                    st.markdown(f'<div class="face-card"><h3>{emoji} {label}</h3><p>Akurasi: {preds[idx_max]*100:.1f}%</p></div>', unsafe_allow_html=True)
                    for j, score in enumerate(preds):
                        l_indo, emo = EMOJI_MAP[EMOTIONS_ENG[j]]
                        st.write(f"{emo} {l_indo}")
                        st.progress(float(score))

            if len(faces) > 1:
                st.divider()
                avg_preds = np.mean(all_preds, axis=0)
                g_label, g_emoji = EMOJI_MAP[EMOTIONS_ENG[np.argmax(avg_preds)]]
                st.markdown(f'<div class="group-card"><h1>{g_emoji}</h1><h3>Mood Kelompok: {g_label}</h3></div>', unsafe_allow_html=True)

            st.divider()
            st.image(img_draw, channels="BGR", use_container_width=True)
        else:
            st.warning("Wajah tidak terdeteksi.")
    else:
        st.info("Silakan ambil foto.")