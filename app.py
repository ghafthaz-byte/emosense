import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Emosense Ultimate - Multi-Face AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS untuk Mirror Kamera dan Kontras Teks
st.markdown("""
    <style>
    /* Mirroring Kamera */
    video {
        transform: scaleX(-1);
    }
    
    /* Memastikan teks terbaca di Light/Dark Mode */
    .face-card {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
        border-left: 5px solid #1e88e5;
        margin-bottom: 20px;
        color: #1a1a1a; /* Warna teks gelap agar terlihat di mode putih */
    }
    
    .face-card h3 {
        color: #1e88e5 !important;
    }

    .group-card {
        background-color: #f0f7ff; 
        padding: 20px; 
        border-radius: 15px;
        border: 1px solid #bbdefb; 
        text-align: center;
        color: #1a1a1a;
    }

    /* Memperbaiki warna teks caption dan subheader agar adaptif */
    .stMarkdown p, .stMarkdown h3 {
        color: inherit;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_assets()

EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

def enhance_image(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

# --- HEADER ---
st.title("🧠 Emosense Ultimate: Multi-Face AI")
st.markdown("Deteksi emosi kelompok dengan teknologi **Auto-Enhance** dan **Sentimen Analisis**.")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Input Kamera (Mode Cermin)")
    img_file = st.camera_input("Ambil foto")
    
with col_right:
    st.subheader("📊 Analisis Hasil")
    
    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # Karena kamera dimirror di UI, kita harus flip gambarnya agar hasil kotak sesuai
        img = cv2.flip(img, 1) 
        
        img_draw = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            all_preds = []
            st.success(f"Berhasil menganalisis {len(faces)} wajah")
            
            tabs = st.tabs([f"Wajah #{i+1}" for i in range(len(faces))])

            for i, (x, y, w, h) in enumerate(faces):
                roi_gray = gray[y:y+h, x:x+w]
                roi_enhanced = enhance_image(roi_gray)
                roi_resized = cv2.resize(roi_enhanced, (48, 48)) / 255.0
                
                preds = emotion_model.predict(roi_resized.reshape(1, 48, 48, 1), verbose=0)[0]
                all_preds.append(preds)
                
                idx_max = np.argmax(preds)
                label_indo, emoji = EMOJI_MAP[EMOTIONS_ENG[idx_max]]

                cv2.rectangle(img_draw, (x, y), (x+w, y+h), (30, 136, 229), 3)
                cv2.putText(img_draw, f"#{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 136, 229), 2)

                with tabs[i]:
                    st.markdown(f"""
                        <div class="face-card">
                            <h3>{emoji} Wajah #{i+1}: {label_indo}</h3>
                            <p><b>Keyakinan: {preds[idx_max]*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for j, score in enumerate(preds):
                        l_indo, emo = EMOJI_MAP[EMOTIONS_ENG[j]]
                        c1, c2 = st.columns([1, 4])
                        with c1: st.write(f"{emo} {l_indo}")
                        with c2: 
                            st.progress(float(score))
                            st.caption(f"{score*100:.1f}%")

            if len(faces) > 1:
                st.divider()
                st.subheader("👨‍👩‍👧‍👦 Ringkasan Mood Kelompok")
                avg_preds = np.mean(all_preds, axis=0)
                group_idx = np.argmax(avg_preds)
                g_label, g_emoji = EMOJI_MAP[EMOTIONS_ENG[group_idx]]
                
                st.markdown(f"""
                    <div class="group-card">
                        <h1 style='font-size: 60px; margin:0;'>{g_emoji}</h1>
                        <h3 style='color: #1e88e5;'>Mood Dominan: {g_label}</h3>
                        <p>Secara keseluruhan, kelompok Anda terlihat <b>{g_label}</b>.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.image(img_draw, channels="BGR", caption="Hasil Deteksi Wajah (Flipped to Match Mirror)", use_container_width=True)
        else:
            st.warning("Wajah tidak terdeteksi. Pastikan pencahayaan cukup.")
    else:
        st.info("Silakan ambil foto untuk memulai.")

st.divider()
st.caption("Emosense Ultimate Edition v4.1 | Fix Mirror & Contrast")