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

# Custom CSS untuk Mirror Kamera dan Kontras Teks Universal
st.markdown("""
    <style>
    /* 1. Mirroring Kamera di Layar (Preview) */
    video {
        transform: scaleX(-1);
    }
    
    /* 2. Memastikan teks di kartu hasil selalu terbaca (Terang/Gelap) */
    .face-card {
        background-color: #ffffff !important; 
        padding: 20px; 
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15); 
        border-left: 5px solid #1e88e5;
        margin-bottom: 20px;
    }
    
    /* Warna teks paksa agar tetap gelap di kartu putih */
    .face-card h3, .face-card p, .face-card b {
        color: #1a1a1a !important;
    }

    .group-card {
        background-color: #f0f7ff !important; 
        padding: 20px; 
        border-radius: 15px;
        border: 2px solid #bbdefb; 
        text-align: center;
    }
    
    .group-card h1, .group-card h3, .group-card p, .group-card b {
        color: #1a1a1a !important;
    }

    /* Warna Progress Bar agar kontras */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #1e88e5 0%, #00d4ff 100%);
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
    """Meningkatkan kontras untuk area gelap (Auto-Enhance)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_img)

# --- HEADER ---
st.title("🧠 Emosense Ultimate: Multi-Face AI")
st.markdown("Deteksi emosi kelompok dengan teknologi **Auto-Enhance** dan **Mirror Sync**.")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Input Kamera")
    img_file = st.camera_input("Ambil foto (Posisi cermin sudah aktif)")
    
with col_right:
    st.subheader("📊 Analisis Hasil")
    
    if img_file:
        # 1. Decode gambar mentah
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 2. SYNC MIRROR: Membalik gambar secara horizontal agar sama dengan preview
        img = cv2.flip(img, 1) 
        
        img_draw = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            all_preds = []
            st.success(f"Berhasil menganalisis {len(faces)} wajah")
            
            # Buat Tab untuk setiap wajah
            tabs = st.tabs([f"Wajah #{i+1}" for i in range(len(faces))])

            for i, (x, y, w, h) in enumerate(faces):
                # Preprocessing ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_enhanced = enhance_image(roi_gray)
                roi_resized = cv2.resize(roi_enhanced, (48, 48)) / 255.0
                
                # Prediksi Model
                preds = emotion_model.predict(roi_resized.reshape(1, 48, 48, 1), verbose=0)[0]
                all_preds.append(preds)
                
                idx_max = np.argmax(preds)
                label_indo, emoji = EMOJI_MAP[EMOTIONS_ENG[idx_max]]

                # Menggambar kotak pada gambar utama (yang sudah di-flip)
                cv2.rectangle(img_draw, (x, y), (x+w, y+h), (30, 136, 229), 3)
                cv2.putText(img_draw, f"#{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 136, 229), 2)

                with tabs[i]:
                    st.markdown(f"""
                        <div class="face-card">
                            <h3>{emoji} Wajah #{i+1}: {label_indo}</h3>
                            <p><b>Akurasi Prediksi: {preds[idx_max]*100:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Detail Probabilitas:**")
                    for j, score in enumerate(preds):
                        l_indo, emo = EMOJI_MAP[EMOTIONS_ENG[j]]
                        c1, c2 = st.columns([1, 4])
                        with c1: 
                            # Menggunakan warna teks standar agar adaptif di kolom
                            st.write(f"{emo} {l_indo}")
                        with c2: 
                            st.progress(float(score))
                            st.caption(f"{score*100:.1f}%")

            # --- ANALISIS KELOMPOK ---
            if len(faces) > 1:
                st.divider()
                st.subheader("👨‍👩‍👧‍👦 Ringkasan Mood Kelompok")
                avg_preds = np.mean(all_preds, axis=0)
                group_idx = np.argmax(avg_preds)
                g_label, g_emoji = EMOJI_MAP[EMOTIONS_ENG[group_idx]]
                
                st.markdown(f"""
                    <div class="group-card">
                        <h1 style='font-size: 60px; margin:0;'>{g_emoji}</h1>
                        <h3>Suasana Dominan: {g_label}</h3>
                        <p>Berdasarkan analisis {len(faces)} wajah, kelompok ini terlihat <b>{g_label}</b>.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()
            # Menampilkan gambar hasil deteksi yang sudah diproses
            st.image(img_draw, channels="BGR", caption="Hasil Deteksi Wajah (Sinkronisasi Cermin Berhasil)", use_container_width=True)
        else:
            st.warning("⚠️ Wajah tidak terdeteksi. Silakan coba lagi dengan posisi wajah lebih jelas.")
    else:
        st.info("💡 Klik 'Take Photo' di sebelah kiri untuk memulai analisis emosi.")

st.divider()
st.caption("Emosense Ultimate Final Build v4.2 | 2025")