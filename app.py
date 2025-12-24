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

# Custom CSS untuk mempercantik UI dan membuat "Card" hasil
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stCamera > div {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .emotion-card {
        background-color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e1e4e8;
    }
    .confidence-badge {
        background-color: #e3f2fd;
        display: inline-block;
        padding: 5px 20px;
        border-radius: 50px;
        color: #1565c0;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL (Cached) ---
@st.cache_resource
def load_my_model():
    # Memastikan model dan cascade tersedia
    model = load_model('model_file_30epochs.h5')
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return model, cascade

emotion_model, face_cascade = load_my_model()

# Daftar Label dan Emoji
EMOTIONS = ["Marah", "Jijik", "Takut", "Bahagia", "Netral", "Sedih", "Terkejut"]
EMOJI_MAP = {
    "Marah": "😡", "Jijik": "🤢", "Takut": "😨", 
    "Bahagia": "😊", "Netral": "😐", "Sedih": "😢", "Terkejut": "😲"
}

# --- HEADER ---
st.title("🧠 Emosense: Deteksi Emosi Berbasis AI")
st.markdown("### Analisis ekspresi wajah Anda secara instan dan akurat.")
st.write("---")

# --- LAYOUT UTAMA ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Ambil Foto")
    img_file = st.camera_input("Posisikan wajah tepat di tengah kamera")
    
    with st.expander("ℹ️ Tips Penggunaan"):
        st.write("""
        1. Pastikan wajah Anda mendapat cahaya yang cukup.
        2. Lepaskan kacamata atau penghalang wajah jika deteksi gagal.
        3. Klik 'Take Photo' dan tunggu AI bekerja.
        """)

with col_right:
    st.subheader("📊 Hasil Analisis")
    
    if img_file:
        # Konversi data gambar
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # Ambil wajah pertama
            (x, y, w, h) = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
            
            # Prediksi dengan Model
            preds = emotion_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
            idx = np.argmax(preds)
            label = EMOTIONS[idx]
            prob = preds[idx] * 100

            # Card Hasil Deteksi
            st.markdown(f"""
                <div class="emotion-card">
                    <h1 style='font-size: 130px; margin: 0;'>{EMOJI_MAP[label]}</h1>
                    <h2 style='color: #1e88e5; margin-bottom: 5px;'>{label}</h2>
                    <div class="confidence-badge">
                        Tingkat Keyakinan: {prob:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            
            # Visualisasi Grafik Batang
            st.markdown("#### Spektrum Emosi")
            chart_data = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
            # Menggunakan warna biru yang konsisten dengan tema
            st.bar_chart(chart_data, color='#1e88e5')
            
            # Pesan berdasarkan emosi
            tips = {
                "Bahagia": "Senyummu menular! Terus sebarkan energi positif. ✨",
                "Sedih": "Tidak apa-apa merasa sedih, ambil waktu untuk istirahat. 💙",
                "Marah": "Tarik napas perlahan... Mari kita rileks sejenak. 🧘",
                "Netral": "Kamu terlihat sangat tenang dan fokus. 🕊️",
                "Terkejut": "Wah, kejutan apa yang baru saja terjadi? 😲"
            }
            st.info(tips.get(label, "AI telah berhasil menganalisis ekspresimu."))

        else:
            st.error("⚠️ Wajah tidak terdeteksi. Coba ambil foto lagi dengan posisi wajah lebih dekat.")
    else:
        # Tampilan Placeholder saat belum ada foto
        st.info("Silakan ambil foto di sebelah kiri untuk melihat hasil analisis di sini.")
        st.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=150)

# --- FOOTER ---
st.divider()
f_col1, f_col2 = st.columns(2)
with f_col1:
    st.caption("Powered by Streamlit | OpenCV | TensorFlow")
with f_col2:
    st.markdown("<p style='text-align: right; color: gray;'>© 2025 Emosense Project</p>", unsafe_allow_html=True)