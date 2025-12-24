import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- KONFIGURASI TAMPILAN ---
st.set_page_config(
    page_title="Emosense - Multi-Face AI",
    page_icon="👥",
    layout="wide"
)

# Custom CSS untuk UI yang lebih dinamis
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
    }
    .face-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #2575fc;
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

EMOTIONS_ENG = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": ("Marah", "😡"), "Disgust": ("Jijik", "🤢"), "Fear": ("Takut", "😨"), 
    "Happy": ("Bahagia", "😊"), "Neutral": ("Netral", "😐"), "Sad": ("Sedih", "😢"), 
    "Surprise": ("Terkejut", "😲")
}

st.title("👥 Emosense: Multi-Face Emotion AI")
st.markdown("Aplikasi ini sekarang dapat mendeteksi **banyak wajah sekaligus** dalam satu foto.")
st.write("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📸 Ambil Foto Grup")
    img_file = st.camera_input("Pastikan semua wajah menghadap ke kamera")

with col_right:
    st.subheader("📊 Hasil Analisis Wajah")
    
    if img_file:
        bytes_data = img_file.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img_draw = img.copy() # Untuk menggambar kotak wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi semua wajah yang ada
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            st.success(f"Berhasil mendeteksi {len(faces)} wajah!")
            
            # Buat tab untuk setiap wajah agar tidak berantakan
            face_tabs = st.tabs([f"Wajah #{i+1}" for i in range(len(faces))])

            for i, (x, y, w, h) in enumerate(faces):
                with face_tabs[i]:
                    # Beri label nomor pada gambar asli (opsional)
                    cv2.rectangle(img_draw, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(img_draw, f"#{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    # Crop wajah untuk prediksi
                    roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)) / 255.0
                    preds = emotion_model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
                    
                    idx_max = np.argmax(preds)
                    label_max, emoji_max = EMOJI_MAP[EMOTIONS_ENG[idx_max]]

                    # Header Ringkasan Wajah
                    st.markdown(f"""
                        <div class="face-header">
                            <h3>{emoji_max} Wajah #{i+1}: {label_max}</h3>
                            <p>Keyakinan Tertinggi: {preds[idx_max]*100:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Detail Probabilitas
                    st.write("**Detail Emosi:**")
                    for j, score in enumerate(preds):
                        label_indo, emoji = EMOJI_MAP[EMOTIONS_ENG[j]]
                        c1, c2 = st.columns([1, 4])
                        with c1:
                            st.write(f"{emoji} {label_indo}")
                        with c2:
                            st.progress(float(score))
                            st.caption(f"{score*100:.1f}%")
            
            # Tampilkan gambar dengan kotak deteksi di bawah tab
            st.divider()
            st.image(img_draw, channels="BGR", caption="Peta Deteksi Wajah", use_container_width=True)
            
        else:
            st.warning("⚠️ Tidak ada wajah yang terdeteksi. Coba posisi atau pencahayaan lain.")
    else:
        st.info("Ambil foto untuk memulai analisis multi-wajah.")

st.divider()
st.caption("Emosense Stable Build v3.0 | Multi-Face Ready")