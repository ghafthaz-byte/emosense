# app.py
from flask import Flask, render_template, Response, jsonify
from camera import Camera
import cv2
from tensorflow.keras.models import load_model
import threading
import time

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# Data Emosi Global yang akan diakses oleh kedua endpoint
last_emotion_data = {}

# --- PENTING: Load Model CNN dan Haar Cascade ---
try:
    # Ganti 'model_file_30epochs.h5' dengan path model Anda
    emotion_model = load_model('model_file_30epochs.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    print("Model dan Haar Cascade berhasil dimuat.")
except Exception as e:
    print(f"Error memuat model atau cascade: {e}")
    emotion_model = None
    face_cascade = None
# -----------------------------------------------

# Fungsi untuk memproses frame video dan menyimpan data emosi
def process_frame(frame):
    global last_emotion_data
    if emotion_model is None or face_cascade is None:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    current_emotion_data = {e: 0 for e in EMOTIONS}
    detected_face = False
    
    for (x, y, w, h) in faces:
        detected_face = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Logika Prediksi Emosi
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except:
            continue
            
        roi = roi_gray.astype('float') / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        # Menggunakan verbose=0 untuk menghindari log spam
        preds = emotion_model.predict(roi, verbose=0)[0]
        
        # Simpan data prediksi ke dictionary
        for i, emotion in enumerate(EMOTIONS):
            current_emotion_data[emotion] = float(preds[i] * 100) # Simpan sebagai persen

        emotion_probability = max(preds)
        label = EMOTIONS[preds.argmax()]
        
        label_text = f"{label}: {emotion_probability:.2f}"
        cv2.putText(frame, label_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Kita hanya memproses satu wajah
        break
        
    # Perbarui variabel global
    if detected_face:
        last_emotion_data = current_emotion_data
    else:
        # Jika tidak ada wajah, pertahankan data terakhir
        if not last_emotion_data:
             last_emotion_data = current_emotion_data 
        
    return frame

# Fungsi generator untuk streaming video (Motion JPEG)
def gen(camera):
    while True:
        frame = camera.get_frame() 
        if frame is None:
            continue
            
        processed_frame = process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route utama, menampilkan halaman HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk streaming video
@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ENDPOINT UNTUK DATA EMOSI
@app.route('/emotion_data')
def emotion_data():
    """Mengembalikan data probabilitas emosi terakhir dalam format JSON."""
    global last_emotion_data
    return jsonify(last_emotion_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)