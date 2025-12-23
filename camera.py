# camera.py
import cv2

class Camera(object):
    def __init__(self):
        # Menggunakan kamera pertama (indeks 0)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Pastikan kamera dilepas saat objek dihancurkan
        self.video.release()

    def get_frame(self):
        # Baca frame dari kamera
        success, image = self.video.read()
        
        # Jika gagal, kembalikan frame kosong
        if not success:
            return None
            
        image = cv2.flip(image, 1)
        
        return image