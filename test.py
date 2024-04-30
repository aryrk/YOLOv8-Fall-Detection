import logging
from ultralytics import YOLO

# Konfigurasi pengaturan logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def object(src):
    try:
        # Inisialisasi model YOLO dengan menggunakan model terbaik ('best.pt')
        model = YOLO('best.pt')

        # Menjalankan model pada video yang disediakan dengan opsi yang diatur
        for result in model(source= src, show=True, save=True, stream=True, show_boxes=True, optimize=True, save_txt=False, verbose=False):
            
            # Cek apakah model mendeteksi sesuatu
            if result:
                logging.info("Fall Detected")
            else:
                logging.info("Normal")
    
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    object()
