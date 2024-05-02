import logging
from ultralytics import YOLO

# Konfigurasi pengaturan logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def object_detection(src):
    model = 'model/600_epoch/best.pt'
    try:
        # Inisialisasi model YOLO dengan menggunakan model terbaik ('best.pt')
        model = YOLO(model)

        # Menjalankan model pada video yang disediakan dengan opsi yang diatur
        frame_video = model(
            source= src,
            show=False, 
            save=True,
            stream=True, 
            show_boxes=True,
            optimize=True,
            save_txt=False, 
            verbose=False
        )
        for result in frame_video:
            if result:
                logging.info("Fall Detected")
                return True
            else:
                logging.info("Not Fall Detected")
                return False
    except Exception as e:
        logging.error(f"Error: {e}")
        return None