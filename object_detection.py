import logging
from ultralytics import YOLO
import torch

# Konfigurasi pengaturan logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
object_model = 'model/600_epoch/best.pt'
object_model = YOLO(object_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
half_param = True
if device == 'cuda':
    half_param = False
object_model.to(device)

def object_detection(source):
    try:
        # Inisialisasi object_model YOLO dengan menggunakan object_model terbaik ('best.pt')

        # Menjalankan object_model pada video yang disediakan dengan opsi yang diatur
        frame_video = object_model(
            source= source,
            show=False,
            save=False,
            stream=False,
            show_boxes=False,
            optimize=True,
            save_txt=False,
            half=half_param,
            max_det=1,
            stream_buffer=False,
            show_labels=False,
            show_conf=False,
            verbose=False
        )
        
        for result in frame_video:
            if result:
                # logging.info("Fall Detected")
                return True
            else:
                # logging.info("Not Fall Detected")
                return False
    except Exception as e:
        # logging.error(f"Error: {e}")
        return None