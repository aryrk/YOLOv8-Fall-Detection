from ultralytics import YOLO
import torch
import cv2


# Initializations
X = 0  # Koordinat X untuk lebar dimensi get_size_bounding_box
Y = 1  # Koordinat Y untuk tinggi dimensi dalam get_size_bounding_box
# Format warna BGR
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_PINK = (255, 0, 255)
source = 'video//Sequence 01.mp4'

# Explain
# 1. Load the model from the ultralytics
model = YOLO('yolov8n-pose.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# mendapatkan ukuran bounding box
def get_size_bounding_box(box_tensor):
    # jika ada bounding box
    if len(box_tensor.xyxy) > 0:
        # mengambil bounding box pertama
        box_tensor = box_tensor.xyxy[0]

        # jika bounding box pertama ada
        if len(box_tensor) > 0:

            # mengambil nilai bounding box dari tensor ke list
            box_list = box_tensor.tolist()

            # mengambil nilai x_min, y_min, x_max, y_max
            x_min = box_list[0]
            y_min = box_list[1]
            x_max = box_list[2]
            y_max = box_list[3]

            # menghitung lebar bounding box
            widht = x_max - x_min
            # menghitung tinggi bounding box
            height = y_max - y_min

            # (lebar, tinggi)
            return (widht, height)
        else:
            return (0, 0)
    else:
        return (0, 0)

# Mendapatkan kondisi jatuh berdasarkan bounding box
def process_bounding_box(dimension):
    '''
    Cara penggunaan:
        process_bounding_box(result.boxes):
    '''

    if dimension[X] > 0 and dimension[Y] > 0:
        if dimension[X] > dimension[Y]:
            return True
        else:
            return False
    else:
        return None

# Mendapatkan lokasi leher dari keypoints
def neck_location(keypoints_xy):
    neck_coord_X6, neck_coord_Y6 = keypoints_xy[0][6]
    neck_coord_X5, neck_coord_Y5 = keypoints_xy[0][5]
    return ((neck_coord_X5 + neck_coord_X6) // 2,
            (neck_coord_Y5 + neck_coord_Y6) // 2)

# Mendapatkan lokasi pinggang dari keypoints
def waist_location(keypoints_xy):
    waist_coord_X11, waist_coord_Y11 = keypoints_xy[0][11]
    waist_coord_X12, waist_coord_Y12 = keypoints_xy[0][12]
    return ((waist_coord_X11 + waist_coord_X12) // 2,
            (waist_coord_Y11 + waist_coord_Y12) // 2)

# Proses koordinat leher dan pinggang
def process_coordinates(neck, waist, video_height, range):
    # leher dan pinggang adalah touple koordinat leher dan pinggang
    # range adalah ukuran tambahan sebagai identifikasi jarak leher dan pinggang
    # range adalah jarak tambahan sebagai identifikasi jarak leher dan pinggang
    if neck[Y] >= waist[Y]-range and neck[Y] <= waist[Y]+range and neck[Y]:
        return True
    return False

# ====================================================
def count_fps(result):
    fps = 0
    try:
        fps = 1000//(result.speed['preprocess'] +
                     result.speed['inference']+result.speed['postprocess'])
    except:
        pass
    return fps

# Print information
def put_information(image, neck, waist, bounding_dimension, fps, video_height, color):
    cv2.circle(image, (int(neck[X]), int(neck[Y])), 10, color, -1)
    cv2.putText(image, f'neck({neck[X]},{neck[Y]})', (int(neck[X]) + 20, int(neck[Y]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)

    cv2.circle(image, (int(waist[X]), int(waist[Y])), 10, color, -1)
    cv2.putText(image, f'waist({waist[X]}, {waist[Y]})', (int(waist[X]) + 20, int(waist[Y]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 2)
    cv2.putText(image, f'FPS: {fps}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(image, f'NeckY: {neck[1]}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, f"NeckX: {neck[0]}'", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(image, f"PinggangY: {waist[Y]} + {bounding_dimension[Y]//4} = {waist[Y]+bounding_dimension[Y]//4}, - {bounding_dimension[Y]//4} = {waist[Y]-bounding_dimension[Y]//4}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, f"PinggangX: {waist[X]}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, f'Height: {video_height}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.circle(image, (int(waist[X]), int(
        waist[Y]+bounding_dimension[Y]//4)), 10, COLOR_BLACK, -1)
    cv2.circle(image, (int(waist[X]), int(
        waist[Y]-bounding_dimension[Y]//4)), 10, COLOR_YELLOW, -1)


# Main function
def main():
    frame_video = model(
        source=source, 
        show=False, 
        save=False, 
        stream=True, 
        show_boxes=False, 
        optimize=True,
        save_txt=False, 
        half=True, 
        max_det=1, 
        stream_buffer=True, 
        show_labels=False, 
        show_conf=False, 
        verbose=False
    )
    for result in frame_video:
        # mengambil keypoints dari hasil deteksi
        keypoints = result.keypoints

        # mengambil gambar dari hasil deteksi
        image = result.orig_img.copy()
        video_height, video_width, _ = image.shape

        # warna dari setiap titik
        color = COLOR_BLUE

        # koordinat dari semua titik badan
        keypoints_xy = keypoints.xy

        # dapatkan dimens bounding box dari orang
        bounding_dimension = get_size_bounding_box(result.boxes)

        # mendapatkan nilai fps
        fps = count_fps(result)

        #
        for i, point in enumerate(keypoints_xy[0]):
            if i == 5 or i == 11:
                # Cari lokasi Neck
                neck = neck_location(keypoints_xy)
                # neck_coord_X6, neck_coord_Y6 = keypoints_xy[0][6]
                # neck_coord_X5, neck_coord_Y5 = keypoints_xy[0][5]
                # neck = ((neck_coord_X5 + neck_coord_X6) // 2,
                #         (neck_coord_Y5 + neck_coord_Y6) // 2)

                # Cari lokasi Pinggang
                waist = waist_location(keypoints_xy)
                # waist_coord_X11, waist_coord_Y11 = keypoints_xy[0][11]
                # waist_coord_X12, waist_coord_Y12 = keypoints_xy[0][12]
                # waist = ((waist_coord_X11 + waist_coord_X12) // 2,
                #          (waist_coord_Y11 + waist_coord_Y12) // 2)

                if process_coordinates(neck, waist, video_height, bounding_dimension[Y]//4) or process_bounding_box(bounding_dimension) == True:
                    color = COLOR_RED
                    print("Fall Detected")

                put_information(image, neck, waist,
                                bounding_dimension, fps, video_height, color)

        # print(neck)

        cv2.imshow('Image with Keypoints', image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


main()
