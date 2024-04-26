from ultralytics import YOLO
import torch
import cv2

model = YOLO('yolov8n-pose.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

fall = 0

def process_coordinates(leher, pinggang, height):
    if leher[1] >= pinggang[1]-30 and leher[1] <= pinggang[1]+30 and leher[1] > height//3:
        return True
    return False


def get_size_bounding_box(box_tensor):
    
    # Memeriksa apakah ada objek yang terdeteksi
    if len(box_tensor.xyxy) > 0:
        # Mengambil tensor dari xyxy
        box_tensor = box_tensor.xyxy[0]

        # Memeriksa apakah tensor tidak kosong
        if len(box_tensor) > 0:
            # Mengkonversi tensor ke dalam list
            box_list = box_tensor.tolist()

            # Mengambil koordinat x_min, y_min, x_max, dan y_max
            x_min = box_list[0]
            y_min = box_list[1]
            x_max = box_list[2]
            y_max = box_list[3]

            # Menghitung lebar dan tinggi bounding box
            widht = x_max - x_min
            height = y_max - y_min

            return widht, height
        else:
            return 0, 0
    else:
        return 0, 0

def falls_based_on_bounding_box(box_tensor):
    '''
    Cara penggunaan:
        falls_based_on_bounding_box(result.boxes):
    '''
    widht, height = get_size_bounding_box(box_tensor)
    if widht > 0 and height > 0:
        if widht > height:
            return True
        else:
            return False
    else:
        return None
                


for result in model(source=2, show=False, save=False, stream=True, show_boxes=False, optimize=True, save_txt=False, half=True, max_det=1, stream_buffer=True, show_labels=False, show_conf=False, verbose=False):
    keypoints = result.keypoints
    image = result.orig_img.copy()
    
    color = (0, 255, 0)
    height, width, _ = image.shape

    keypoints_xy = keypoints.xy

    neck = (0, 0)
    fps = 0
    try:
        fps = 1000//(result.speed['preprocess'] +
                     result.speed['inference']+result.speed['postprocess'])
    except:
        pass

    for i, point in enumerate(keypoints_xy[0]):
        # x, y = point
        # cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
        # cv2.putText(image, str(i), (int(x) + 20, int(y) + 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if i == 5 or i == 11:
            x6, y6 = keypoints_xy[0][6]
            x5, y5 = keypoints_xy[0][5]
            x_half = (x5 + x6) // 2
            y_half = (y5 + y6) // 2
            neck = (x_half, y_half)
            
            x11, y11 = keypoints_xy[0][11]
            x12, y12 = keypoints_xy[0][12]
            x11_half = (x11 + x12) // 2
            y11_half = (y11 + y12) // 2
            pinggang = (x11_half, y11_half)
            
            if process_coordinates(neck, pinggang, height) or falls_based_on_bounding_box(result.boxes) == True:
                color = (0, 0, 255)
                fall+=1
                print("Fall" + str(fall))
            cv2.circle(image, (int(x_half), int(y_half)), 50, color, -1)
            cv2.putText(image, '5-6', (int(x_half) + 20, int(y_half) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.circle(image, (int(x11_half), int(y11_half)), 10, color, -1)
            cv2.putText(image, '11-12', (int(x11_half) + 20, int(y11_half) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.putText(image, f'FPS: {fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f'Neck: {neck[1]}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(image, f'Height: {height}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # print(neck)

    cv2.imshow('Image with Keypoints', image)
    cv2.waitKey(1)

cv2.destroyAllWindows()
