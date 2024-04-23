from ultralytics import YOLO
import torch
import cv2

model = YOLO('yolov8n-pose.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for result in model(source='video\\15 Minute Intense HIIT Workout For Fat Burn  Cardio No Equipment No Repeat Home Workout_360p.mp4', show=False, save=False, stream=True, show_boxes=False, optimize=True, save_txt=False, half=True, max_det=1, stream_buffer=True, show_labels=False, show_conf=False, verbose=False):
    keypoints = result.keypoints
    image = result.orig_img.copy()

    keypoints_xy = keypoints.xy

    neck = (0, 0)
    fps = 0
    try:
        fps = 1000//(result.speed['preprocess'] +
                     result.speed['inference']+result.speed['postprocess'])
    except:
        pass

    for i, point in enumerate(keypoints_xy[0]):
        x, y = point

        if i == 5:
            x6, y6 = keypoints_xy[0][6]
            x5, y5 = keypoints_xy[0][5]
            x_half = (x5 + x6) // 2
            y_half = (y5 + y6) // 2
            neck = (x_half, y_half)
            cv2.circle(image, (int(x_half), int(y_half)), 5, (0, 255, 0), -1)
            cv2.putText(image, '5-6', (int(x_half) + 10, int(y_half) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(image, f'FPS: {fps}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # print(neck)

    cv2.imshow('Image with Keypoints', image)
    cv2.waitKey(1)

cv2.destroyAllWindows()
