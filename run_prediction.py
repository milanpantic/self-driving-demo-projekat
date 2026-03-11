import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import csv

MODEL_PATH = "best_model.h5"
CSV_PATH = "complete_dataset/merged/driving_log_merged.csv"
OUTPUT_VIDEO_NAME = "rezultat_voznje_final.mp4"
OUTPUT_CSV_NAME = "rezultati_predikcije.csv"
SEQ_LENGTH = 4
IMG_HEIGHT, IMG_WIDTH = 66, 200
def preprocess_img(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cropped = img_rgb[60:135, :, :] 
    img_resized = cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT))
    return img_resized / 255.0

if not os.path.exists(MODEL_PATH):
    print(f"Greška: Model {MODEL_PATH} nije pronađen!")
    exit()

print("Učitavam model i podatke...")
model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH, header=None)
image_paths = df[0].values
actual_angles = df[3].values

steering_history = []
LANE_CHANGE_THRESHOLD = 0.10 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = None

csv_file = open(OUTPUT_CSV_NAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_path', 'actual_steering_angle', 'predicted_steering_angle', 'lane_change_detected'])

print(f"Pokrećem obradu. Rezultati idu u {OUTPUT_VIDEO_NAME} i {OUTPUT_CSV_NAME}")

limit = min(1200, len(image_paths) - SEQ_LENGTH)

for i in range(limit):
    current_seq_paths = image_paths[i : i + SEQ_LENGTH]
    processed_frames = []
    
    for p in current_seq_paths:
        img = cv2.imread(p)
        if img is not None:
            processed_frames.append(preprocess_img(img))
    
    if len(processed_frames) < SEQ_LENGTH:
        continue

    input_data = np.expand_dims(np.array(processed_frames), axis=0)
    pred_angle = model.predict(input_data, verbose=0)[0][0]

    steering_history.append(pred_angle)
    if len(steering_history) > 15:
        steering_history.pop(0)
    
    is_lane_changing = False
    if len(steering_history) >= 5:
        diff = np.abs(np.diff(steering_history)).mean()
        if diff > LANE_CHANGE_THRESHOLD:
            is_lane_changing = True

    actual_angle = actual_angles[i + SEQ_LENGTH - 1]
    last_img_path = current_seq_paths[-1]
    csv_writer.writerow([last_img_path, actual_angle, pred_angle, int(is_lane_changing)])

    display_frame = cv2.imread(last_img_path)
    h, w, _ = display_frame.shape

    if out_video is None:
        out_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 20.0, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.4
    f_thick = 1

    cv2.rectangle(display_frame, (5, 5), (150, 50), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Predikcija: {pred_angle:.2f}", (10, 20), 
                font, f_scale, (0, 255, 0), f_thick)
    cv2.putText(display_frame, f"Stvarni: {actual_angle:.2f}", (10, 40), 
                font, f_scale, (255, 255, 255), f_thick)

    if is_lane_changing:
        cv2.rectangle(display_frame, (w-140, 5), (w-5, 30), (0, 0, 255), -1)
        cv2.putText(display_frame, "PROMENA TRAKE", (w-130, 22), 
                    font, 0.4, (255, 255, 255), 1)

    out_video.write(display_frame)
    
    if i % 100 == 0:
        print(f"Napredak: {i}/{limit} frejmova...")

if out_video is not None:
    out_video.release()
csv_file.close()

print(f"Gotovo! Snimak: {OUTPUT_VIDEO_NAME}, Podaci: {OUTPUT_CSV_NAME}")