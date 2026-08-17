import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os
import csv

MODEL_PATH = "best_model.h5"
CSV_PATH = "complete_dataset/merged/test.csv"
OUTPUT_VIDEO_NAME = "rezultat_voznje_final.mp4"
OUTPUT_CSV_NAME = "rezultati_predikcije.csv"
VIDEO_SEQUENCE_GROUP = "jungle_test_contiguous"
VIDEO_FPS = 14.5
SEQ_LENGTH = 4
SEQUENCE_GROUP_COL = 7
IMG_HEIGHT, IMG_WIDTH = 66, 200


def valid_sequence_starts(sequence_groups, seq_length):
    starts = np.arange(len(sequence_groups) - seq_length + 1)
    valid = np.ones(len(starts), dtype=bool)
    for offset in range(1, seq_length):
        valid &= sequence_groups[starts] == sequence_groups[starts + offset]
    return starts[valid]


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
if SEQUENCE_GROUP_COL not in df.columns:
    raise ValueError(
        f"{CSV_PATH} nema ID vremenskog segmenta u koloni {SEQUENCE_GROUP_COL}. "
        "Prvo pokreni data_inscpection.py."
    )
image_paths = df[0].values
actual_angles = df[3].values
sequence_groups = df[SEQUENCE_GROUP_COL].astype(str).values

steering_history = []
LANE_CHANGE_THRESHOLD = 0.10 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = None

csv_file = open(OUTPUT_CSV_NAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'frame_index',
    'sequence_group',
    'image_path',
    'actual_steering_angle',
    'predicted_steering_angle',
    'error',
    'absolute_error',
    'lane_change_detected'
])

print(f"Pokrećem obradu. Rezultati idu u {OUTPUT_VIDEO_NAME} i {OUTPUT_CSV_NAME}")

sequence_starts = valid_sequence_starts(sequence_groups, SEQ_LENGTH)
group_sizes = df.groupby(SEQUENCE_GROUP_COL).size()
video_group = VIDEO_SEQUENCE_GROUP or group_sizes.idxmax()
if video_group not in group_sizes.index:
    raise ValueError(
        f"Video segment {video_group} ne postoji. Dostupni: {list(group_sizes.index)}"
    )
sequence_starts = sequence_starts[sequence_groups[sequence_starts] == video_group]
print(
    f"Video koristi kontinuirani segment {video_group}: "
    f"{len(sequence_starts)} izlaznih frejmova"
)
previous_group = None

for sequence_number, i in enumerate(sequence_starts):
    current_group = sequence_groups[i]
    if current_group != previous_group:
        steering_history.clear()
        previous_group = current_group

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
    error = pred_angle - actual_angle
    csv_writer.writerow([
        i + SEQ_LENGTH - 1,
        current_group,
        last_img_path,
        float(actual_angle),
        float(pred_angle),
        float(error),
        float(abs(error)),
        int(is_lane_changing)
    ])

    display_frame = cv2.imread(last_img_path)
    h, w, _ = display_frame.shape

    if out_video is None:
        out_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, VIDEO_FPS, (w, h))

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
    
    if sequence_number % 100 == 0:
        print(f"Napredak: {sequence_number}/{len(sequence_starts)} frejmova...")

if out_video is not None:
    out_video.release()
csv_file.close()

print(f"Gotovo! Snimak: {OUTPUT_VIDEO_NAME}, Podaci: {OUTPUT_CSV_NAME}")
