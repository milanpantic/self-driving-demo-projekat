import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# ====== POSTAVKE ======
MODEL_PATH = "best_model.h5"
CSV_PATH = "complete_dataset/merged/driving_log_merged.csv"
OUTPUT_VIDEO_NAME = "rezultat_voznje_final.mp4"
SEQ_LENGTH = 5
IMG_HEIGHT, IMG_WIDTH = 66, 200

# ====== POMOĆNE FUNKCIJE ======
def preprocess_img(img):
    """Obrada slike usklađena sa treningom."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cropped = img_rgb[60:135, :, :] 
    img_resized = cv2.resize(img_cropped, (IMG_WIDTH, IMG_HEIGHT))
    return img_resized / 255.0

# Provera modela
if not os.path.exists(MODEL_PATH):
    print(f"Greška: Model {MODEL_PATH} nije pronađen!")
    exit()

# Učitavanje modela i podataka
print("Učitavam model i podatke...")
model = load_model(MODEL_PATH)
df = pd.read_csv(CSV_PATH, header=None)
image_paths = df[0].values
actual_angles = df[3].values

# Parametri za detekciju promene trake
steering_history = []
LANE_CHANGE_THRESHOLD = 0.10 

# Video Writer inicijalizacija
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

print(f"Pokrećem obradu. Video će biti sačuvan kao: {OUTPUT_VIDEO_NAME}")

# Obrađujemo npr. 1200 frejmova
limit = min(1200, len(image_paths) - SEQ_LENGTH)

for i in range(limit):
    # 1. Priprema sekvence (CNN+LSTM zahteva niz slika)
    current_seq_paths = image_paths[i : i + SEQ_LENGTH]
    processed_frames = []
    
    for p in current_seq_paths:
        img = cv2.imread(p)
        if img is not None:
            processed_frames.append(preprocess_img(img))
    
    if len(processed_frames) < SEQ_LENGTH:
        continue

    # 2. Predikcija ugla upravljanja
    input_data = np.expand_dims(np.array(processed_frames), axis=0)
    pred_angle = model.predict(input_data, verbose=0)[0][0]
    
    # 3. Logika za detekciju promene trake (Lane Change)
    steering_history.append(pred_angle)
    if len(steering_history) > 15:
        steering_history.pop(0)
    
    is_lane_changing = False
    if len(steering_history) >= 5:
        diff = np.abs(np.diff(steering_history)).mean()
        if diff > LANE_CHANGE_THRESHOLD:
            is_lane_changing = True

    # 4. Kreiranje frejma za video
    display_frame = cv2.imread(current_seq_paths[-1])
    h, w, _ = display_frame.shape

    if out is None:
        out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 20.0, (w, h))

    # --- VIZUELIZACIJA (EXTRA SMALL FONT) ---
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.4  # Smanjen font sa 0.5 na 0.4
    f_thick = 1

    # GORE LEVO: Telemetrija (Kompaktniji crni pravougaonik)
    cv2.rectangle(display_frame, (5, 5), (150, 50), (0, 0, 0), -1)
    
    cv2.putText(display_frame, f"Predikcija: {pred_angle:.2f}", (10, 20), 
                font, f_scale, (0, 255, 0), f_thick)
    cv2.putText(display_frame, f"Stvarni: {actual_angles[i+SEQ_LENGTH-1]:.2f}", (10, 40), 
                font, f_scale, (255, 255, 255), f_thick)

    # GORE DESNO: Upozorenje za promenu trake (Manji indikator)
    if is_lane_changing:
        # Manji crveni panel u samom uglu
        cv2.rectangle(display_frame, (w-140, 5), (w-5, 30), (0, 0, 255), -1)
        cv2.putText(display_frame, "PROMENA TRAKE", (w-130, 22), 
                    font, 0.4, (255, 255, 255), 1)

    # Upisivanje u video fajl
    out.write(display_frame)
    
    if i % 100 == 0:
        print(f"Napredak: {i}/{limit} frejmova...")

# Zatvaranje resursa
if out is not None:
    out.release()
print(f"Gotovo! Snimak je spreman: {OUTPUT_VIDEO_NAME}")