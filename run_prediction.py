import csv
import os

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from plot_prediction_results import create_prediction_plot


MODEL_PATH = "best_model.h5"
CSV_PATH = "complete_dataset/merged/test.csv"
OUTPUT_VIDEO_NAME = "rezultat_voznje_final.mp4"
OUTPUT_CSV_NAME = "rezultati_predikcije.csv"
OUTPUT_PLOT_NAME = "rezultati_predikcije_grafik.png"
VIDEO_SEQUENCE_GROUP = "jungle_test_contiguous"
VIDEO_FPS = 14.5
SEQ_LENGTH = 4
SEQUENCE_GROUP_COL = 7
IMG_HEIGHT = 66
IMG_WIDTH = 200
LANE_CHANGE_THRESHOLD = 0.10


def valid_sequence_starts(sequence_groups, seq_length):
    starts = np.arange(len(sequence_groups) - seq_length + 1)
    valid = np.ones(len(starts), dtype=bool)

    for offset in range(1, seq_length):
        valid = valid & (
            sequence_groups[starts] == sequence_groups[starts + offset]
        )

    return starts[valid]


def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:135]
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image / 255.0


def load_data():
    df = pd.read_csv(CSV_PATH, header=None)
    if SEQUENCE_GROUP_COL not in df.columns:
        raise ValueError(
            f"{CSV_PATH} nema ID vremenskog segmenta u koloni "
            f"{SEQUENCE_GROUP_COL}. Prvo pokreni data_inscpection.py."
        )

    image_paths = df[0].values
    actual_angles = df[3].values
    sequence_groups = df[SEQUENCE_GROUP_COL].astype(str).values
    return df, image_paths, actual_angles, sequence_groups


def select_video_sequences(df, sequence_groups):
    starts = valid_sequence_starts(sequence_groups, SEQ_LENGTH)
    group_sizes = df.groupby(SEQUENCE_GROUP_COL).size()
    video_group = VIDEO_SEQUENCE_GROUP or group_sizes.idxmax()

    if video_group not in group_sizes.index:
        raise ValueError(
            f"Video segment {video_group} ne postoji. "
            f"Dostupni: {list(group_sizes.index)}"
        )

    selected_starts = starts[sequence_groups[starts] == video_group]
    return selected_starts, video_group


def predict_sequence(model, sequence_paths):
    processed_frames = []

    for image_path in sequence_paths:
        image = cv2.imread(image_path)
        if image is None:
            return None
        processed_frames.append(preprocess_img(image))

    input_data = np.expand_dims(np.asarray(processed_frames), axis=0)
    return model.predict(input_data, verbose=0)[0][0]


def draw_result(frame, predicted_angle, actual_angle, lane_change):
    _, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (5, 5), (150, 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Predikcija: {predicted_angle:.2f}",
        (10, 20),
        font,
        0.4,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        frame,
        f"Stvarni: {actual_angle:.2f}",
        (10, 40),
        font,
        0.4,
        (255, 255, 255),
        1,
    )

    if lane_change:
        cv2.rectangle(frame, (width - 140, 5), (width - 5, 30), (0, 0, 255), -1)
        cv2.putText(
            frame,
            "PROMENA TRAKE",
            (width - 130, 22),
            font,
            0.4,
            (255, 255, 255),
            1,
        )

    return frame


def create_video_and_csv(
    model,
    image_paths,
    actual_angles,
    sequence_groups,
    sequence_starts,
):
    steering_history = []
    previous_group = None
    out_video = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with open(OUTPUT_CSV_NAME, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_index",
            "sequence_group",
            "image_path",
            "actual_steering_angle",
            "predicted_steering_angle",
            "error",
            "absolute_error",
            "lane_change_detected",
        ])

        for sequence_number, start in enumerate(sequence_starts):
            current_group = sequence_groups[start]
            if current_group != previous_group:
                steering_history.clear()
                previous_group = current_group

            sequence_paths = image_paths[start:start + SEQ_LENGTH]
            predicted_angle = predict_sequence(model, sequence_paths)
            if predicted_angle is None:
                continue

            steering_history.append(predicted_angle)
            if len(steering_history) > 15:
                steering_history.pop(0)

            lane_change = False
            if len(steering_history) >= 5:
                average_change = np.abs(np.diff(steering_history)).mean()
                if average_change > LANE_CHANGE_THRESHOLD:
                    lane_change = True

            frame_index = start + SEQ_LENGTH - 1
            actual_angle = actual_angles[frame_index]
            last_image_path = sequence_paths[-1]
            error = predicted_angle - actual_angle

            csv_writer.writerow([
                frame_index,
                current_group,
                last_image_path,
                float(actual_angle),
                float(predicted_angle),
                float(error),
                float(abs(error)),
                int(lane_change),
            ])

            display_frame = cv2.imread(last_image_path)
            display_frame = draw_result(
                display_frame,
                predicted_angle,
                actual_angle,
                lane_change,
            )

            if out_video is None:
                height, width, _ = display_frame.shape
                out_video = cv2.VideoWriter(
                    OUTPUT_VIDEO_NAME,
                    fourcc,
                    VIDEO_FPS,
                    (width, height),
                )

            out_video.write(display_frame)

            if sequence_number % 100 == 0:
                print(
                    f"Napredak: {sequence_number}/"
                    f"{len(sequence_starts)} frejmova..."
                )

    if out_video is not None:
        out_video.release()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Greška: Model {MODEL_PATH} nije pronađen!")
        return

    print("Učitavam model i podatke...")
    model = load_model(MODEL_PATH)
    df, image_paths, actual_angles, sequence_groups = load_data()

    print(
        f"Pokrećem obradu. Rezultati idu u {OUTPUT_VIDEO_NAME} "
        f"i {OUTPUT_CSV_NAME}"
    )
    sequence_starts, video_group = select_video_sequences(df, sequence_groups)
    print(
        f"Video koristi kontinuirani segment {video_group}: "
        f"{len(sequence_starts)} izlaznih frejmova"
    )

    create_video_and_csv(
        model,
        image_paths,
        actual_angles,
        sequence_groups,
        sequence_starts,
    )
    create_prediction_plot(OUTPUT_CSV_NAME, OUTPUT_PLOT_NAME, fps=VIDEO_FPS)

    print(
        f"Gotovo! Snimak: {OUTPUT_VIDEO_NAME}, Podaci: {OUTPUT_CSV_NAME}, "
        f"Grafikon: {OUTPUT_PLOT_NAME}"
    )


if __name__ == "__main__":
    main()
