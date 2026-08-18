import math
import os
import time


USE_GPU = True
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


TRAIN_CSV = "complete_dataset/merged/train.csv"
VAL_CSV = "complete_dataset/merged/val.csv"
TEST_CSV = "complete_dataset/merged/test.csv"
BATCH_SIZE = 32
EPOCHS = 20
IMG_HEIGHT = 66
IMG_WIDTH = 200
SEQ_LENGTH = 4
ANGLE_COL = 3
GROUP_COL = 7
SEED = 42


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if GROUP_COL not in df.columns:
        raise ValueError(f"{csv_path} nema ID vremenskog segmenta u koloni {GROUP_COL}")
    return df


def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Slika nije pronađena: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:135]
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image.astype(np.float32) / 255.0


def valid_sequence_starts(df):
    groups = df[GROUP_COL].astype(str).to_numpy()
    starts = np.arange(len(df) - SEQ_LENGTH + 1)
    valid = np.ones(len(starts), dtype=bool)

    for offset in range(1, SEQ_LENGTH):
        valid &= groups[starts] == groups[starts + offset]
    starts = starts[valid]
    if len(starts) == 0:
        raise ValueError("Dataset nema validnih vremenskih sekvenci")
    return starts


def balance_sequence_starts(df, starts, rng):
    angles = df[ANGLE_COL].to_numpy(dtype=np.float32)
    targets = angles[starts + SEQ_LENGTH - 1]
    classes = [
        starts[targets < 0],
        starts[targets == 0],
        starts[targets > 0],
    ]
    if any(len(group) == 0 for group in classes):
        raise ValueError("Trening skup mora sadržati leve, prave i desne sekvence")

    samples_per_class = max(map(len, classes))
    balanced = np.concatenate([
        rng.choice(group, samples_per_class, replace=len(group) < samples_per_class)
        for group in classes
    ])
    rng.shuffle(balanced)
    return balanced


def augment_sequence(sequence, angle, rng):
    sequence = np.asarray(sequence, dtype=np.float32)

    if rng.random() < 0.5:
        sequence = sequence[:, :, ::-1]
        angle = -angle

    if rng.random() < 0.5:
        sequence = np.clip(sequence * rng.uniform(0.6, 1.4), 0.0, 1.0)

    if rng.random() < 0.5:
        x1, x2 = sorted(rng.integers(0, IMG_WIDTH, size=2))
        sequence[:, :, x1:x2] *= rng.uniform(0.4, 0.7)

    return sequence, angle


def sequence_generator(df, training=False):
    paths = df[0].astype(str).to_numpy()
    angles = df[ANGLE_COL].to_numpy(dtype=np.float32)
    starts = valid_sequence_starts(df)
    rng = np.random.default_rng(SEED)

    while True:
        epoch_starts = (
            balance_sequence_starts(df, starts, rng)
            if training
            else starts
        )

        for batch_start in range(0, len(epoch_starts), BATCH_SIZE):
            batch_indices = epoch_starts[batch_start:batch_start + BATCH_SIZE]
            x_batch, y_batch = [], []

            for start in batch_indices:
                sequence = [load_image(paths[start + offset]) for offset in range(SEQ_LENGTH)]
                angle = angles[start + SEQ_LENGTH - 1]
                if training:
                    sequence, angle = augment_sequence(sequence, angle, rng)
                x_batch.append(sequence)
                y_batch.append(angle)

            yield np.asarray(x_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)


def sequence_count(df, balanced=False):
    starts = valid_sequence_starts(df)
    if not balanced:
        return len(starts)

    angles = df[ANGLE_COL].to_numpy()[starts + SEQ_LENGTH - 1]
    class_counts = [(angles < 0).sum(), (angles == 0).sum(), (angles > 0).sum()]
    return 3 * max(class_counts)


def build_model():
    inputs = Input(shape=(SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3))
    x = TimeDistributed(Conv2D(16, 5, strides=2, activation="relu"))(inputs)
    x = TimeDistributed(Conv2D(32, 3, strides=2, activation="relu"))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="tanh")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def evaluate_model(model, test_df):
    starts = valid_sequence_starts(test_df)
    angles = test_df[ANGLE_COL].to_numpy(dtype=np.float32)
    actual = angles[starts + SEQ_LENGTH - 1]
    predicted = model.predict(
        sequence_generator(test_df),
        steps=math.ceil(len(starts) / BATCH_SIZE),
        verbose=1,
    ).reshape(-1)[:len(actual)]

    error = predicted - actual
    print(f"Test MSE: {np.mean(error ** 2):.4f}")
    print(f"Test MAE: {np.mean(np.abs(error)):.4f}")

    for name, mask in (
        ("levo", actual < 0),
        ("pravo", actual == 0),
        ("desno", actual > 0),
    ):
        print(f"MAE {name} ({mask.sum()} sekvenci): {np.mean(np.abs(error[mask])):.4f}")

    previous_angles = angles[starts + SEQ_LENGTH - 2]
    print(f"Baseline nula MAE: {np.mean(np.abs(actual)):.4f}")
    print(f"Baseline prosek MAE: {np.mean(np.abs(actual - actual.mean())):.4f}")
    print(f"Baseline prethodni ugao MAE: {np.mean(np.abs(actual - previous_angles)):.4f}")


def main():
    np.random.seed(SEED)
    device_name = "GPU" if USE_GPU else "CPU"
    print(f"Trening se pokreće na: {device_name}")

    train_df = load_dataset(TRAIN_CSV)
    val_df = load_dataset(VAL_CSV)
    test_df = load_dataset(TEST_CSV)
    train_count = sequence_count(train_df, balanced=True)
    val_count = sequence_count(val_df)

    print(
        f"Balansiranih trening sekvenci po epohi: {train_count}, "
        f"validacionih: {val_count}, test: {sequence_count(test_df)}"
    )

    model = build_model()
    callbacks = [
        ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    started_at = time.perf_counter()
    history = model.fit(
        sequence_generator(train_df, training=True),
        steps_per_epoch=math.ceil(train_count / BATCH_SIZE),
        validation_data=sequence_generator(val_df),
        validation_steps=math.ceil(val_count / BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    duration = time.perf_counter() - started_at

    print(f"Trajanje treninga ({device_name}): {duration:.1f} s ({duration / 60:.2f} min)")
    print(f"Prosečno po epohi: {duration / len(history.history['loss']):.1f} s")

    best_model = load_model("best_model.h5")
    evaluate_model(best_model, test_df)
    best_model.save("final_cnn_lstm_model.h5")
    print("Trening završen.")


if __name__ == "__main__":
    main()
