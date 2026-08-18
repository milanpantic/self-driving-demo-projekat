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
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    TimeDistributed,
)
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
        raise ValueError(f"{csv_path} nema ID vremenskog segmenta")
    return df


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Slika nije pronađena: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[60:135]
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image.astype(np.float32) / 255.0


def find_valid_sequence_starts(df):
    """Sekvenca je validna samo ako sva 4 frejma pripadaju istoj grupi."""
    groups = df[GROUP_COL].astype(str).to_numpy()
    starts = np.arange(len(df) - SEQ_LENGTH + 1)
    valid = np.ones(len(starts), dtype=bool)

    for offset in range(1, SEQ_LENGTH):
        valid = valid & (groups[starts] == groups[starts + offset])

    valid_starts = starts[valid]
    if len(valid_starts) == 0:
        raise ValueError("Dataset nema validnih vremenskih sekvenci")
    return valid_starts


def balance_training_sequences(df, valid_starts, rng):
    """Uzorkuje isti broj levih, pravih i desnih sekvenci."""
    angles = df[ANGLE_COL].to_numpy(dtype=np.float32)
    targets = angles[valid_starts + SEQ_LENGTH - 1]

    left = valid_starts[targets < 0]
    straight = valid_starts[targets == 0]
    right = valid_starts[targets > 0]

    if len(left) == 0 or len(straight) == 0 or len(right) == 0:
        raise ValueError("Trening skup mora sadržati leve, prave i desne sekvence")

    number_per_class = max(len(left), len(straight), len(right))
    selected_left = rng.choice(left, number_per_class, replace=len(left) < number_per_class)
    selected_straight = rng.choice(
        straight,
        number_per_class,
        replace=len(straight) < number_per_class,
    )
    selected_right = rng.choice(
        right,
        number_per_class,
        replace=len(right) < number_per_class,
    )

    balanced_starts = np.concatenate([
        selected_left,
        selected_straight,
        selected_right,
    ])
    rng.shuffle(balanced_starts)
    return balanced_starts


def augment_sequence(sequence, angle, rng):
    """Iste izmene primenjuje na sva četiri frejma."""
    sequence = np.asarray(sequence, dtype=np.float32)

    if rng.random() < 0.5:
        sequence = sequence[:, :, ::-1]
        angle = -angle

    if rng.random() < 0.5:
        brightness = rng.uniform(0.6, 1.4)
        sequence = np.clip(sequence * brightness, 0.0, 1.0)

    if rng.random() < 0.5:
        x1, x2 = sorted(rng.integers(0, IMG_WIDTH, size=2))
        shadow = rng.uniform(0.4, 0.7)
        sequence[:, :, x1:x2] *= shadow

    return sequence, angle


def sequence_generator(df, training=False):
    image_paths = df[0].astype(str).to_numpy()
    angles = df[ANGLE_COL].to_numpy(dtype=np.float32)
    valid_starts = find_valid_sequence_starts(df)
    rng = np.random.default_rng(SEED)

    while True:
        if training:
            epoch_starts = balance_training_sequences(df, valid_starts, rng)
        else:
            epoch_starts = valid_starts

        for batch_start in range(0, len(epoch_starts), BATCH_SIZE):
            batch_indices = epoch_starts[batch_start:batch_start + BATCH_SIZE]
            x_batch = []
            y_batch = []

            for start in batch_indices:
                sequence = []
                for offset in range(SEQ_LENGTH):
                    sequence.append(load_image(image_paths[start + offset]))

                angle = angles[start + SEQ_LENGTH - 1]
                if training:
                    sequence, angle = augment_sequence(sequence, angle, rng)

                x_batch.append(sequence)
                y_batch.append(angle)

            yield np.asarray(x_batch), np.asarray(y_batch)


def count_sequences(df, balanced=False):
    valid_starts = find_valid_sequence_starts(df)
    if not balanced:
        return len(valid_starts)

    angles = df[ANGLE_COL].to_numpy()[valid_starts + SEQ_LENGTH - 1]
    left_count = (angles < 0).sum()
    straight_count = (angles == 0).sum()
    right_count = (angles > 0).sum()
    return 3 * max(left_count, straight_count, right_count)


def build_model():
    inputs = Input(shape=(SEQ_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3))
    x = TimeDistributed(Conv2D(16, (5, 5), strides=(2, 2), activation="relu"))(inputs)
    x = TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="tanh")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def evaluate_model(model, test_df):
    valid_starts = find_valid_sequence_starts(test_df)
    angles = test_df[ANGLE_COL].to_numpy(dtype=np.float32)
    actual = angles[valid_starts + SEQ_LENGTH - 1]

    predicted = model.predict(
        sequence_generator(test_df),
        steps=math.ceil(len(valid_starts) / BATCH_SIZE),
        verbose=1,
    )
    predicted = predicted.reshape(-1)[:len(actual)]
    errors = predicted - actual

    print(f"Test MSE: {np.mean(errors ** 2):.4f}")
    print(f"Test MAE: {np.mean(np.abs(errors)):.4f}")

    directions = [
        ("levo", actual < 0),
        ("pravo", actual == 0),
        ("desno", actual > 0),
    ]
    for name, mask in directions:
        direction_mae = np.mean(np.abs(errors[mask]))
        print(f"MAE {name} ({mask.sum()} sekvenci): {direction_mae:.4f}")

    previous = angles[valid_starts + SEQ_LENGTH - 2]
    print(f"Baseline nula MAE: {np.mean(np.abs(actual)):.4f}")
    print(f"Baseline prosek MAE: {np.mean(np.abs(actual - actual.mean())):.4f}")
    print(f"Baseline prethodni ugao MAE: {np.mean(np.abs(actual - previous)):.4f}")


def main():
    np.random.seed(SEED)
    device_name = "GPU" if USE_GPU else "CPU"
    print(f"Trening se pokreće na: {device_name}")

    train_df = load_dataset(TRAIN_CSV)
    val_df = load_dataset(VAL_CSV)
    test_df = load_dataset(TEST_CSV)

    train_count = count_sequences(train_df, balanced=True)
    val_count = count_sequences(val_df)
    test_count = count_sequences(test_df)
    print(
        f"Balansiranih trening sekvenci po epohi: {train_count}, "
        f"validacionih: {val_count}, test: {test_count}"
    )

    model = build_model()
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        save_best_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    start_time = time.perf_counter()
    history = model.fit(
        sequence_generator(train_df, training=True),
        steps_per_epoch=math.ceil(train_count / BATCH_SIZE),
        validation_data=sequence_generator(val_df),
        validation_steps=math.ceil(val_count / BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping],
    )
    duration = time.perf_counter() - start_time

    print(f"Trajanje treninga ({device_name}): {duration:.1f} s ({duration / 60:.2f} min)")
    print(f"Prosečno po epohi: {duration / len(history.history['loss']):.1f} s")

    best_model = load_model("best_model.h5")
    evaluate_model(best_model, test_df)
    best_model.save("final_cnn_lstm_model.h5")
    print("Trening završen.")


if __name__ == "__main__":
    main()
