import math

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


TRAIN_CSV_PATH = "complete_dataset/merged/train.csv"
VAL_CSV_PATH = "complete_dataset/merged/val.csv"
TEST_CSV_PATH = "complete_dataset/merged/test.csv"
BATCH_SIZE = 32
EPOCHS = 20
IMG_HEIGHT = 66
IMG_WIDTH = 200
SEQ_LENGTH = 4
SEQUENCE_GROUP_COL = 7
SEED = 42


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Slika nije pronađena: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:135, :, :]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32) / 255.0


def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if SEQUENCE_GROUP_COL not in df.columns:
        raise ValueError(
            f"{csv_path} nema ID vremenskog segmenta u koloni {SEQUENCE_GROUP_COL}. "
            "Prvo pokreni data_inscpection.py."
        )
    return (
        df[0].astype(str).to_numpy(),
        df[3].to_numpy(dtype=np.float32),
        df[SEQUENCE_GROUP_COL].astype(str).to_numpy(),
    )


def valid_sequence_starts(sequence_groups, seq_length):
    """Return starts whose complete sequence belongs to one chronological segment."""
    starts = np.arange(len(sequence_groups) - seq_length + 1)
    if len(starts) == 0:
        return starts

    valid = np.ones(len(starts), dtype=bool)
    for offset in range(1, seq_length):
        valid &= sequence_groups[starts] == sequence_groups[starts + offset]
    return starts[valid]


def balanced_sequence_starts(valid_starts, steering, seq_length, rng):
    """Oversample left/straight/right targets equally for one training epoch."""
    target_angles = steering[valid_starts + seq_length - 1]
    classes = [
        valid_starts[target_angles < 0],
        valid_starts[target_angles == 0],
        valid_starts[target_angles > 0],
    ]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("Trening skup mora sadržati leve, prave i desne sekvence")

    samples_per_class = max(map(len, classes))
    balanced = np.concatenate([
        rng.choice(indices, size=samples_per_class, replace=len(indices) < samples_per_class)
        for indices in classes
    ])
    rng.shuffle(balanced)
    return balanced


def augment_sequence(sequence, angle, rng):
    """Apply identical geometric/light augmentation to every frame in a sequence."""
    sequence = np.asarray(sequence, dtype=np.float32)

    if rng.random() < 0.5:
        sequence = sequence[:, :, ::-1, :]
        angle = -angle

    if rng.random() < 0.5:
        brightness = rng.uniform(0.6, 1.4)
        sequence = np.clip(sequence * brightness, 0.0, 1.0)

    if rng.random() < 0.5:
        x1, x2 = sorted(rng.integers(0, IMG_WIDTH, size=2))
        if x1 != x2:
            shadow = rng.uniform(0.4, 0.7)
            sequence[:, :, x1:x2, :] *= shadow

    return sequence, angle


def sequence_generator(
    image_paths,
    steering,
    sequence_groups,
    batch_size,
    seq_length,
    shuffle=True,
    augment=False,
    balance=False,
    seed=SEED,
):
    valid_starts = valid_sequence_starts(sequence_groups, seq_length)
    if len(valid_starts) == 0:
        raise ValueError("Nema validnih vremenskih sekvenci")
    rng = np.random.default_rng(seed)

    while True:
        if balance:
            epoch_starts = balanced_sequence_starts(valid_starts, steering, seq_length, rng)
        else:
            epoch_starts = valid_starts.copy()
            if shuffle:
                rng.shuffle(epoch_starts)

        for batch_start in range(0, len(epoch_starts), batch_size):
            batch_indices = epoch_starts[batch_start:batch_start + batch_size]
            x_batch, y_batch = [], []

            for idx in batch_indices:
                sequence = [load_image(image_paths[idx + offset]) for offset in range(seq_length)]
                angle = steering[idx + seq_length - 1]
                if augment:
                    sequence, angle = augment_sequence(sequence, angle, rng)
                x_batch.append(sequence)
                y_batch.append(angle)

            yield np.asarray(x_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)


def epoch_sequence_count(steering, sequence_groups, seq_length, balance=False):
    starts = valid_sequence_starts(sequence_groups, seq_length)
    if not balance:
        return len(starts)
    target_angles = steering[starts + seq_length - 1]
    class_counts = [(target_angles < 0).sum(), (target_angles == 0).sum(), (target_angles > 0).sum()]
    if min(class_counts) == 0:
        raise ValueError("Trening skup mora sadržati leve, prave i desne sekvence")
    return 3 * max(class_counts)


def build_basic_cnn_lstm(seq_length):
    input_layer = Input(shape=(seq_length, IMG_HEIGHT, IMG_WIDTH, 3))
    x = TimeDistributed(Conv2D(16, (5, 5), strides=(2, 2), activation="relu"))(input_layer)
    x = TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="tanh")(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def evaluate_predictions(model, generator, steering, sequence_groups, seq_length, batch_size):
    starts = valid_sequence_starts(sequence_groups, seq_length)
    actual = steering[starts + seq_length - 1]
    predicted = model.predict(
        generator,
        steps=math.ceil(len(starts) / batch_size),
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

    mean_baseline = np.full_like(actual, actual.mean())
    previous_angle_baseline = steering[starts + seq_length - 2]
    print(f"Baseline nula MAE: {np.mean(np.abs(actual)):.4f}")
    print(f"Baseline prosek MAE: {np.mean(np.abs(mean_baseline - actual)):.4f}")
    print(
        "Baseline prethodni ugao MAE: "
        f"{np.mean(np.abs(previous_angle_baseline - actual)):.4f}"
    )


def main():
    np.random.seed(SEED)
    train_paths, train_steering, train_groups = load_dataset(TRAIN_CSV_PATH)
    val_paths, val_steering, val_groups = load_dataset(VAL_CSV_PATH)
    test_paths, test_steering, test_groups = load_dataset(TEST_CSV_PATH)

    train_count = epoch_sequence_count(train_steering, train_groups, SEQ_LENGTH, balance=True)
    val_count = epoch_sequence_count(val_steering, val_groups, SEQ_LENGTH)
    test_count = epoch_sequence_count(test_steering, test_groups, SEQ_LENGTH)
    print(
        f"Balansiranih trening sekvenci po epohi: {train_count}, "
        f"validacionih: {val_count}, test: {test_count}"
    )

    train_gen = sequence_generator(
        train_paths, train_steering, train_groups, BATCH_SIZE, SEQ_LENGTH,
        shuffle=True, augment=True, balance=True, seed=SEED,
    )
    val_gen = sequence_generator(
        val_paths, val_steering, val_groups, BATCH_SIZE, SEQ_LENGTH,
        shuffle=False, augment=False, balance=False, seed=SEED,
    )
    test_gen = sequence_generator(
        test_paths, test_steering, test_groups, BATCH_SIZE, SEQ_LENGTH,
        shuffle=False, augment=False, balance=False, seed=SEED,
    )

    model = build_basic_cnn_lstm(SEQ_LENGTH)
    checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        train_gen,
        steps_per_epoch=math.ceil(train_count / BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=math.ceil(val_count / BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
    )

    evaluate_predictions(
        model,
        test_gen,
        test_steering,
        test_groups,
        SEQ_LENGTH,
        BATCH_SIZE,
    )

    model.save("final_cnn_lstm_model.h5")
    print("Trening završen.")


if __name__ == "__main__":
    main()
