import cv2
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

CSV_PATH = "complete_dataset/merged/driving_log_merged.csv"
BATCH_SIZE = 32
EPOCHS = 15

IMG_HEIGHT = 66
IMG_WIDTH = 200

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[60:135, :, :]  # crop
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img

def build_cnn():
    model = Sequential()

    model.add(Conv2D(
        24, (5, 5), strides=(2, 2),
        activation='relu',
        input_shape=(66, 200, 3)
    ))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    # 🔹 REGULARIZACIJA POČINJE OVDE
    model.add(Dense(
        100,
        activation='relu',
        kernel_regularizer=l2(1e-4)
    ))
    model.add(Dropout(0.3))

    model.add(Dense(
        50,
        activation='relu',
        kernel_regularizer=l2(1e-4)
    ))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))  # steering angle

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse'
    )

    return model

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    image_paths = df[0].values        # center cam
    steering = df[3].values.astype(np.float32)
    return image_paths, steering

def generator(image_paths, steering, batch_size):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_angles = steering[i:i+batch_size]

            images = []
            angles = []

            for p, a in zip(batch_paths, batch_angles):
                img = load_image(p)
                images.append(img)
                angles.append(a)

            yield np.array(images), np.array(angles)

def main():
    image_paths, steering = load_dataset(CSV_PATH)

    # Podela na train/val/test: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, steering, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_gen = generator(X_train, y_train, BATCH_SIZE)
    val_gen = generator(X_val, y_val, BATCH_SIZE)

    model = build_cnn()
    model.summary()

    checkpoint = ModelCheckpoint(
        "cnn_steering.h5",
        monitor="val_loss",
        save_best_only=True
    )

    model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(X_val) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    np.savez("test_data.npz", X_test=X_test, y_test=y_test)

    evaluate_model()

def evaluate_model(model_path="cnn_steering.h5", test_data_path="test_data.npz"):

    model = load_model(model_path)

    data = np.load(test_data_path, allow_pickle=True)
    X_test_paths = data["X_test"]
    y_test = data["y_test"]

    X_test = np.array([load_image(p) for p in X_test_paths])

    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

    mse = np.mean((y_test - y_pred.flatten())**2)
    rmse = np.sqrt(mse)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    import matplotlib.pyplot as plt
    for i in range(10):
        plt.imshow(X_test[i])
        plt.title(f"True: {y_test[i]:.3f}, Pred: {y_pred[i][0]:.3f}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()