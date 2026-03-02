import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Dropout, Input, 
                                     TimeDistributed, LSTM, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

CSV_PATH = "complete_dataset/merged/driving_log_merged.csv"
BATCH_SIZE = 32
EPOCHS = 20
IMG_HEIGHT = 66
IMG_WIDTH = 200
SEQ_LENGTH = 5  

def load_image(path):
    img = cv2.imread(path)
    if img is None: return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
    img = img[60:135, :, :]
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img / 255.0

def load_dataset(csv_path):
     
    df = pd.read_csv(csv_path, header=None)
    image_paths = df[0].values   
    steering = df[3].values.astype(np.float32)  
    return image_paths, steering

def sequence_generator(image_paths, steering, batch_size, seq_length):
    num_samples = len(image_paths)
    while True:
         
        indices = np.arange(num_samples - seq_length)
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch, y_batch = [], []
            
            for idx in batch_indices:
                 
                seq_imgs = [load_image(image_paths[idx + k]) for k in range(seq_length)]
                angle = steering[idx + seq_length - 1]
                
                 
                if np.random.rand() > 0.5:
                    seq_imgs = [cv2.flip(img, 1) for img in seq_imgs]
                    angle = -angle  
                
                X_batch.append(seq_imgs)
                y_batch.append(angle)
                
            yield np.array(X_batch), np.array(y_batch)

def build_cnn_lstm(seq_length):
    input_layer = Input(shape=(seq_length, IMG_HEIGHT, IMG_WIDTH, 3))
     
    x = TimeDistributed(Conv2D(16, (5, 5), strides=(2, 2), activation='elu'))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='elu'))(x)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(64, return_sequences=False, dropout=0.3)(x)

    x = Dense(32, activation='elu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)  

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

def detect_lane_change(steering_history, threshold=0.12):
    if len(steering_history) < 5: return False
    diff = np.abs(np.diff(steering_history)).mean()
    return diff > threshold
 
def main():
    image_paths, steering = load_dataset(CSV_PATH)
     
    n_train = int(len(image_paths) * 0.8)
    X_train_paths, y_train = image_paths[:n_train], steering[:n_train]
    X_val_paths, y_val = image_paths[n_train:], steering[n_train:]

    print(f"Trening uzoraka: {len(X_train_paths)}, Validacionih uzoraka: {len(X_val_paths)}")

    train_gen = sequence_generator(X_train_paths, y_train, BATCH_SIZE, SEQ_LENGTH)
    val_gen = sequence_generator(X_val_paths, y_val, BATCH_SIZE, SEQ_LENGTH)

    model = build_cnn_lstm(SEQ_LENGTH)
    
    checkpoint = ModelCheckpoint("best_model.h5", monitor="val_loss", save_best_only=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(
        train_gen,
        steps_per_epoch=len(X_train_paths) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(X_val_paths) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )

    model.save("final_cnn_lstm_model.h5")
    print("Trening završen.")

if __name__ == "__main__":
    main()