import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2


import cv2
import numpy as np
import pandas as pd
import os
import random


def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    factor = 0.6 + np.random.rand() * 0.8   
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_shadow(img):
    h, w = img.shape[:2]
    x1, x2 = np.random.randint(0, w, 2)
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    shadow_mask[:, min(x1, x2):max(x1, x2)] = 1
    alpha = 0.4 + np.random.rand() * 0.3
    img_shadow = img.copy()
    img_shadow[shadow_mask == 1] = (img_shadow[shadow_mask == 1] * alpha).astype(np.uint8)
    return img_shadow

def augment(img):
    if np.random.rand() < 0.5:
        img = random_brightness(img)
    if np.random.rand() < 0.5:
        img = random_shadow(img)
    return img

def augment_dataset_sequential(
    input_csv,
    output_csv,
    aug_img_dir,
    aug_ratio=0.15,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(aug_img_dir, exist_ok=True)

    df = pd.read_csv(input_csv, header=None)
    new_rows = []

    num_aug = int(len(df) * aug_ratio)
    aug_indices = set(np.random.choice(len(df), num_aug, replace=False))

    print(f"[AUG] Ukupno: {len(df)} | Augmentiramo: {num_aug} (~{aug_ratio*100:.0f}%)")

    for i, row in df.iterrows():
        img_path = row[0]

         
        new_rows.append(row.tolist())

         
        if i in aug_indices:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_img = augment(img)

            new_name = os.path.basename(img_path).replace(".jpg", "_aug.jpg")
            new_path = os.path.join(aug_img_dir, new_name)

            cv2.imwrite(new_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            aug_row = row.tolist()
            aug_row[0] = new_path   

            new_rows.append(aug_row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False, header=False)

    print(f"[AUG] Gotovo. Novi dataset: {len(new_df)} uzoraka")
    print(f"[AUG] CSV sačuvan u: {output_csv}")

def fix_image_paths(csv_path,
                    old_prefix = "C:\\Users\\Andy\\Desktop\\",
                    new_prefix="complete_dataset/",
                    save_path=None):

    df = pd.read_csv(csv_path, header=None)

    print(df.head())

    path_columns = [0, 1, 2]

    for col in path_columns:

        df[col] = df[col].str.replace(old_prefix, new_prefix, regex=False)

        df[col] = df[col].str.replace("\\", "/", regex=False)

    if save_path is not None:
        df.to_csv(save_path, index=False, header=None)

    return df

def inspect_dataset(csv_path, plot_histogram=True):

    df = pd.read_csv(csv_path, header=None)

    steering_col = 3

    steering = df[steering_col]

    print(f"Dataset: {csv_path}")
    print(f"Broj uzoraka: {len(df)}")
    print(f"Min ugao: {steering.min():.3f}")
    print(f"Max ugao: {steering.max():.3f}")
    print(f"Mean: {steering.mean():.3f}")
    print(f"Std: {steering.std():.3f}")

    n_left = (steering < 0).sum()
    n_right = (steering > 0).sum()
    n_zero = (steering == 0).sum()

    print(f"Levo (<0): {n_left}")
    print(f"Pravo (==0): {n_zero}")
    print(f"Desno (>0): {n_right}")

    if plot_histogram:
        plt.figure(figsize=(8,5))
        plt.hist(steering, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Distribucija ugla upravljanja: {csv_path}")
        plt.xlabel("Steering angle")
        plt.ylabel("Broj uzoraka")
        plt.grid(True)
        plt.show()

    return df

def reduce_zero(csv_path, steering_col=3):

    df = pd.read_csv(csv_path, header=None)

    left_right = df[df[steering_col] != 0]
    zero = df[df[steering_col] == 0]

    n_keep = len(left_right)
    zero_reduced = zero.iloc[:n_keep]

    df_new = pd.concat([left_right, zero_reduced], ignore_index=True)

    return df_new

def balance_make_dataset(
    csv_path,
    img_root,
    out_csv,
    out_img_dir,
    keep_zero_ratio=0.25,
    angle_col=3
):
    os.makedirs(out_img_dir, exist_ok=True)

    df = pd.read_csv(csv_path, header=None)

    zeros = df[df[angle_col] == 0.0]
    left = df[df[angle_col] < 0.0]
    right = df[df[angle_col] > 0.0]

    print(f"Pre: {len(df)}")
    print(f"Levo: {len(left)}, Pravo: {len(zeros)}, Desno: {len(right)}")

    zeros_keep = zeros.iloc[:int(len(zeros) * keep_zero_ratio)]

    new_rows = []
    new_rows.extend(zeros_keep.values.tolist())
    new_rows.extend(left.values.tolist())
    new_rows.extend(right.values.tolist())

  
    for row in left.itertuples(index=False):
        img_path = row[0]
        angle = row[angle_col]

        full_path = img_path
        img = cv2.imread(full_path)
        if img is None:
            print('IMAGE IS NONE')
            continue

        img_flipped = cv2.flip(img, 1)

        new_name = os.path.basename(img_path).replace(".jpg", "_mirror.jpg")
        new_path = os.path.join(out_img_dir, new_name)

        cv2.imwrite(new_path, img_flipped)

        new_row = list(row)
        new_row[0] = new_path
        new_row[angle_col] = -angle

        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(out_csv, index=False, header=False)

    print("Posle:", len(new_df))
    print("Levo:", (new_df[angle_col] < 0).sum())
    print("Pravo:", (new_df[angle_col] == 0).sum())
    print("Desno:", (new_df[angle_col] > 0).sum())

def reduce_straight_driving(
    csv_path,
    zero_keep_ratio=0.70,
    angle_col=3,
    save_path=None
):
    df = pd.read_csv(csv_path, header=None)

    zero_mask = df[angle_col] == 0.0
    non_zero_df = df[~zero_mask]
    zero_df = df[zero_mask]

    keep_n = int(len(zero_df) * zero_keep_ratio)
    zero_df_reduced = zero_df.iloc[:keep_n]   

    new_df = pd.concat([non_zero_df, zero_df_reduced], axis=0)
    new_df = new_df.sort_index()   

    print("Pre:", len(df))
    print("Posle:", len(new_df))
    print("Nule:", len(zero_df_reduced))

    if save_path:
        new_df.to_csv(save_path, index=False, header=False)

    return new_df

def merge_datasets(csv_paths, out_csv):
    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path, header=None)
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

    merged_df.to_csv(out_csv, index=False, header=False)

    print(f"✔ Spojeno {len(csv_paths)} dataset-a")
    print(f"✔ Ukupno uzoraka: {len(merged_df)}")

    return merged_df

def main():
    input_csv_jungle = "complete_dataset/self_driving_car_dataset_jungle/driving_log.csv"
    output_csv_jungle = "complete_dataset/self_driving_car_dataset_jungle/driving_log_fixed.csv"

    input_csv_lake = "complete_dataset/self_driving_car_dataset/driving_log.csv"
    output_csv_lake = "complete_dataset/self_driving_car_dataset/driving_log_fixed.csv"
    
    df = fix_image_paths(
        csv_path=input_csv_jungle,
        save_path=output_csv_jungle
    )

    df = fix_image_paths(
        csv_path=input_csv_lake,
        save_path=output_csv_lake
    )

    inspect_dataset(output_csv_jungle)
    inspect_dataset(output_csv_lake)

    reduce_straight_driving(
    "complete_dataset/self_driving_car_dataset_jungle/driving_log_fixed.csv",
    zero_keep_ratio=0.70,
    save_path="complete_dataset/self_driving_car_dataset_jungle/driving_log_reduced.csv"
    )

    balance_make_dataset(
    csv_path="complete_dataset/self_driving_car_dataset/driving_log_fixed.csv",
    img_root="complete_dataset/self_driving_car_dataset",
    out_csv="complete_dataset/self_driving_car_dataset/driving_log_balanced.csv",
    out_img_dir="complete_dataset/self_driving_car_dataset/mirrored",
    keep_zero_ratio=0.25
    )
    
    inspect_dataset("complete_dataset/self_driving_car_dataset_jungle/driving_log_reduced.csv")
    inspect_dataset("complete_dataset/self_driving_car_dataset/driving_log_balanced.csv")

    merged_csv = "complete_dataset/merged/driving_log_merged.csv"
    os.makedirs("complete_dataset/merged", exist_ok=True)

    merge_datasets(
    csv_paths=[
        "complete_dataset/self_driving_car_dataset_jungle/driving_log_reduced.csv",
        "complete_dataset/self_driving_car_dataset/driving_log_balanced.csv"
    ],
    out_csv=merged_csv
    )

    inspect_dataset(merged_csv) 


    augment_dataset_sequential(
        input_csv="complete_dataset/merged/driving_log_merged.csv",
        output_csv="complete_dataset/merged/driving_log_augmented.csv",
        aug_img_dir="complete_dataset/merged/augmented",
        aug_ratio=0.15
    )

if __name__ == "__main__":
    main()