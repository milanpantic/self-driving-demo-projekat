import os

import numpy as np
import pandas as pd


ANGLE_COL = 3
GROUP_COL = 7
SEGMENT_LENGTH = 100
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
MERGED_DIR = "complete_dataset/merged"


def fix_image_paths(csv_path, save_path):
    """Menja stare Windows putanje putanjama koje koristi projekat."""
    df = pd.read_csv(csv_path, header=None)
    old_prefix = "C:\\Users\\Andy\\Desktop\\"

    for column in [0, 1, 2]:
        df[column] = df[column].str.replace(
            old_prefix,
            "complete_dataset/",
            regex=False,
        )
        df[column] = df[column].str.replace("\\", "/", regex=False)

    df.to_csv(save_path, index=False, header=False)
    return df


def print_dataset_info(name, df):
    """Ispisuje osnovnu statistiku uglova."""
    angles = df[ANGLE_COL]

    print(f"Dataset: {name}")
    print(f"Broj uzoraka: {len(df)}")
    print(f"Min ugao: {angles.min():.3f}")
    print(f"Max ugao: {angles.max():.3f}")
    print(f"Mean: {angles.mean():.3f}")
    print(f"Std: {angles.std():.3f}")
    print(f"Levo (<0): {(angles < 0).sum()}")
    print(f"Pravo (==0): {(angles == 0).sum()}")
    print(f"Desno (>0): {(angles > 0).sum()}")
    print(f"Broj vremenskih segmenata: {df[GROUP_COL].nunique()}")


def create_segments(df, source_name):
    """Deli jednu vožnju na hronološke segmente od po 100 frejmova."""
    segments = []
    segment_number = 0

    for start in range(0, len(df), SEGMENT_LENGTH):
        end = start + SEGMENT_LENGTH
        segment = df.iloc[start:end].copy()

        if len(segment) >= 4:
            segment[GROUP_COL] = f"{source_name}_{segment_number:04d}"
            segments.append(segment)

        segment_number += 1

    return segments


def get_angle_distribution(df):
    """Vraća udeo levih, pravih i desnih uglova."""
    angles = df[ANGLE_COL]
    left_ratio = (angles < 0).mean()
    straight_ratio = (angles == 0).mean()
    right_ratio = (angles > 0).mean()
    return left_ratio, straight_ratio, right_ratio


def find_test_start(segments, number_of_test_segments):
    """Bira kontinuirani test blok sličan celoj vožnji."""
    complete_drive = pd.concat(segments, ignore_index=True)
    target_left, target_straight, target_right = get_angle_distribution(complete_drive)

    best_start = 0
    best_score = float("inf")
    expected_size = number_of_test_segments * SEGMENT_LENGTH
    last_possible_start = len(segments) - number_of_test_segments

    for start in range(last_possible_start + 1):
        end = start + number_of_test_segments
        candidate = pd.concat(segments[start:end], ignore_index=True)
        left, straight, right = get_angle_distribution(candidate)

        distribution_error = (
            abs(left - target_left)
            + abs(straight - target_straight)
            + abs(right - target_right)
        )
        size_error = abs(len(candidate) - expected_size) / expected_size
        score = distribution_error + size_error

        if score < best_score:
            best_score = score
            best_start = start

    return best_start


def split_drive(df, source_name, seed):
    """Pravi train, validation i jedan kontinuirani test deo jedne vožnje."""
    segments = create_segments(df, source_name)
    number_of_val_segments = max(1, round(len(segments) * VAL_RATIO))
    number_of_test_segments = max(1, round(len(segments) * TEST_RATIO))

    test_start = find_test_start(segments, number_of_test_segments)
    test_end = test_start + number_of_test_segments

    remaining_indices = []
    for index in range(len(segments)):
        if index < test_start or index >= test_end:
            remaining_indices.append(index)

    remaining_indices = np.array(remaining_indices)
    random_generator = np.random.default_rng(seed)
    random_generator.shuffle(remaining_indices)

    number_of_train_segments = len(remaining_indices) - number_of_val_segments
    train_segments = []
    val_segments = []

    for position, segment_index in enumerate(remaining_indices):
        if position < number_of_train_segments:
            train_segments.append(segments[segment_index])
        else:
            val_segments.append(segments[segment_index])

    test_segments = segments[test_start:test_end]
    test_df = pd.concat(test_segments, ignore_index=True)
    test_df[GROUP_COL] = f"{source_name}_test_contiguous"

    return segments, train_segments, val_segments, test_df


def create_and_save_splits(jungle_df, lake_df):
    """Spaja delove obe vožnje i čuva završne CSV fajlove."""
    jungle_all, jungle_train, jungle_val, jungle_test = split_drive(
        jungle_df,
        "jungle",
        SEED,
    )
    lake_all, lake_train, lake_val, lake_test = split_drive(
        lake_df,
        "lake",
        SEED + 1000,
    )

    all_segments = jungle_all + lake_all
    train_segments = jungle_train + lake_train
    val_segments = jungle_val + lake_val

    merged_df = pd.concat(all_segments, ignore_index=True)
    train_df = pd.concat(train_segments, ignore_index=True)
    val_df = pd.concat(val_segments, ignore_index=True)
    test_df = pd.concat([jungle_test, lake_test], ignore_index=True)

    os.makedirs(MERGED_DIR, exist_ok=True)
    merged_df.to_csv(
        os.path.join(MERGED_DIR, "driving_log_merged.csv"),
        index=False,
        header=False,
    )
    train_df.to_csv(
        os.path.join(MERGED_DIR, "train.csv"),
        index=False,
        header=False,
    )
    val_df.to_csv(
        os.path.join(MERGED_DIR, "val.csv"),
        index=False,
        header=False,
    )
    test_df.to_csv(
        os.path.join(MERGED_DIR, "test.csv"),
        index=False,
        header=False,
    )

    print(
        f"[SPLIT] Train: {len(train_df)} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def main():
    jungle_df = fix_image_paths(
        "complete_dataset/self_driving_car_dataset_jungle/driving_log.csv",
        "complete_dataset/self_driving_car_dataset_jungle/driving_log_fixed.csv",
    )
    lake_df = fix_image_paths(
        "complete_dataset/self_driving_car_dataset/driving_log.csv",
        "complete_dataset/self_driving_car_dataset/driving_log_fixed.csv",
    )

    train_df, val_df, test_df = create_and_save_splits(jungle_df, lake_df)
    print_dataset_info("train.csv", train_df)
    print_dataset_info("val.csv", val_df)
    print_dataset_info("test.csv", test_df)


if __name__ == "__main__":
    main()
