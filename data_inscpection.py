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
    df = pd.read_csv(csv_path, header=None)
    old_prefix = "C:\\Users\\Andy\\Desktop\\"

    for col in (0, 1, 2):
        df[col] = df[col].str.replace(old_prefix, "complete_dataset/", regex=False)
        df[col] = df[col].str.replace("\\", "/", regex=False)

    df.to_csv(save_path, index=False, header=False)
    return df


def print_dataset_info(name, df):
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


def make_segments(df, source_name):
    segments = []
    for number, start in enumerate(range(0, len(df), SEGMENT_LENGTH)):
        segment = df.iloc[start:start + SEGMENT_LENGTH].copy()
        if len(segment) < 4:
            continue
        segment[GROUP_COL] = f"{source_name}_{number:04d}"
        segments.append(segment)
    return segments


def angle_distribution(df):
    angles = df[ANGLE_COL]
    return np.array([
        (angles < 0).mean(),
        (angles == 0).mean(),
        (angles > 0).mean(),
    ])


def find_representative_test_start(segments, block_count):
    """Find a continuous block similar to the complete drive."""
    complete_drive = pd.concat(segments, ignore_index=True)
    target_distribution = angle_distribution(complete_drive)
    target_size = block_count * SEGMENT_LENGTH

    best_start = 0
    best_score = float("inf")

    for start in range(len(segments) - block_count + 1):
        candidate = pd.concat(segments[start:start + block_count], ignore_index=True)
        distribution_error = np.abs(
            angle_distribution(candidate) - target_distribution
        ).sum()
        size_error = abs(len(candidate) - target_size) / target_size
        score = distribution_error + size_error

        if score < best_score:
            best_start = start
            best_score = score

    return best_start


def split_drive(df, source_name, seed):
    segments = make_segments(df, source_name)
    n_val = max(1, round(len(segments) * VAL_RATIO))
    n_test = max(1, round(len(segments) * TEST_RATIO))

    test_start = find_representative_test_start(segments, n_test)
    test_indices = np.arange(test_start, test_start + n_test)
    remaining_indices = np.setdiff1d(np.arange(len(segments)), test_indices)
    np.random.default_rng(seed).shuffle(remaining_indices)

    n_train = len(remaining_indices) - n_val
    test_block = pd.concat(
        [segments[index] for index in test_indices],
        ignore_index=True,
    )
    test_block[GROUP_COL] = f"{source_name}_test_contiguous"

    split = {
        "train": [segments[index] for index in remaining_indices[:n_train]],
        "val": [segments[index] for index in remaining_indices[n_train:]],
        "test": [test_block],
    }
    return segments, split


def create_splits(sources):
    all_segments = []
    split_segments = {"train": [], "val": [], "test": []}

    for source_number, (source_name, source_df) in enumerate(sources):
        segments, source_split = split_drive(
            source_df,
            source_name,
            seed=SEED + source_number * 1000,
        )
        all_segments.extend(segments)
        for split_name in split_segments:
            split_segments[split_name].extend(source_split[split_name])

    datasets = {
        "driving_log_merged": pd.concat(all_segments, ignore_index=True),
        **{
            name: pd.concat(segments, ignore_index=True)
            for name, segments in split_segments.items()
        },
    }

    os.makedirs(MERGED_DIR, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(os.path.join(MERGED_DIR, f"{name}.csv"), index=False, header=False)

    print(
        f"[SPLIT] Train: {len(datasets['train'])} | "
        f"Val: {len(datasets['val'])} | Test: {len(datasets['test'])}"
    )
    return datasets


def main():
    jungle = fix_image_paths(
        "complete_dataset/self_driving_car_dataset_jungle/driving_log.csv",
        "complete_dataset/self_driving_car_dataset_jungle/driving_log_fixed.csv",
    )
    lake = fix_image_paths(
        "complete_dataset/self_driving_car_dataset/driving_log.csv",
        "complete_dataset/self_driving_car_dataset/driving_log_fixed.csv",
    )

    datasets = create_splits([("jungle", jungle), ("lake", lake)])
    print_dataset_info("train.csv", datasets["train"])
    print_dataset_info("val.csv", datasets["val"])
    print_dataset_info("test.csv", datasets["test"])


if __name__ == "__main__":
    main()
