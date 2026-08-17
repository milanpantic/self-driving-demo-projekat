import os
import random

import numpy as np
import pandas as pd


ANGLE_COL = 3
SEQUENCE_GROUP_COL = 7
SEGMENT_LENGTH = 100
SEED = 42

 
def fix_image_paths(
    csv_path,
    old_prefix="C:\\Users\\Andy\\Desktop\\",
    new_prefix="complete_dataset/",
    save_path=None,
):
    df = pd.read_csv(csv_path, header=None)

    for col in (0, 1, 2):
        df[col] = df[col].str.replace(old_prefix, new_prefix, regex=False)
        df[col] = df[col].str.replace("\\", "/", regex=False)

    if save_path is not None:
        df.to_csv(save_path, index=False, header=False)

    return df


def inspect_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    steering = df[ANGLE_COL]

    print(f"Dataset: {csv_path}")
    print(f"Broj uzoraka: {len(df)}")
    print(f"Min ugao: {steering.min():.3f}")
    print(f"Max ugao: {steering.max():.3f}")
    print(f"Mean: {steering.mean():.3f}")
    print(f"Std: {steering.std():.3f}")
    print(f"Levo (<0): {(steering < 0).sum()}")
    print(f"Pravo (==0): {(steering == 0).sum()}")
    print(f"Desno (>0): {(steering > 0).sum()}")

    if SEQUENCE_GROUP_COL in df.columns:
        print(f"Broj vremenskih segmenata: {df[SEQUENCE_GROUP_COL].nunique()}")

    return df


def balance_make_dataset(
    csv_path,
    out_csv,
    keep_zero_ratio=0.25,
    angle_col=ANGLE_COL,
    seed=SEED,
):
    """Filter zeros without reordering rows; intended only for a single-frame CNN.

    CNN+LSTM training keeps every chronological frame and balances complete
    sequences in the generator instead, because removing frames creates time gaps.
    """
    df = pd.read_csv(csv_path, header=None)
    zero_indices = df.index[df[angle_col] == 0.0].to_numpy()
    rng = np.random.default_rng(seed)
    keep_count = int(len(zero_indices) * keep_zero_ratio)
    kept_zero_indices = rng.choice(zero_indices, size=keep_count, replace=False)

    keep_mask = (df[angle_col] != 0.0) | df.index.isin(kept_zero_indices)
    balanced_df = df.loc[keep_mask].sort_index()
    balanced_df.to_csv(out_csv, index=False, header=False)

    print(f"Pre: {len(df)}")
    print(f"Posle: {len(balanced_df)}")
    print(f"Levo: {(balanced_df[angle_col] < 0).sum()}")
    print(f"Pravo: {(balanced_df[angle_col] == 0).sum()}")
    print(f"Desno: {(balanced_df[angle_col] > 0).sum()}")
    return balanced_df


def make_segments(df, source_name, segment_length=SEGMENT_LENGTH):
    """Create chronological segments and store their ID in column 7."""
    if segment_length < 4:
        raise ValueError("Segment mora imati najmanje 4 frejma")

    segments = []
    for segment_number, start in enumerate(range(0, len(df), segment_length)):
        segment = df.iloc[start:start + segment_length].copy()
        if len(segment) < 4:
            continue
        segment[SEQUENCE_GROUP_COL] = f"{source_name}_{segment_number:04d}"
        segments.append(segment)
    return segments


def _representative_contiguous_block_start(segments, block_size):
    """Choose a continuous block whose angle distribution resembles its drive."""
    complete_drive = pd.concat(segments, ignore_index=True)
    complete_angles = complete_drive[ANGLE_COL]
    target_distribution = np.array([
        (complete_angles < 0).mean(),
        (complete_angles == 0).mean(),
        (complete_angles > 0).mean(),
    ])
    target_size = min(block_size * max(map(len, segments)), len(complete_drive))

    best_start = 0
    best_score = float("inf")
    for start in range(len(segments) - block_size + 1):
        candidate = pd.concat(segments[start:start + block_size], ignore_index=True)
        angles = candidate[ANGLE_COL]
        distribution = np.array([
            (angles < 0).mean(),
            (angles == 0).mean(),
            (angles > 0).mean(),
        ])
        distribution_error = np.abs(distribution - target_distribution).sum()
        size_error = abs(len(candidate) - target_size) / target_size
        score = distribution_error + size_error
        if score < best_score:
            best_start = start
            best_score = score

    return best_start


def _split_source_segments(
    segments,
    source_name,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=SEED,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train/val/test odnosi moraju dati zbir 1.0")
    if len(segments) < 3:
        raise ValueError("Potrebna su najmanje tri segmenta po izvornoj vožnji")

    n_val = max(1, round(len(segments) * val_ratio))
    n_test = max(1, round(len(segments) * test_ratio))
    test_start = _representative_contiguous_block_start(segments, n_test)
    test_indices = np.arange(test_start, test_start + n_test)

    indices = np.setdiff1d(np.arange(len(segments)), test_indices)
    np.random.default_rng(seed).shuffle(indices)
    n_train = len(indices) - n_val
    if n_train < 1:
        raise ValueError("Nema dovoljno segmenata za trening")

    test_block = pd.concat(
        [segments[i] for i in test_indices],
        ignore_index=True,
    )
    test_block[SEQUENCE_GROUP_COL] = f"{source_name}_test_contiguous"

    return {
        "train": [segments[i] for i in indices[:n_train]],
        "val": [segments[i] for i in indices[n_train:]],
        "test": [test_block],
    }


def prepare_segmented_split(
    sources,
    merged_csv,
    train_csv,
    val_csv,
    test_csv,
    segment_length=SEGMENT_LENGTH,
    seed=SEED,
):
    """Split both drives by chronological segments, before any augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    all_segments = []
    split_segments = {"train": [], "val": [], "test": []}

    for source_number, (source_name, csv_path) in enumerate(sources):
        source_df = pd.read_csv(csv_path, header=None)
        segments = make_segments(source_df, source_name, segment_length)
        source_split = _split_source_segments(
            segments,
            source_name,
            seed=seed + source_number * 1000,
        )
        all_segments.extend(segments)
        for split_name in split_segments:
            split_segments[split_name].extend(source_split[split_name])

    merged_df = pd.concat(all_segments, ignore_index=True)
    split_dfs = {
        name: pd.concat(segments, ignore_index=True)
        for name, segments in split_segments.items()
    }

    merged_df.to_csv(merged_csv, index=False, header=False)
    split_dfs["train"].to_csv(train_csv, index=False, header=False)
    split_dfs["val"].to_csv(val_csv, index=False, header=False)
    split_dfs["test"].to_csv(test_csv, index=False, header=False)

    print(
        f"[SPLIT] Train: {len(split_dfs['train'])} | "
        f"Val: {len(split_dfs['val'])} | Test: {len(split_dfs['test'])}"
    )
    return split_dfs["train"], split_dfs["val"], split_dfs["test"]


def main():
    jungle_raw = "complete_dataset/self_driving_car_dataset_jungle/driving_log.csv"
    jungle_fixed = "complete_dataset/self_driving_car_dataset_jungle/driving_log_fixed.csv"
    lake_raw = "complete_dataset/self_driving_car_dataset/driving_log.csv"
    lake_fixed = "complete_dataset/self_driving_car_dataset/driving_log_fixed.csv"
    merged_dir = "complete_dataset/merged"
    os.makedirs(merged_dir, exist_ok=True)

    fix_image_paths(jungle_raw, save_path=jungle_fixed)
    fix_image_paths(lake_raw, save_path=lake_fixed)

    prepare_segmented_split(
        sources=[("jungle", jungle_fixed), ("lake", lake_fixed)],
        merged_csv=os.path.join(merged_dir, "driving_log_merged.csv"),
        train_csv=os.path.join(merged_dir, "train.csv"),
        val_csv=os.path.join(merged_dir, "val.csv"),
        test_csv=os.path.join(merged_dir, "test.csv"),
        segment_length=SEGMENT_LENGTH,
        seed=SEED,
    )

    inspect_dataset(os.path.join(merged_dir, "train.csv"))
    inspect_dataset(os.path.join(merged_dir, "val.csv"))
    inspect_dataset(os.path.join(merged_dir, "test.csv"))


if __name__ == "__main__":
    main()
