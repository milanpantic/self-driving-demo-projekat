import pandas as pd
import numpy as np

SEQ_LENGTH = 4
GROUP_COL = 7


def valid_sequence_starts(sequence_groups, seq_length):
    starts = np.arange(len(sequence_groups) - seq_length + 1)
    valid = np.ones(len(starts), dtype=bool)
    for offset in range(1, seq_length):
        valid &= sequence_groups[starts] == sequence_groups[starts + offset]
    return starts[valid]


paths = {
    "train": "complete_dataset/merged/train.csv",
    "val": "complete_dataset/merged/val.csv",
    "test": "complete_dataset/merged/test.csv",
}
datasets = {name: pd.read_csv(path, header=None) for name, path in paths.items()}
merged = pd.read_csv("complete_dataset/merged/driving_log_merged.csv", header=None)

print("merged", len(merged))
print("split sum", sum(map(len, datasets.values())))
assert len(merged) == sum(map(len, datasets.values()))

for name, df in datasets.items():
    angles = df[3]
    starts = valid_sequence_starts(df[GROUP_COL].astype(str).to_numpy(), SEQ_LENGTH)
    print(
        name,
        "rows", len(df),
        "segments", df[GROUP_COL].nunique(),
        "valid sequences", len(starts),
        "left/straight/right",
        int((angles < 0).sum()),
        int((angles == 0).sum()),
        int((angles > 0).sum()),
    )
    assert len(starts) > 0
    assert (angles < 0).any() and (angles == 0).any() and (angles > 0).any()

for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
    first_paths = set(datasets[first][0])
    second_paths = set(datasets[second][0])
    first_groups = set(datasets[first][GROUP_COL])
    second_groups = set(datasets[second][GROUP_COL])
    print(f"{first}-{second} paths disjoint", first_paths.isdisjoint(second_paths))
    print(f"{first}-{second} segments disjoint", first_groups.isdisjoint(second_groups))
    assert first_paths.isdisjoint(second_paths)
    assert first_groups.isdisjoint(second_groups)

print("Sve provere su prošle.")
