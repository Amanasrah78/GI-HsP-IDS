from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE = Path("/home/ahmad/datasets-working/MQTTset/Data/CSV")

OUTPUT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "datasets/processed/mqttset_raw_temporal"
)

FILES = {
    "bruteforce": "bruteforce.csv",
    "flood": "flood.csv",
    "legitimate": "legitimate_1w.csv",
    "dos": "malaria.csv",
    "malformed": "malformed.csv",
    "slowite": "slowite.csv",
}

LABELS = {
    "bruteforce": 0,
    "dos": 1,
    "flood": 2,
    "legitimate": 3,
    "malformed": 4,
    "slowite": 5,
}

WINDOW = 10

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.10
TEST_FRACTION = 0.30

TRAIN_CAP = 50000
VAL_CAP = 10000
TEST_CAP = 20000

CHUNK_SIZE = 200000
SEED = 42


FEATURES = [
    "tcp.time_delta",
    "tcp.len",
    "tcp.window_size_value",
    "mqtt.conack.flags.reserved",
    "mqtt.conack.flags.sp",
    "mqtt.conack.val",
    "mqtt.conflag.cleansess",
    "mqtt.conflag.passwd",
    "mqtt.conflag.qos",
    "mqtt.conflag.reserved",
    "mqtt.conflag.retain",
    "mqtt.conflag.uname",
    "mqtt.conflag.willflag",
    "mqtt.dupflag",
    "mqtt.kalive",
    "mqtt.len",
    "mqtt.msgid",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.qos",
    "mqtt.retain",
    "mqtt.sub.qos",
    "mqtt.suback.qos",
    "mqtt.ver",
]

TIME_COLUMN = "frame.time_epoch"
USECOLS = [TIME_COLUMN] + FEATURES


def clean_chunk(chunk):
    chunk[TIME_COLUMN] = pd.to_numeric(
        chunk[TIME_COLUMN],
        errors="coerce",
    )

    for col in FEATURES:
        chunk[col] = pd.to_numeric(
            chunk[col],
            errors="coerce",
        )

    chunk = chunk.dropna(subset=[TIME_COLUMN])

    chunk[FEATURES] = (
        chunk[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return chunk


def count_rows(path):
    total = 0

    for chunk in pd.read_csv(
        path,
        usecols=[TIME_COLUMN],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total += len(chunk)

    return total


def get_boundaries(total_rows):
    train_end = int(
        total_rows * TRAIN_FRACTION
    )

    val_end = int(
        total_rows
        * (TRAIN_FRACTION + VAL_FRACTION)
    )

    return train_end, val_end


def fit_scaler(totals):
    scaler = MinMaxScaler()

    print(
        "\nFitting scaler incrementally "
        "on TRAIN rows only..."
    )

    for class_name, filename in FILES.items():
        path = BASE / filename

        total_rows = totals[class_name]
        train_end, _ = get_boundaries(total_rows)

        seen = 0

        print(
            f"{class_name:12s} "
            f"train_rows={train_end}"
        )

        for chunk in pd.read_csv(
            path,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk = clean_chunk(chunk)

            if seen >= train_end:
                break

            remaining = train_end - seen

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

            X = chunk[FEATURES].to_numpy(
                dtype=np.float32
            )

            scaler.partial_fit(X)

            seen += len(chunk)

    return scaler


def split_range(total_rows, split):
    train_end, val_end = get_boundaries(
        total_rows
    )

    if split == "train":
        return 0, train_end

    if split == "val":
        return train_end, val_end

    if split == "test":
        return val_end, total_rows

    raise ValueError(split)


def reservoir_windows(
    path,
    split,
    label,
    scaler,
    cap,
    total_rows,
):
    start_row, end_row = split_range(
        total_rows,
        split,
    )

    rng = random.Random(
        SEED
        + 1000 * label
        + {
            "train": 0,
            "val": 100,
            "test": 200,
        }[split]
    )

    reservoir = []
    buffer = []

    global_row = 0
    windows_seen = 0

    for chunk in pd.read_csv(
        path,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk = clean_chunk(chunk)

        chunk_start = global_row
        chunk_end = global_row + len(chunk)

        global_row = chunk_end

        if chunk_end <= start_row:
            continue

        if chunk_start >= end_row:
            break

        local_start = max(
            0,
            start_row - chunk_start,
        )

        local_end = min(
            len(chunk),
            end_row - chunk_start,
        )

        if local_start >= local_end:
            continue

        # Preserve original capture order.
        sub = chunk.iloc[
            local_start:local_end
        ]

        X = sub[FEATURES].to_numpy(
            dtype=np.float32
        )

        X = scaler.transform(
            X
        ).astype(np.float32)

        for row in X:
            buffer.append(row)

            if len(buffer) > WINDOW:
                buffer.pop(0)

            if len(buffer) < WINDOW:
                continue

            window = np.stack(
                buffer,
                axis=0,
            ).astype(np.float32)

            windows_seen += 1

            if len(reservoir) < cap:
                reservoir.append(
                    window.copy()
                )
            else:
                j = rng.randint(
                    0,
                    windows_seen - 1,
                )

                if j < cap:
                    reservoir[j] = (
                        window.copy()
                    )

    if not reservoir:
        X_out = np.empty(
            (0, WINDOW, len(FEATURES)),
            dtype=np.float32,
        )
    else:
        X_out = np.stack(
            reservoir,
            axis=0,
        ).astype(np.float32)

    y_out = np.full(
        len(X_out),
        label,
        dtype=np.int64,
    )

    return (
        X_out,
        y_out,
        windows_seen,
    )


def build_split(
    split,
    cap,
    scaler,
    totals,
    metadata,
):
    X_parts = []
    y_parts = []

    print(
        f"\nGenerating {split.upper()} windows..."
    )

    for class_name, filename in FILES.items():
        label = LABELS[class_name]

        X, y, total_windows = (
            reservoir_windows(
                path=BASE / filename,
                split=split,
                label=label,
                scaler=scaler,
                cap=cap,
                total_rows=totals[
                    class_name
                ],
            )
        )

        print(
            f"{class_name:12s} "
            f"total={total_windows:8d} "
            f"kept={len(y):6d}"
        )

        metadata[class_name][
            f"{split}_total_windows"
        ] = int(total_windows)

        metadata[class_name][
            f"{split}_kept_windows"
        ] = int(len(y))

        X_parts.append(X)
        y_parts.append(y)

    X = np.concatenate(
        X_parts,
        axis=0,
    )

    y = np.concatenate(
        y_parts,
        axis=0,
    )

    rng = np.random.default_rng(
        SEED
        + {
            "train": 0,
            "val": 1,
            "test": 2,
        }[split]
    )

    perm = rng.permutation(len(y))

    return X[perm], y[perm]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Counting capture rows...")

    totals = {}

    metadata = {
        "window": WINDOW,
        "train_fraction":
            TRAIN_FRACTION,
        "val_fraction":
            VAL_FRACTION,
        "test_fraction":
            TEST_FRACTION,
        "train_cap":
            TRAIN_CAP,
        "val_cap":
            VAL_CAP,
        "test_cap":
            TEST_CAP,
        "seed": SEED,
        "features": FEATURES,
        "labels": LABELS,
        "sequence_order":
            "original capture order / frame.number order",
        "split_policy":
            "raw capture split into 60% train, "
            "10% validation, 30% test before "
            "window construction",
        "window_overlap_across_splits":
            False,
    }

    for class_name, filename in FILES.items():
        total = count_rows(
            BASE / filename
        )

        totals[class_name] = total

        train_end, val_end = (
            get_boundaries(total)
        )

        metadata[class_name] = {
            "rows_total": int(total),
            "rows_train": int(train_end),
            "rows_val": int(
                val_end - train_end
            ),
            "rows_test": int(
                total - val_end
            ),
        }

        print(
            f"{class_name:12s} "
            f"rows={total}"
        )

    scaler = fit_scaler(totals)

    X_train, y_train = build_split(
        "train",
        TRAIN_CAP,
        scaler,
        totals,
        metadata,
    )

    X_val, y_val = build_split(
        "val",
        VAL_CAP,
        scaler,
        totals,
        metadata,
    )

    X_test, y_test = build_split(
        "test",
        TEST_CAP,
        scaler,
        totals,
        metadata,
    )

    print("\nFinal arrays:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)

    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test,
    )

    joblib.dump(
        scaler,
        OUTPUT_DIR
        / "minmax_scaler.joblib",
    )

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nSaved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()