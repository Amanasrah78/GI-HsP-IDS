from pathlib import Path
import json

import numpy as np


INPUT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "datasets/processed/mqttset_baseline"
)

OUTPUT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "datasets/processed/mqttset_temporal"
)

WINDOW = 10
STRIDE = 1


def make_windows(X, y, window, stride):
    n = len(X)

    count = (n - window) // stride + 1

    Xw = np.empty(
        (count, window, X.shape[1]),
        dtype=np.float32,
    )

    yw = np.empty(
        count,
        dtype=np.int64,
    )

    out_idx = 0

    for start in range(
        0,
        n - window + 1,
        stride,
    ):
        end = start + window

        Xw[out_idx] = X[start:end]

        # Label the window using the final observation,
        # consistent with sequence-to-current-state classification.
        yw[out_idx] = y[end - 1]

        out_idx += 1

    return Xw, yw


def main():
    print("Loading flat MQTTset arrays...")

    X_train = np.load(
        INPUT_DIR / "X_train.npy"
    ).astype(np.float32)

    X_test = np.load(
        INPUT_DIR / "X_test.npy"
    ).astype(np.float32)

    y_train = np.load(
        INPUT_DIR / "y_train.npy"
    ).astype(np.int64)

    y_test = np.load(
        INPUT_DIR / "y_test.npy"
    ).astype(np.int64)

    print("Flat train:", X_train.shape)
    print("Flat test :", X_test.shape)

    print("\nCreating train windows...")
    X_train_w, y_train_w = make_windows(
        X_train,
        y_train,
        WINDOW,
        STRIDE,
    )

    print("Creating test windows...")
    X_test_w, y_test_w = make_windows(
        X_test,
        y_test,
        WINDOW,
        STRIDE,
    )

    print("\nTemporal shapes:")
    print("X_train:", X_train_w.shape)
    print("y_train:", y_train_w.shape)
    print("X_test :", X_test_w.shape)
    print("y_test :", y_test_w.shape)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train_w,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test_w,
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train_w,
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test_w,
    )

    metadata = {
        "window_length": WINDOW,
        "stride": STRIDE,
        "label_rule": "last_sample",
        "train_windows": int(len(y_train_w)),
        "test_windows": int(len(y_test_w)),
        "features": int(X_train.shape[1]),
        "split_policy": (
            "train/test separated before windowing"
        ),
    }

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
