import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


TRAIN_PATH = Path(
    "/home/ahmad/datasets-working/MQTTset/Data/FINAL_CSV/train70_reduced.csv"
)

TEST_PATH = Path(
    "/home/ahmad/datasets-working/MQTTset/Data/FINAL_CSV/test30_reduced.csv"
)

OUTPUT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/datasets/processed/mqttset_baseline"
)

TARGET = "target"

# High-cardinality payload-like field.
DROP_COLUMNS = [
    "mqtt.msg",
]

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-12


def load_data():
    print("Loading MQTTset...")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    return train, test


def remove_selected_columns(train, test):
    columns_to_drop = [
        c for c in DROP_COLUMNS
        if c in train.columns
    ]

    print("Dropping columns:", columns_to_drop)

    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)

    return train, test


def encode_categorical_columns(train, test):
    categorical_columns = [
        c
        for c in train.columns
        if c != TARGET
        and (
            train[c].dtype == "object"
            or isinstance(train[c].dtype, pd.StringDtype)
        )
    ]

    print("\nCategorical columns:")
    for c in categorical_columns:
        print(" -", c)

    mappings = {}

    for column in categorical_columns:

        train_values = (
            train[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        test_values = (
            test[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        categories = sorted(train_values.unique().tolist())

        mapping = {
            value: index
            for index, value in enumerate(categories)
        }

        mappings[column] = mapping

        train[column] = (
            train_values
            .map(mapping)
            .fillna(-1)
            .astype(np.int32)
        )

        # unseen test categories become -1
        test[column] = (
            test_values
            .map(mapping)
            .fillna(-1)
            .astype(np.int32)
        )

    return train, test, mappings


def remove_zero_variance_columns(train, test):
    feature_columns = [
        c for c in train.columns
        if c != TARGET
    ]

    variances = train[feature_columns].var(numeric_only=True)

    zero_variance_columns = variances[
        variances <= NEAR_ZERO_VARIANCE_THRESHOLD
    ].index.tolist()

    print("\nZero/near-zero variance columns:")
    for c in zero_variance_columns:
        print(" -", c)

    train = train.drop(columns=zero_variance_columns)
    test = test.drop(columns=zero_variance_columns)

    return train, test, zero_variance_columns


def encode_target(train, test):
    labels = sorted(train[TARGET].unique().tolist())

    label_mapping = {
        label: index
        for index, label in enumerate(labels)
    }

    print("\nLabel mapping:")
    for label, index in label_mapping.items():
        print(f" {label} -> {index}")

    y_train = train[TARGET].map(label_mapping).astype(np.int64)
    y_test = test[TARGET].map(label_mapping)

    if y_test.isna().any():
        unknown = test.loc[y_test.isna(), TARGET].unique()
        raise ValueError(
            f"Unknown labels found in test set: {unknown}"
        )

    y_test = y_test.astype(np.int64)

    X_train = train.drop(columns=[TARGET])
    X_test = test.drop(columns=[TARGET])

    return X_train, X_test, y_train, y_test, label_mapping


def scale_features(X_train, X_test):
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def save_outputs(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    mappings,
    label_mapping,
    zero_variance_columns,
    scaler,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    with open(
        OUTPUT_DIR / "feature_names.json",
        "w"
    ) as f:
        json.dump(feature_names, f, indent=2)

    with open(
        OUTPUT_DIR / "categorical_mappings.json",
        "w"
    ) as f:
        json.dump(mappings, f, indent=2)

    with open(
        OUTPUT_DIR / "label_mapping.json",
        "w"
    ) as f:
        json.dump(label_mapping, f, indent=2)

    with open(
        OUTPUT_DIR / "dropped_zero_variance.json",
        "w"
    ) as f:
        json.dump(zero_variance_columns, f, indent=2)

    joblib.dump(
        scaler,
        OUTPUT_DIR / "minmax_scaler.joblib"
    )

    print("\nSaved processed files to:")
    print(OUTPUT_DIR)


def main():
    train, test = load_data()

    train, test = remove_selected_columns(
        train,
        test,
    )

    train, test, mappings = encode_categorical_columns(
        train,
        test,
    )

    train, test, zero_variance_columns = (
        remove_zero_variance_columns(
            train,
            test,
        )
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_mapping,
    ) = encode_target(
        train,
        test,
    )

    feature_names = X_train.columns.tolist()

    print("\nFinal number of features:")
    print(len(feature_names))

    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_test,
    )

    print("\nFinal arrays:")
    print("X_train:", X_train_scaled.shape)
    print("X_test :", X_test_scaled.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    save_outputs(
        X_train_scaled,
        X_test_scaled,
        y_train.to_numpy(),
        y_test.to_numpy(),
        feature_names,
        mappings,
        label_mapping,
        zero_variance_columns,
        scaler,
    )


if __name__ == "__main__":
    main()
