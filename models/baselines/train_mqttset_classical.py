from pathlib import Path
import json
import time

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


DATA_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "datasets/processed/mqttset_baseline"
)

RESULT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "results/processed/mqttset_classical"
)

MODEL_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints/mqttset_classical"
)

SEED = 42


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test, training_time):
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start

    results = {
        "model": name,
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y_test, y_pred)
        ),
        "training_seconds": float(training_time),
        "inference_seconds": float(inference_time),
        "samples_per_second": float(
            len(y_test) / inference_time
        ),
    }

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for key, value in results.items():
        if key != "model":
            print(f"{key}: {value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_DIR / f"{name}_metrics.json",
        "w",
    ) as f:
        json.dump(results, f, indent=2)

    np.save(
        RESULT_DIR / f"{name}_predictions.npy",
        y_pred,
    )

    return results


def train_random_forest(
    X_train,
    y_train,
):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1,
        class_weight=None,
    )

    print("\nTraining Random Forest...")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def train_hgb(
    X_train,
    y_train,
):
    model = HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_iter=200,
        random_state=SEED,
    )

    print("\nTraining HistGradientBoosting...")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return model, training_time


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, X_test, y_train, y_test = load_data()

    print("Dataset loaded:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    rf, rf_time = train_random_forest(
        X_train,
        y_train,
    )

    evaluate(
        "random_forest_standard",
        rf,
        X_test,
        y_test,
        rf_time,
    )

    joblib.dump(
        rf,
        MODEL_DIR / "random_forest.joblib",
    )

    hgb, hgb_time = train_hgb(
        X_train,
        y_train,
    )

    evaluate(
        "hist_gradient_boosting",
        hgb,
        X_test,
        y_test,
        hgb_time,
    )

    joblib.dump(
        hgb,
        MODEL_DIR / "hist_gradient_boosting.joblib",
    )


if __name__ == "__main__":
    main()
