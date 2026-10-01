from pathlib import Path
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    "datasets/processed/mqttset_temporal"
)

RESULT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "results/processed/mqttset_gru"
)

MODEL_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints/mqttset_gru"
)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5
HIDDEN_SIZE = 64


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MQTTGRU(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        output, hidden = self.gru(x)

        # Final hidden state
        z = hidden[-1]

        return self.classifier(z)


def load_data():
    X_train = np.load(
        DATA_DIR / "X_train.npy"
    ).astype(np.float32)

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    ).astype(np.float32)

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    ).astype(np.int64)

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    ).astype(np.int64)

    return X_train, X_test, y_train, y_test


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate_model(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(
                preds.cpu().numpy()
            )

            all_targets.append(
                yb.numpy()
            )

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return y_true, y_pred


def main():
    set_seed(SEED)

    device = torch.device("cpu")
    print("Device:", device)

    X_train, X_test, y_train, y_test = load_data()

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )

    test_loader = make_loader(
        X_test,
        y_test,
        shuffle=False,
    )

    model = MQTTGRU(
        input_dim=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_classes=len(np.unique(y_train)),
    ).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_loss = float("inf")
    patience_counter = 0

    start_training = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        epoch_loss = total_loss / total_samples

        print(
            f"Epoch {epoch:02d} "
            f"train_loss={epoch_loss:.6f}"
        )

        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "mqttset_gru_best.pt",
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

    training_seconds = (
        time.perf_counter()
        - start_training
    )

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "mqttset_gru_best.pt",
            map_location=device,
        )
    )

    start_inference = time.perf_counter()

    y_true, y_pred = evaluate_model(
        model,
        test_loader,
        device,
    )

    inference_seconds = (
        time.perf_counter()
        - start_inference
    )

    results = {
        "model": "mqttset_gru_erm",
        "seed": SEED,

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),

        "training_seconds": float(
            training_seconds
        ),

        "inference_seconds": float(
            inference_seconds
        ),

        "samples_per_second": float(
            len(y_true) / inference_seconds
        ),

        "epochs_completed": int(epoch),
    }

    print("\n" + "=" * 70)
    print("mqttset_gru_erm")
    print("=" * 70)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    np.save(
        RESULT_DIR / "predictions.npy",
        y_pred,
    )


if __name__ == "__main__":
    main()
