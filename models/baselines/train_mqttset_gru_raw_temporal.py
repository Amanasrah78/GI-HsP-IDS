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
    "datasets/processed/mqttset_raw_temporal"
)

RESULT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "results/processed/mqttset_gru_raw_temporal"
)

MODEL_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints/mqttset_gru_raw_temporal"
)

SEED = 42
BATCH_SIZE = 512
EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6
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
        _, hidden = self.gru(x)
        z = hidden[-1]
        return self.classifier(z)


def make_loader(X, y, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def predict(model, loader, device):
    model.eval()

    all_pred = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)
            pred = torch.argmax(logits, dim=1)

            all_pred.append(pred.cpu().numpy())
            all_true.append(yb.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


def main():
    set_seed(SEED)

    device = torch.device("cpu")

    print("Device:", device)

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    ).astype(np.float32)

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    ).astype(np.int64)

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    ).astype(np.float32)

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    ).astype(np.int64)

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    ).astype(np.float32)

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    ).astype(np.int64)

    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    num_classes = len(np.unique(y_train))

    counts = np.bincount(
        y_train,
        minlength=num_classes,
    )
    class_weights = (
        len(y_train)
        / (
            num_classes
            * counts
        )
    )

    print("\nTrain class counts:")
    print(counts)

    print("Class weights:")
    print(class_weights)

    weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    train_loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        shuffle=False,
    )

    test_loader = make_loader(
        X_test,
        y_test,
        shuffle=False,
    )

    model = MQTTGRU(
        input_dim=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes,
    ).to(device)

    print("\n", model)

    criterion = nn.CrossEntropyLoss(
        weight=weights_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_macro_f1 = -1.0
    patience_counter = 0
    best_epoch = 0

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

            bs = xb.size(0)

            total_loss += (
                loss.item() * bs
            )

            total_samples += bs

        train_loss = (
            total_loss / total_samples
        )

        y_val_true, y_val_pred = predict(
            model,
            val_loader,
            device,
        )

        val_macro_f1 = f1_score(
            y_val_true,
            y_val_pred,
            average="macro",
            zero_division=0,
        )

        val_balanced = balanced_accuracy_score(
            y_val_true,
            y_val_pred,
        )

        print(
            f"Epoch {epoch:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_macro_f1={val_macro_f1:.6f} "
            f"val_bal_acc={val_balanced:.6f}"
        )

        if val_macro_f1 > best_macro_f1 + 1e-5:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best.pt",
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    training_seconds = (
        time.perf_counter()
        - start_training
    )

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "best.pt",
            map_location=device,
        )
    )

    start_inference = time.perf_counter()

    y_true, y_pred = predict(
        model,
        test_loader,
        device,
    )

    inference_seconds = (
        time.perf_counter()
        - start_inference
    )

    results = {
        "model": "mqttset_gru_raw_temporal",
        "seed": SEED,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(
            best_macro_f1
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
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
            len(y_true)
            / inference_seconds
        ),
    }

    print("\n" + "=" * 70)
    print("mqttset_gru_raw_temporal")
    print("=" * 70)

    for k, v in results.items():
        print(f"{k}: {v}")

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

    np.save(
        RESULT_DIR / "targets.npy",
        y_true,
    )


if __name__ == "__main__":
    main()
