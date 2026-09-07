from pathlib import Path
import json
import math
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
    "results/processed/mqttset_mftst"
)

MODEL_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints/mqttset_mftst"
)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 6

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.30

M_PAPER = 16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DCT1D(nn.Module):
    def __init__(self, T):
        super().__init__()

        n = torch.arange(
            T,
            dtype=torch.float32,
        ).unsqueeze(0)

        k = torch.arange(
            T,
            dtype=torch.float32,
        ).unsqueeze(1)

        basis = torch.cos(
            math.pi
            / T
            * (n + 0.5)
            * k
        )

        basis[0] *= math.sqrt(1.0 / T)

        if T > 1:
            basis[1:] *= math.sqrt(2.0 / T)

        self.register_buffer(
            "basis",
            basis,
        )

    def forward(self, x):
        # x: B x T x D
        #
        # DCT is applied across the
        # temporal dimension T for each
        # learned feature channel D.
        return torch.einsum(
            "kt,btd->bkd",
            self.basis,
            x,
        )


class MFCA(nn.Module):
    def __init__(
        self,
        d_model,
        T,
        m_paper=16,
    ):
        super().__init__()

        self.T = T

        self.m_eff = min(
            m_paper,
            T,
        )

        self.dct = DCT1D(T)

        # Learnable weights over selected
        # temporal-frequency components.
        self.freq_weights = nn.Parameter(
            torch.ones(self.m_eff)
            / self.m_eff
        )

        # g = [q ; r ; q*r]
        self.channel_gate = nn.Sequential(
            nn.Linear(
                3 * d_model,
                d_model,
            ),
            nn.ReLU(),
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.Sigmoid(),
        )

    def forward(self, H):
        # H: B x T x D

        F = self.dct(H)

        # Select first M_eff temporal
        # DCT components.
        F_selected = F[:, :self.m_eff, :]

        weights = torch.softmax(
            self.freq_weights,
            dim=0,
        )

        q = torch.sum(
            F_selected
            * weights.view(
                1,
                self.m_eff,
                1,
            ),
            dim=1,
        )

        # Temporal context descriptor
        r = H.mean(dim=1)

        g = torch.cat(
            [
                q,
                r,
                q * r,
            ],
            dim=1,
        )

        a = self.channel_gate(g)

        H_refined = (
            H * a.unsqueeze(1)
        )

        return H_refined, q, r, a


class MFTST(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        T,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            D_MODEL,
        )

        self.positional_encoding = (
            PositionalEncoding(
                D_MODEL
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dim_feedforward=
                    DIM_FEEDFORWARD,
                dropout=DROPOUT,
                batch_first=True,
                activation="relu",
                norm_first=False,
            )
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.mfca = MFCA(
            d_model=D_MODEL,
            T=T,
            m_paper=M_PAPER,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(
                D_MODEL,
                num_classes,
            ),
        )

    def forward(self, x):
        x = self.input_projection(x)

        x = self.positional_encoding(x)

        H = self.encoder(x)

        H_refined, q, r, a = (
            self.mfca(H)
        )

        z = H_refined.mean(dim=1)

        logits = self.classifier(z)

        return logits


def make_loader(
    X,
    y,
    shuffle,
):
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


def predict(
    model,
    loader,
    device,
):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)

            pred = torch.argmax(
                logits,
                dim=1,
            )

            all_true.append(
                yb.numpy()
            )

            all_pred.append(
                pred.cpu().numpy()
            )

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

    T = X_train.shape[1]

    print("Window T:", T)
    print("M paper :", M_PAPER)
    print("M eff   :", min(M_PAPER, T))

    num_classes = len(
        np.unique(y_train)
    )

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

    weight_tensor = torch.tensor(
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

    model = MFTST(
        input_dim=X_train.shape[2],
        num_classes=num_classes,
        T=T,
    ).to(device)

    print("\n", model)

    params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        params,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    start_training = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            bs = xb.size(0)

            total_loss += (
                loss.item() * bs
            )

            total_samples += bs

        train_loss = (
            total_loss
            / total_samples
        )

        y_val_true, y_val_pred = (
            predict(
                model,
                val_loader,
                device,
            )
        )

        val_macro_f1 = f1_score(
            y_val_true,
            y_val_pred,
            average="macro",
            zero_division=0,
        )

        val_bal_acc = (
            balanced_accuracy_score(
                y_val_true,
                y_val_pred,
            )
        )

        print(
            f"Epoch {epoch:02d} "
            f"train_loss="
            f"{train_loss:.6f} "
            f"val_macro_f1="
            f"{val_macro_f1:.6f} "
            f"val_bal_acc="
            f"{val_bal_acc:.6f}"
        )

        if (
            val_macro_f1
            > best_macro_f1 + 1e-5
        ):
            best_macro_f1 = (
                val_macro_f1
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best.pt",
            )

        else:
            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):
            print(
                "Early stopping "
                f"at epoch {epoch}."
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

    start_inference = (
        time.perf_counter()
    )

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
        "model":
            "mqttset_mftst_reimplementation",
        "seed":
            SEED,
        "window_T":
            int(T),
        "M_paper":
            M_PAPER,
        "M_effective":
            min(M_PAPER, T),
        "best_epoch":
            int(best_epoch),
        "best_val_macro_f1":
            float(best_macro_f1),
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        "mcc":
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            ),
        "parameters":
            int(params),
        "training_seconds":
            float(training_seconds),
        "inference_seconds":
            float(inference_seconds),
        "samples_per_second":
            float(
                len(y_true)
                / inference_seconds
            ),
    }

    print(
        "\n"
        + "=" * 70
    )

    print(
        "mqttset_mftst_reimplementation"
    )

    print("=" * 70)

    for key, value in (
        results.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

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
        RESULT_DIR
        / "predictions.npy",
        y_pred,
    )

    np.save(
        RESULT_DIR
        / "targets.npy",
        y_true,
    )


if __name__ == "__main__":
    main()