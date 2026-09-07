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
    matthews_corrcoef,
)


DATA_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "datasets/processed/mqttset_raw_temporal"
)

RESULT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "results/processed/mqttset_pgd_at"
)

MODEL_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints"
)

SEED = 42
BATCH_SIZE = 256
EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 6

EPSILON = 0.03
PGD_ALPHA = 0.003

# Use fewer PGD steps during training
# to keep CPU cost manageable.
PGD_TRAIN_STEPS = 5

# Final evaluation uses the stronger
# PGD-20 setting.
PGD_EVAL_STEPS = 20

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

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MQTTTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            D_MODEL,
        )

        self.positional_encoding = PositionalEncoding(
            D_MODEL
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
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

        z = self.encoder(x)
        z = z.mean(dim=1)

        return self.classifier(z)


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
            math.pi / T
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

        self.m_eff = min(
            m_paper,
            T,
        )

        self.dct = DCT1D(T)

        self.freq_weights = nn.Parameter(
            torch.ones(self.m_eff)
            / self.m_eff
        )

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
        F = self.dct(H)

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

        return H * a.unsqueeze(1)


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

        self.positional_encoding = PositionalEncoding(
            D_MODEL
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
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
        H = self.mfca(H)

        z = H.mean(dim=1)

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


def metrics(y_true, y_pred):
    return {
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
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
    }


def pgd_attack(
    model,
    xb,
    yb,
    epsilon,
    alpha,
    steps,
):
    x_orig = xb.detach()

    x_adv = (
        x_orig
        + torch.empty_like(
            x_orig
        ).uniform_(
            -epsilon,
            epsilon,
        )
    )

    x_adv = torch.clamp(
        x_adv,
        0.0,
        1.0,
    )

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = model(x_adv)

        loss = nn.functional.cross_entropy(
            logits,
            yb,
        )

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.detach()

        x_adv = (
            x_adv.detach()
            + alpha * grad.sign()
        )

        delta = torch.clamp(
            x_adv - x_orig,
            -epsilon,
            epsilon,
        )

        x_adv = torch.clamp(
            x_orig + delta,
            0.0,
            1.0,
        ).detach()

    return x_adv


def evaluate_clean(
    model,
    loader,
    device,
):
    model.eval()

    ys = []
    preds = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = model(xb)

            pred = torch.argmax(
                logits,
                dim=1,
            )

            ys.append(
                yb.numpy()
            )

            preds.append(
                pred.cpu().numpy()
            )

    return (
        np.concatenate(ys),
        np.concatenate(preds),
    )


def evaluate_pgd(
    model,
    loader,
    device,
):
    model.eval()

    ys = []
    preds = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        x_adv = pgd_attack(
            model,
            xb,
            yb,
            EPSILON,
            PGD_ALPHA,
            PGD_EVAL_STEPS,
        )

        with torch.no_grad():
            logits = model(x_adv)

            pred = torch.argmax(
                logits,
                dim=1,
            )

        ys.append(
            yb.cpu().numpy()
        )

        preds.append(
            pred.cpu().numpy()
        )

    return (
        np.concatenate(ys),
        np.concatenate(preds),
    )


def train_one(
    name,
    model,
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    device,
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model = model.to(device)

    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoint_dir = (
        MODEL_DIR
        / f"{name}_pgd_at"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    start = time.perf_counter()

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

            # Generate PGD-5 examples.
            model.eval()

            x_adv = pgd_attack(
                model,
                xb,
                yb,
                EPSILON,
                PGD_ALPHA,
                PGD_TRAIN_STEPS,
            )

            model.train()

            optimizer.zero_grad()

            logits_clean = model(xb)
            logits_adv = model(x_adv)

            loss_clean = criterion(
                logits_clean,
                yb,
            )

            loss_adv = criterion(
                logits_adv,
                yb,
            )

            # 50/50 clean + adversarial objective.
            loss = (
                0.5 * loss_clean
                + 0.5 * loss_adv
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
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
            evaluate_clean(
                model,
                val_loader,
                device,
            )
        )

        val_f1 = f1_score(
            y_val_true,
            y_val_pred,
            average="macro",
            zero_division=0,
        )

        print(
            f"Epoch {epoch:02d} "
            f"train_loss={train_loss:.6f} "
            f"val_macro_f1={val_f1:.6f}"
        )

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                checkpoint_dir / "best.pt",
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
        - start
    )

    model.load_state_dict(
        torch.load(
            checkpoint_dir / "best.pt",
            map_location=device,
        )
    )

    y_true, y_pred = evaluate_clean(
        model,
        test_loader,
        device,
    )

    clean = metrics(
        y_true,
        y_pred,
    )

    print("\nClean:")
    print(clean)

    y_true, y_pred = evaluate_pgd(
        model,
        test_loader,
        device,
    )

    pgd = metrics(
        y_true,
        y_pred,
    )

    print("\nPGD-20:")
    print(pgd)

    return {
        "model": name,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "training_seconds": training_seconds,
        "clean": clean,
        "pgd20": pgd,
    }


def main():
    set_seed(SEED)

    device = torch.device("cpu")

    print("Device:", device)

    print(
        "PGD training:",
        f"eps={EPSILON}",
        f"alpha={PGD_ALPHA}",
        f"steps={PGD_TRAIN_STEPS}",
    )

    print(
        "PGD evaluation:",
        f"steps={PGD_EVAL_STEPS}",
    )

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

    print("Class weights:")
    print(class_weights)

    train_loader = make_loader(
        X_train,
        y_train,
        True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        False,
    )

    test_loader = make_loader(
        X_test,
        y_test,
        False,
    )

    input_dim = X_train.shape[2]
    T = X_train.shape[1]

    models = {
        "transformer": MQTTTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
        ),

        "mftst": MFTST(
            input_dim=input_dim,
            num_classes=num_classes,
            T=T,
        ),
    }

    results = {
        "epsilon": EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_train_steps": PGD_TRAIN_STEPS,
        "pgd_eval_steps": PGD_EVAL_STEPS,
    }

    for name, model in models.items():
        results[name] = train_one(
            name,
            model,
            train_loader,
            val_loader,
            test_loader,
            class_weights,
            device,
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

    print("\nSaved to:")
    print(RESULT_DIR)


if __name__ == "__main__":
    main()