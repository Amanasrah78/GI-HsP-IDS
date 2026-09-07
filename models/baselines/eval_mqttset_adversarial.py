from pathlib import Path
import json
import math
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

CHECKPOINT_BASE = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "models/checkpoints"
)

RESULT_DIR = Path(
    "/home/ahmad/research/gi-hsp-ids/"
    "results/processed/mqttset_adversarial"
)

BATCH_SIZE = 256

EPSILON = 0.03

PGD_ALPHA = 0.003
PGD_STEPS = 20

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.30

M_PAPER = 16


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model,
        )

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
            * (
                -math.log(10000.0)
                / d_model
            )
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
        return (
            x
            + self.pe[:, :x.size(1)]
        )


class MQTTGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_classes,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(
                hidden_size,
                num_classes,
            ),
        )

    def forward(self, x):
        _, hidden = self.gru(x)

        z = hidden[-1]

        return self.classifier(z)


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

        self.positional_encoding = (
            PositionalEncoding(
                D_MODEL
            )
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
            math.pi
            / T
            * (n + 0.5)
            * k
        )

        basis[0] *= math.sqrt(
            1.0 / T
        )

        if T > 1:
            basis[1:] *= math.sqrt(
                2.0 / T
            )

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

        F_selected = (
            F[:, :self.m_eff, :]
        )

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

        return (
            H * a.unsqueeze(1)
        )


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


def make_loader(X, y):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
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


def clean_predict(
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


def fgsm_attack(
    model,
    xb,
    yb,
    epsilon,
):
    x_adv = (
        xb.detach()
        .clone()
        .requires_grad_(True)
    )

    logits = model(x_adv)

    loss = nn.functional.cross_entropy(
        logits,
        yb,
    )

    model.zero_grad()

    loss.backward()

    grad = x_adv.grad.detach()

    x_adv = (
        x_adv
        + epsilon * grad.sign()
    )

    x_adv = torch.clamp(
        x_adv,
        0.0,
        1.0,
    )

    return x_adv.detach()


def pgd_attack(
    model,
    xb,
    yb,
    epsilon,
    alpha,
    steps,
):
    x_orig = xb.detach()

    # random start inside epsilon ball
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

        loss = (
            nn.functional.cross_entropy(
                logits,
                yb,
            )
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
            min=-epsilon,
            max=epsilon,
        )

        x_adv = torch.clamp(
            x_orig + delta,
            0.0,
            1.0,
        ).detach()

    return x_adv


def adversarial_predict(
    model,
    loader,
    device,
    attack,
):
    model.eval()

    ys = []
    preds = []

    start = time.perf_counter()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if attack == "fgsm":
            x_adv = fgsm_attack(
                model,
                xb,
                yb,
                EPSILON,
            )

        elif attack == "pgd":
            x_adv = pgd_attack(
                model,
                xb,
                yb,
                EPSILON,
                PGD_ALPHA,
                PGD_STEPS,
            )

        else:
            raise ValueError(attack)

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

    seconds = (
        time.perf_counter()
        - start
    )

    return (
        np.concatenate(ys),
        np.concatenate(preds),
        seconds,
    )


def evaluate_model(
    name,
    model,
    checkpoint,
    loader,
    device,
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
        )
    )

    model.to(device)
    model.eval()

    y_true, y_pred = clean_predict(
        model,
        loader,
        device,
    )

    clean = metrics(
        y_true,
        y_pred,
    )

    print("\nClean:")
    print(clean)

    y_true, y_pred, fgsm_seconds = (
        adversarial_predict(
            model,
            loader,
            device,
            "fgsm",
        )
    )

    fgsm = metrics(
        y_true,
        y_pred,
    )

    fgsm[
        "evaluation_seconds"
    ] = float(fgsm_seconds)

    print("\nFGSM:")
    print(fgsm)

    y_true, y_pred, pgd_seconds = (
        adversarial_predict(
            model,
            loader,
            device,
            "pgd",
        )
    )

    pgd = metrics(
        y_true,
        y_pred,
    )

    pgd[
        "evaluation_seconds"
    ] = float(pgd_seconds)

    print("\nPGD:")
    print(pgd)

    return {
        "clean": clean,
        "fgsm": fgsm,
        "pgd": pgd,
    }


def main():
    device = torch.device("cpu")

    print("Device:", device)

    print(
        "epsilon:",
        EPSILON,
    )

    print(
        "PGD alpha:",
        PGD_ALPHA,
    )

    print(
        "PGD steps:",
        PGD_STEPS,
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    ).astype(np.float32)

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    ).astype(np.int64)

    print(
        "Test:",
        X_test.shape,
    )

    loader = make_loader(
        X_test,
        y_test,
    )

    input_dim = X_test.shape[2]
    T = X_test.shape[1]

    num_classes = len(
        np.unique(y_test)
    )

    models = {
        "gru": (
            MQTTGRU(
                input_dim=input_dim,
                hidden_size=64,
                num_classes=num_classes,
            ),
            CHECKPOINT_BASE
            / "mqttset_gru_raw_temporal"
            / "best.pt",
        ),

        "transformer": (
            MQTTTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
            ),
            CHECKPOINT_BASE
            / "mqttset_transformer"
            / "best.pt",
        ),

        "mftst": (
            MFTST(
                input_dim=input_dim,
                num_classes=num_classes,
                T=T,
            ),
            CHECKPOINT_BASE
            / "mqttset_mftst"
            / "best.pt",
        ),
    }

    results = {
        "epsilon": EPSILON,
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
    }

    for name, (
        model,
        checkpoint,
    ) in models.items():

        if not checkpoint.exists():
            print(
                "\nMissing checkpoint:",
                checkpoint,
            )

            continue

        results[name] = (
            evaluate_model(
                name,
                model,
                checkpoint,
                loader,
                device,
            )
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_DIR
        / "metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nSaved to:",
        RESULT_DIR,
    )


if __name__ == "__main__":
    main()