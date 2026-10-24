
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


V3_PATH = Path("data/processed/churn_dataset_v3.csv")
V4_PATH = Path("data/processed/churn_dataset_v4.csv")

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
]


def main():
    v3 = pd.read_csv(V3_PATH)
    v4 = pd.read_csv(V4_PATH)

    X = v3[FEATURES]
    y = (v3["churn"] == 0).astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    # Select threshold using V3 validation only.
    val_prob = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(0.05, 0.501, 0.01):
        pred = (val_prob >= threshold).astype(int)
        score = f1_score(y_val, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    # Final temporal evaluation on V4.
    X4 = v4[FEATURES]
    y4 = (v4["churn"] == 0).astype(int)
    prob4 = model.predict_proba(X4)[:, 1]
    pred4 = (prob4 >= best_threshold).astype(int)

    print("\nTEMPORAL EVALUATION: MODEL B")
    print("-" * 55)
    print(f"Training V3: {len(X_train)} clients")
    print(f"Validation V3: {len(X_val)} clients")
    print(f"V4: {len(v4)} clients")
    print(f"V4 observed returns: {int(y4.sum())} ({y4.mean():.2%})")
    print(f"Threshold selected on V3 validation: {best_threshold:.2f}")
    print(f"Validation F1: {best_f1:.4f}")
    print(f"V4 Average Precision: {average_precision_score(y4, prob4):.4f}")
    print(f"V4 Precision: {precision_score(y4, pred4, zero_division=0):.4f}")
    print(f"V4 Recall: {recall_score(y4, pred4, zero_division=0):.4f}")
    print(f"V4 F1: {f1_score(y4, pred4, zero_division=0):.4f}")
    print("Confusion matrix [actual rows, predicted columns]:")
    print(confusion_matrix(y4, pred4, labels=[0, 1]))

    baseline = y4.mean()
    order = np.argsort(prob4)[::-1]

    print("\nTOP-K TARGETING")
    for pct in (0.01, 0.02, 0.05, 0.10):
        k = max(1, int(len(y4) * pct))
        detected = int(y4.iloc[order[:k]].sum())
        precision_k = detected / k
        recall_k = detected / int(y4.sum())
        lift = precision_k / baseline if baseline else 0
        print(
            f"Top {pct:.0%}: {k} clients, {detected} returns, "
            f"precision={precision_k:.2%}, "
            f"recall={recall_k:.2%}, lift={lift:.2f}x"
        )


if __name__ == "__main__":
    main()
