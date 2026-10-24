
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("data/processed/churn_dataset_v3.csv")
TARGET = "churn"

# churn = 0: a delivered return was observed
# churn = 1: no delivered return was observed
# We predict the return class as the positive class.
FEATURE_SETS = {
    "A_all_features": [
        "frequence_180j",
        "montant_articles_180j",
        "recence_jours",
        "panier_moyen_180j",
    ],
    "B_without_average_basket": [
        "frequence_180j",
        "montant_articles_180j",
        "recence_jours",
    ],
    "C_log_amount": [
        "frequence_180j",
        "montant_log",
        "recence_jours",
    ],
}


def evaluate_top_k(y_true, probabilities, percentage):
    k = max(1, int(len(y_true) * percentage))
    ranking = np.argsort(probabilities)[::-1][:k]
    detected = int(y_true.iloc[ranking].sum())
    total_returns = int(y_true.sum())

    precision_at_k = detected / k
    recall_at_k = detected / total_returns if total_returns else 0
    baseline = float(y_true.mean())
    lift = precision_at_k / baseline if baseline else 0

    return k, detected, precision_at_k, recall_at_k, lift


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = [
        TARGET,
        "frequence_180j",
        "montant_articles_180j",
        "recence_jours",
        "panier_moyen_180j",
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Missing values detected in target/features.")

    # Transform the skewed amount feature for configuration C.
    if (df["montant_articles_180j"] < 0).any():
        raise ValueError("Negative amounts detected; log1p is not applicable.")

    df["montant_log"] = np.log1p(df["montant_articles_180j"])

    # Positive class = observed return within 180 days.
    y = (df[TARGET] == 0).astype(int)

    X = df[list(dict.fromkeys(
        column
        for features in FEATURE_SETS.values()
        for column in features
    ))]

    # Same 60/20/20 split protocol, with stratification.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )

    X_val, _, y_val, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print("\nFEATURE SET COMPARISON - V3")
    print("-" * 65)
    print(f"Total clients: {len(df)}")
    print(f"Training: {len(X_train)} | Returns: {int(y_train.sum())}")
    print(f"Validation: {len(X_val)} | Returns: {int(y_val.sum())}")
    print(f"Validation return rate: {y_val.mean():.2%}")
    print("\nAP is the primary metric. Thresholds are selected on validation only.")

    results = []

    for name, features in FEATURE_SETS.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        )

        model.fit(X_train[features], y_train)
        probabilities = model.predict_proba(X_val[features])[:, 1]

        # Select threshold by F1 on validation data only.
        best = None
        for threshold in np.arange(0.05, 0.501, 0.01):
            predictions = (probabilities >= threshold).astype(int)
            f1 = f1_score(y_val, predictions, zero_division=0)

            if best is None or f1 > best["f1"]:
                best = {
                    "threshold": float(threshold),
                    "f1": float(f1),
                    "precision": float(
                        precision_score(y_val, predictions, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_val, predictions, zero_division=0)
                    ),
                }

        ap = average_precision_score(y_val, probabilities)

        print(f"\n{name}")
        print(f"Features: {features}")
        print(f"Average Precision: {ap:.4f}")
        print(f"Best threshold: {best['threshold']:.2f}")
        print(f"Precision at threshold: {best['precision']:.4f}")
        print(f"Recall at threshold: {best['recall']:.4f}")
        print(f"F1 at threshold: {best['f1']:.4f}")

        for percentage in (0.01, 0.02, 0.05, 0.10):
            k, detected, precision_k, recall_k, lift = evaluate_top_k(
                y_val, probabilities, percentage
            )
            print(
                f"Top {percentage:.0%}: k={k}, returns={detected}, "
                f"precision={precision_k:.2%}, "
                f"recall={recall_k:.2%}, lift={lift:.2f}x"
            )

        results.append({
            "model": name,
            "average_precision": ap,
            "threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        })

    comparison = pd.DataFrame(results).sort_values(
        "average_precision", ascending=False
    )

    print("\n" + "=" * 65)
    print("FINAL COMPARISON (ranked by Average Precision)")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nNo files were written. V4 was not used for model selection.")


if __name__ == "__main__":
    main()
