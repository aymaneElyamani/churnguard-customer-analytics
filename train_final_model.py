
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/churn_dataset_v3.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churnguard_model_v1.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + ["churn"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Missing values found. Clean the data first.")

    X = df[FEATURES]
    # Positive class = a delivered return observed within 180 days.
    y = (df["churn"] == 0).astype(int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    # Refit on all V3 data after the feature-set selection.
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "ChurnGuard Logistic Regression",
        "version": "1.0",
        "dataset": str(DATA_PATH),
        "features": FEATURES,
        "target": "observed delivered return within 180 days",
        "positive_class": 1,
        "positive_class_meaning": "return observed",
        "training_rows": int(len(df)),
        "training_return_rate": float(y.mean()),
        "threshold": 0.50,
        "threshold_note": (
            "Provisional threshold selected during V3 validation; "
            "not a calibrated business decision threshold."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("Final model trained successfully.")
    print(f"Training rows: {len(df)}")
    print(f"Observed returns: {int(y.sum())} ({y.mean():.2%})")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
