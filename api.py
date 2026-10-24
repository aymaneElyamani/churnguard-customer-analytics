
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "models" / "churnguard_model_v1.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train_final_model.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Customer return prediction based on historical Olist behavior. "
        "A positive prediction means a return was observed in the training target."
    ),
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    frequence_180j: int = Field(ge=1, le=100)
    montant_articles_180j: float = Field(ge=0, le=1_000_000)
    recence_jours: int = Field(ge=0, le=180)


@app.get("/")
def root():
    return {
        "project": "ChurnGuard",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: CustomerFeatures):
    customer = pd.DataFrame(
        [{
            "frequence_180j": features.frequence_180j,
            "montant_articles_180j": features.montant_articles_180j,
            "recence_jours": features.recence_jours,
        }]
    )

    probabilities = model.predict_proba(customer)[0]
    classes = list(model.classes_)
    return_index = classes.index(1)
    return_probability = float(probabilities[return_index])

    threshold = 0.50
    prediction = int(return_probability >= threshold)

    return {
        "prediction": prediction,
        "prediction_label": (
            "return_observed" if prediction == 1 else "no_return_observed"
        ),
        "return_probability": round(return_probability, 4),
        "threshold": threshold,
        "interpretation": (
            "Estimated score for an observed delivered return within 180 days. "
            "This is a model score, not a guarantee or a calibrated probability."
        ),
    }
