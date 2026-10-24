
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={
            "frequence_180j": 2,
            "montant_articles_180j": 250.0,
            "recence_jours": 45,
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["prediction_label"] in [
        "return_observed",
        "no_return_observed",
    ]
    assert 0 <= result["return_probability"] <= 1
    assert result["threshold"] == 0.5


def test_invalid_input_is_rejected():
    response = client.post(
        "/predict",
        json={
            "frequence_180j": 0,
            "montant_articles_180j": 250.0,
            "recence_jours": 45,
        },
    )

    assert response.status_code == 422
