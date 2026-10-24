
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/churn_dataset_v1.csv")
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

# L'identifiant ne doit pas servir de variable explicative.
X = df[
    [
        "frequence_180j",
        "montant_articles_180j",
        "recence_jours",
        "panier_moyen_180j",
    ]
]
y = df["churn"]

print("Répartition globale :")
print(y.value_counts())
print("\nTaux de retour :", round((y == 0).mean() * 100, 2), "%")

# Garder les proportions des classes dans les deux ensembles.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

models = {
    "Baseline majoritaire": DummyClassifier(strategy="most_frequent"),
    "Régression logistique": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # La classe positive métier est le retour client (churn = 0).
    # On calcule donc le recall sur cette classe explicitement.
    recall_return = recall_score(
        y_test, predictions, pos_label=0, zero_division=0
    )

    print(f"\n{'=' * 55}")
    print(name)
    print("Recall des clients qui reviennent :", round(recall_return, 4))
    print(
        "PR-AUC des clients qui reviennent :",
        round(
            average_precision_score(
                (y_test == 0).astype(int),
                model.predict_proba(X_test)[:, list(model.classes_).index(0)]
                if hasattr(model, "classes_")
                else model.predict_proba(X_test)[:, 1],
            ),
            4,
        ),
    )
    print("Matrice de confusion (ordre des classes : 0, 1) :")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Retour (0)", "Sans retour (1)"],
            zero_division=0,
        )
    )
