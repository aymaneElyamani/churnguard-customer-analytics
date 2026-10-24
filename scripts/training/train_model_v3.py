
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Charger V3 : cible sur 180 jours
DATA_PATH = Path("data/processed/churn_dataset_v3.csv")
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

X = df[FEATURES]
y = df["churn"]

# Classe positive : le client revient (churn == 0)
y_return = (y == 0).astype(int)

print("=" * 60)
print("CHURNGUARD - MODEL V3 - HORIZON 180 JOURS")
print("=" * 60)
print(f"Clients : {len(df)}")
print(f"Clients revenus : {y_return.sum()}")
print(f"Taux de retour : {y_return.mean():.2%}")

# 2. Separation : 60 % train, 20 % validation, 20 % test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

y_val_return = (y_val == 0).astype(int)
y_test_return = (y_test == 0).astype(int)

print("\nRepartition :")
print(f"Train : {len(X_train)} clients, {(y_train == 0).sum()} retours")
print(f"Validation : {len(X_val)} clients, {(y_val == 0).sum()} retours")
print(f"Test : {len(X_test)} clients, {(y_test == 0).sum()} retours")

# 3. Modeles
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

dummy = DummyClassifier(strategy="most_frequent")

# L'apprentissage se fait uniquement sur train
model.fit(X_train, y_train)
dummy.fit(X_train, y_train)

classes = list(model.named_steps["classifier"].classes_)
return_column = classes.index(0)

# 4. Selection du seuil sur validation seulement
val_prob = model.predict_proba(X_val)[:, return_column]

print("\n" + "=" * 60)
print("COMPARAISON DES SEUILS - VALIDATION")
print("=" * 60)
print("Seuil | Precision | Recall | F1")

results = []

for threshold in np.arange(0.05, 0.51, 0.05):
    pred = (val_prob >= threshold).astype(int)

    precision = precision_score(
        y_val_return, pred, zero_division=0
    )
    recall = recall_score(
        y_val_return, pred, zero_division=0
    )
    f1 = f1_score(
        y_val_return, pred, zero_division=0
    )

    results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    print(
        f"{threshold:.2f}  | {precision:.3f}     | "
        f"{recall:.3f}  | {f1:.3f}"
    )

best = max(results, key=lambda r: (r["f1"], r["precision"]))
best_threshold = best["threshold"]

print(f"\nSeuil retenu : {best_threshold:.2f}")
print(f"F1 validation : {best['f1']:.4f}")

# 5. Evaluation finale sur test
test_prob = model.predict_proba(X_test)[:, return_column]
test_pred = (test_prob >= best_threshold).astype(int)

dummy_return_column = list(dummy.classes_).index(0)
dummy_prob = dummy.predict_proba(X_test)[:, dummy_return_column]
dummy_pred = (dummy.predict(X_test) == 0).astype(int)

print("\n" + "=" * 60)
print("RESULTATS FINAUX - TEST")
print("=" * 60)

print("\nBaseline Dummy :")
print(
    "Average Precision : "
    f"{average_precision_score(y_test_return, dummy_prob):.4f}"
)
print(
    "Recall retour : "
    f"{recall_score(y_test_return, dummy_pred, zero_division=0):.4f}"
)

print("\nRegression logistique :")
print(f"Seuil : {best_threshold:.2f}")
print(
    "Average Precision : "
    f"{average_precision_score(y_test_return, test_prob):.4f}"
)
print(
    "Precision retour : "
    f"{precision_score(y_test_return, test_pred, zero_division=0):.4f}"
)
print(
    "Recall retour : "
    f"{recall_score(y_test_return, test_pred, zero_division=0):.4f}"
)
print(
    "F1 retour : "
    f"{f1_score(y_test_return, test_pred, zero_division=0):.4f}"
)

print("\nMatrice de confusion")
print("Lignes = realite [sans retour, retour]")
print("Colonnes = prediction [sans retour, retour]")
print(confusion_matrix(y_test_return, test_pred, labels=[0, 1]))
