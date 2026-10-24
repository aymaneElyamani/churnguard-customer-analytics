
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Charger le dataset
df = pd.read_csv("data/processed/churn_dataset_v1.csv")

features = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

X = df[features]
y = df["churn"]

# Classe positive : client revenu (churn == 0)
y_return = (y == 0).astype(int)

# Reproduire exactement la séparation de train_model_v2.py
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

y_val_return = (y_val == 0).astype(int)

# Entraîner seulement sur le train
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

# Probabilité estimée de retour
classes = list(model.named_steps["classifier"].classes_)
return_column = classes.index(0)
scores = model.predict_proba(X_val)[:, return_column]

# Classer les clients du score le plus élevé au plus faible
ranking = pd.DataFrame({
    "score_retour": scores,
    "retour_reel": y_val_return.to_numpy(),
}).sort_values("score_retour", ascending=False)

total_returns = int(ranking["retour_reel"].sum())

print("=" * 60)
print("CHURNGUARD - CLASSEMENT SUR LA VALIDATION")
print("=" * 60)
print(f"Clients en validation : {len(ranking)}")
print(f"Clients revenus       : {total_returns}")
print("\nPart ciblée | Clients ciblés | Retours détectés | Precision | Recall")

for percentage in [0.01, 0.02, 0.05, 0.10]:
    k = max(1, int(len(ranking) * percentage))
    top_clients = ranking.head(k)

    detected = int(top_clients["retour_reel"].sum())
    precision = detected / k
    recall = detected / total_returns if total_returns else 0

    print(
        f"{percentage:>6.0%}      | "
        f"{k:>6}         | "
        f"{detected:>6}          | "
        f"{precision:>7.2%}  | "
        f"{recall:>7.2%}"
    )

print("\nDistribution des scores de retour sur la validation :")
print(ranking["score_retour"].describe().round(4))
