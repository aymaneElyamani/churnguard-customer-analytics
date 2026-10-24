
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/churn_dataset_v3.csv")

FEATURES = [
    "frequence_180j",
    "montant_articles_180j",
    "recence_jours",
    "panier_moyen_180j",
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["churn"]

# Reproduire exactement la separation de train_model_v3.py
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )),
])

model.fit(X_train, y_train)

# La classe positive de notre analyse est le retour : churn == 0
classes = list(model.named_steps["classifier"].classes_)
return_column = classes.index(0)

scores = model.predict_proba(X_test)[:, return_column]

results = pd.DataFrame({
    "score_retour": scores,
    "retour_reel": (y_test == 0).astype(int).to_numpy(),
})

results = results.sort_values(
    "score_retour", ascending=False
).reset_index(drop=True)

total_returns = int(results["retour_reel"].sum())
total_clients = len(results)
baseline_rate = results["retour_reel"].mean()

print("=" * 65)
print("CHURNGUARD V3 - ANALYSE TOP-K SUR LE TEST")
print("=" * 65)
print(f"Clients dans le test : {total_clients}")
print(f"Retours reels : {total_returns}")
print(f"Taux de retour global : {baseline_rate:.2%}")

for pct in [1, 2, 5, 10]:
    k = max(1, int(total_clients * pct / 100))
    top = results.head(k)

    found = int(top["retour_reel"].sum())
    precision = found / k
    recall = found / total_returns if total_returns else 0
    lift = precision / baseline_rate if baseline_rate else 0

    print(f"\nTop {pct}% : {k} clients")
    print(f"Retours identifies : {found}")
    print(f"Precision : {precision:.2%}")
    print(f"Rappel des retours : {recall:.2%}")
    print(f"Lift par rapport au taux global : {lift:.2f}x")

print("\nAnalyse terminee. Aucun fichier modifie.")
