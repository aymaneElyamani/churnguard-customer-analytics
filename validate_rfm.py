
import pandas as pd

file_path = "data/processed/customer_rfm.csv"

rfm = pd.read_csv(file_path)

print("=== DIMENSIONS ===")
print("Nombre de lignes :", len(rfm))
print("Nombre de colonnes :", len(rfm.columns))

print("\n=== VALEURS MANQUANTES ===")
print(rfm.isnull().sum())

print("\n=== IDENTIFIANTS ===")
if "customer_unique_id" in rfm.columns:
    print("Identifiants uniques :", rfm["customer_unique_id"].nunique())
    print("Doublons :", rfm["customer_unique_id"].duplicated().sum())
else:
    print("La colonne customer_unique_id est absente.")

print("\n=== SCORES RFM ===")
for column in ["R_score", "F_score", "M_score"]:
    print(f"{column} :")
    print("  Minimum :", rfm[column].min())
    print("  Maximum :", rfm[column].max())

print("\n=== SEGMENTS ===")
print(rfm["segment"].value_counts())

print("\n=== VALIDATION ===")
assert len(rfm) == 664, "Nombre de clients inattendu"
assert rfm[["R_score", "F_score", "M_score"]].notna().all().all()
assert rfm[["R_score", "F_score", "M_score"]].isin([1, 2, 3, 4, 5]).all().all()
assert rfm["segment"].notna().all()

print("Validation terminée avec succès.")