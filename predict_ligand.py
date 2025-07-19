import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Define feature names used during training
fpocket_feature_cols = [
    'score', 'druggability_score', 'number_of_alpha_spheres', 'total_sasa', 'polar_sasa', 'apolar_sasa',
    'volume', 'mean_local_hydrophobic_density', 'mean_alpha_sphere_radius', 'mean_alp_sph_solvent_access',
    'apolar_alpha_sphere_proportion', 'hydrophobicity_score', 'volume_score', 'polarity_score', 'charge_score',
    'proportion_of_polar_atoms', 'alpha_sphere_density', 'cent_of_mass___alpha_sphere_max_dist', 'flexibility'
]

# Argument: path to CSV file
parser = argparse.ArgumentParser()
parser.add_argument("--features", required=True, help="CSV file with pocket features")
args = parser.parse_args()

# Load trained models
clf_presence = joblib.load("models/ligand_presence.pkl")
clf_identity = joblib.load("models/ligand_identity.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Load features
df = pd.read_csv(args.features)

# Extract feature matrix (structural + embedding)
X_struct = df[fpocket_feature_cols].values
X_embed = df[[col for col in df.columns if col.startswith("embedding_")]].values
X = np.hstack([X_struct, X_embed])  # shape: (n_samples, 1299)
proba_presence = clf_presence.predict_proba(X)[:, 1]  # confidence it's TRUE
# Predict ligand presence
y_pred_presence = clf_presence.predict(X)
df["contains_ligand"] = y_pred_presence
X_ligand = X[y_pred_presence == 1]
proba_identity = clf_identity.predict_proba(X_ligand)
y_pred_identity = clf_identity.predict(X_ligand)

df.loc[df["contains_ligand"] == True, "ligand_pred"] = label_encoder.inverse_transform(y_pred_identity)
print(f"\n[DEBUG] Total pockets: {len(X)}, Predicted ligand-containing: {len(X_ligand)}")

print("\n=== Ligand presence prediction ===")

for i, (pred, prob) in enumerate(zip(y_pred_presence, proba_presence)):
    print(f"Pocket {i}: Contains Ligand? {'✅' if pred else '❌'} (Confidence: {prob:.2f})")
print("\n=== Ligand identity prediction (for pockets with ligand) ===")
for i, probs in enumerate(proba_identity):
    top3 = sorted(zip(label_encoder.classes_, probs), key=lambda x: -x[1])[:3]
    print(f"Pocket {i}: Predicted = {top3[0][0]} (Confidence: {top3[0][1]:.2f}) | Top 3: {top3}")

# Output
print(df[["pocket_id", "contains_ligand", "ligand_pred"]])
df.to_csv("ligand_predictions.csv", index=False)
print("[+] Saved predictions to ligand_predictions.csv")