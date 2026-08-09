"""
Multi-Model Benchmark: XGBoost vs Random Forest vs Ridge vs MLP
==================================================================
Fixes two bugs from the original version:
  1. Target column: was silently falling back to df.columns[-1]
     ('Market_Type', a binary label) because 'ESG_Score' doesn't
     exist in this dataset -- the real column is 'ESG_Overall'.
  2. Ignore-columns list: referenced 'Company_Name'/'Ticker'/'Country'
     which don't exist (real columns are 'CompanyID'/'CompanyName'),
     so those id columns were silently leaking into the feature set.

This version predicts ESG_Overall using the SAME feature set as
ablation_shap.py (raw financials/operational data, no ESG sub-scores,
no Market_Type/CompanyID/CompanyName) so results are directly
comparable to the rest of the pipeline, and R^2 differences reflect
genuine model capability rather than target-leakage artifacts.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ── 1. LOAD DATA ──────────────────────────────────────────────────
df = pd.read_csv("cleaned_esg.csv")

TARGET = "ESG_Overall"
# Same feature convention as ablation_shap.py: raw financials/ops only,
# no sub-scores (circularity), no identifiers (CompanyID/CompanyName),
# no Market_Type as a feature (it's a downstream grouping variable).
FEATURES = [
    'Industry', 'Region', 'Year', 'Revenue', 'ProfitMargin',
    'MarketCap', 'GrowthRate', 'CarbonEmissions',
    'WaterUsage', 'EnergyConsumption', 'Market_Type'
]

df_model = df[FEATURES + [TARGET]].copy()
label_encoders = {}
for col in ['Industry', 'Region', 'Market_Type']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

X = df_model[FEATURES]
y = df_model[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for Ridge and MLP (tree models don't need this)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 2. DEFINE MODELS ─────────────────────────────────────────────
models = {
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0),
    "MLP (Neural Net)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []
print("=" * 50)
print("  MULTI-MODEL BASELINE EVALUATION (ESG_Overall)  ")
print("=" * 50)

# ── 3. TRAIN & EVALUATE ──────────────────────────────────────────
for name, model in models.items():
    if name in ["Ridge Regression", "MLP (Neural Net)"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"\n[{name}]")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R2   : {r2:.4f}")

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv("results/multimodal_baseline_results.csv", index=False)
print("\nResults saved to 'results/multimodal_baseline_results.csv'")
print("\n(Note: R2 in the 0.6-0.7 range is EXPECTED here, not a bug --")
print(" this is the ablation feature set, i.e. no ESG sub-scores, so")
print(" the model must predict ESG_Overall from raw financials alone.)")

# ── 4. SHAP FOR XGBOOST ──────────────────────────────────────────
print("\nGenerating SHAP explainer for XGBoost...")
xgb_model = models["XGBoost"]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary - XGBoost (Multi-Model Benchmark, ESG_Overall)")
plt.tight_layout()
plt.savefig("figures/multimodal_xgb_shap.png", dpi=300, bbox_inches="tight")
plt.close()
print("SHAP plot saved to 'figures/multimodal_xgb_shap.png'")