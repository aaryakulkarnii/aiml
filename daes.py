import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap

# ── 1. LOAD DATA & DII MAPPING (computed from real World Bank data) ──
df = pd.read_csv('cleaned_esg.csv')

# DII is computed by compute_dii.py from real 2024 World Bank Worldwide
# Governance Indicators (Regulatory Quality, -2.5 to +2.5 scale),
# aggregated to region level and min-max normalized to 0-1. Run
# `python compute_dii.py` first to (re)generate data/dii_by_region.csv.
# This REPLACES a previous version of this script that used a hand-
# picked 7-value dictionary not derived from any data file, despite
# world_bank_governance.csv being present in the repo.
DII_PATH = 'data/dii_by_region.csv'
if not os.path.exists(DII_PATH):
    raise FileNotFoundError(
        f"{DII_PATH} not found. Run `python compute_dii.py` first to "
        "compute region-level DII from data/world_bank_governance.csv."
    )
dii_lookup = pd.read_csv(DII_PATH).set_index('Region')['DII']

df['DII'] = df['Region'].map(dii_lookup)
if df['DII'].isna().any():
    missing = df.loc[df['DII'].isna(), 'Region'].unique().tolist()
    raise ValueError(
        f"No computed DII value for region(s): {missing}. "
        "Check data/dii_by_region.csv covers all regions in cleaned_esg.csv."
    )

# ── 2. FEATURE DEFINITIONS ───────────────────────────────────────
FEATURES = [
    'Industry', 'Region', 'Year', 'Revenue', 'ProfitMargin',
    'MarketCap', 'GrowthRate', 'CarbonEmissions',
    'WaterUsage', 'EnergyConsumption', 'Market_Type'
]
TARGET = 'ESG_Overall'

north_regions = ['Europe', 'North America', 'Oceania']
df['is_global_south'] = ~df['Region'].isin(north_regions)

# ── 3. DYNAMIC DAES CORRECTION ────────────────────────────────────
# alpha_0 = 0.275 is the accuracy-fairness knee point on the RMSE-vs-
# alpha sweep (results/daes_alpha_sweep.csv), verified with the kneedle
# algorithm (Satopaa et al. 2011) -- see daes_alpha_sweep.py. This is
# the CONSERVATIVE operating value: it closes most of the gap while
# limiting RMSE degradation.
#
# For reference, oaxaca_decomposition.py independently derives
# alpha_0 = 0.431 (computed using the same real World Bank-derived DII
# values as above) as the value that would close the ENTIRE unexplained
# (bias-attributable) component of the North-South gap, per a Blinder-
# Oaxaca (1973) decomposition. The two values disagree because closing
# the full unexplained gap costs more model RMSE than the kneedle-
# optimal point considers worthwhile -- see results/oaxaca_derived_alpha.txt
# and the paper's Discussion section for this trade-off.
OPTIMAL_ALPHA_0 = 0.275

firm_alpha = OPTIMAL_ALPHA_0 * (1 - df['DII'])
df['DAES_ESG'] = df[TARGET] * (1 + firm_alpha)

print("── DYNAMIC DAES SCORES: Before vs After Correction ──\n")
region_summary = df.groupby('Region')[[TARGET, 'DAES_ESG']].mean().round(2)
print(region_summary)

# ── 4. VISUALISE: Before vs After ────────────────────────────────
region_comparison = df.groupby('Region').agg(
    Original_ESG =(TARGET,     'mean'),
    DAES_Score   =('DAES_ESG',  'mean')
).sort_values('Original_ESG')

colors_orig = ['#c0392b' if r not in north_regions else '#2980b9' for r in region_comparison.index]
x = np.arange(len(region_comparison))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, region_comparison['Original_ESG'], width, label='Original ESG Score', color=colors_orig, alpha=0.7)
bars2 = ax.bar(x + width/2, region_comparison['DAES_Score'], width, label='Dynamic DAES Score', color=colors_orig, alpha=1.0)

ax.set_xlabel('Region')
ax.set_ylabel('Mean ESG Score')
ax.set_title(f'Dynamic DAES Correction: Original vs Disclosure-Adjusted Scores\n(Red = Global South, Blue = Global North | α₀ = {OPTIMAL_ALPHA_0})')
ax.set_xticks(x)
ax.set_xticklabels(region_comparison.index, rotation=15)
ax.legend()
ax.axhline(df[TARGET].mean(), color='black', linestyle='--', alpha=0.5, label='Global Mean')

os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/daes_before_after.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: figures/daes_before_after.png")

# ── 5. RETRAIN XGBOOST MODEL ON DYNAMIC DAES TARGET ────────────────
df_model = df[FEATURES + ['DAES_ESG']].copy()
label_encoders = {}
for col in ['Industry', 'Region', 'Market_Type']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

X = df_model[FEATURES]
y = df_model['DAES_ESG']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

daes_model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
daes_model.fit(X_train, y_train)

y_pred_daes = daes_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_daes))
mae  = mean_absolute_error(y_test, y_pred_daes)
r2   = r2_score(y_test, y_pred_daes)

print("\n── DYNAMIC DAES MODEL EVALUATION ──")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ── 6. SHAP FEATURE IMPORTANCE & BIAS REDUCTION ───────────────────
print("\nRunning SHAP on Dynamic DAES model...")
explainer_daes   = shap.TreeExplainer(daes_model)
shap_values_daes = explainer_daes.shap_values(X_test)

mean_shap_daes = pd.DataFrame({
    'Feature'  : FEATURES,
    'Mean_SHAP': np.abs(shap_values_daes).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

os.makedirs('results', exist_ok=True)
mean_shap_daes.to_csv('results/daes_shap_importance.csv', index=False)
print("Saved: results/daes_shap_importance.csv")

if os.path.exists('results/ablation_shap_importance.csv'):
    ablation_shap = pd.read_csv('results/ablation_shap_importance.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(ablation_shap['Feature'], ablation_shap['Mean_SHAP'], color='#c0392b')
    ax1.set_title('Before DAES Correction\n(Raw Target)')
    ax1.set_xlabel('Mean |SHAP|')
    ax1.invert_yaxis()

    ax2.barh(mean_shap_daes['Feature'], mean_shap_daes['Mean_SHAP'], color='#27ae60')
    ax2.set_title('After Dynamic DAES Correction\n(Adjusted Target)')
    ax2.set_xlabel('Mean |SHAP|')
    ax2.invert_yaxis()

    plt.suptitle('SHAP Feature Importance: Before vs After Dynamic DAES', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/daes_shap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/daes_shap_comparison.png")

print("\nDynamic DAES script execution complete.")