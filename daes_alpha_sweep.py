import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ── 1. LOAD DATA & DII MAPPING (computed from real World Bank data) ──
# Same loading logic as daes.py: DII comes from data/dii_by_region.csv,
# produced by compute_dii.py from real 2024 World Bank Worldwide
# Governance Indicators (Regulatory Quality, -2.5 to +2.5 scale),
# aggregated to region level and min-max normalized to 0-1.
#
# This REPLACES the previous hardcoded 7-value dii_region_map in this
# file, which used different numbers than daes.py's DII source despite
# both scripts claiming to model the same correction. Run
# `python compute_dii.py` first if data/dii_by_region.csv is missing.
df = pd.read_csv('cleaned_esg.csv')

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

print("── DII values used in this sweep (data/dii_by_region.csv) ──")
print(dii_lookup.sort_values(ascending=False))
print()

# ── 2. FEATURE DEFINITIONS & PREPROCESSING ───────────────────────
FEATURES = [
    'Industry', 'Region', 'Year', 'Revenue', 'ProfitMargin',
    'MarketCap', 'GrowthRate', 'CarbonEmissions',
    'WaterUsage', 'EnergyConsumption', 'Market_Type'
]
TARGET = 'ESG_Overall'

north_regions = ['Europe', 'North America', 'Oceania']
df['is_global_south'] = ~df['Region'].isin(north_regions)

# Encode categorical features
df_model = df[FEATURES + [TARGET, 'DII', 'is_global_south']].copy()
label_encoders = {}
for col in ['Industry', 'Region', 'Market_Type']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

X = df_model[FEATURES]
y = df_model[TARGET]
dii = df_model['DII']
is_south = df_model['is_global_south']

X_train, X_test, y_train, y_test, dii_train, dii_test, south_train, south_test = train_test_split(
    X, y, dii, is_south, test_size=0.2, random_state=42
)

# ── 3. DYNAMIC DAES ALPHA SWEEP [0.0 to 0.5] ────────────────────
alpha_range = np.linspace(0.0, 0.5, 21)
sweep_results = []

print("=" * 75)
print("  DYNAMIC DAES ALPHA SWEEP (MACRO DII DEFICIT ADJUSTMENT)  ")
print("=" * 75)

for alpha_0 in alpha_range:
    firm_alpha_train = alpha_0 * (1 - dii_train)
    y_train_daes = y_train * (1 + firm_alpha_train)

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train_daes)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    north_mean = y_pred[~south_test].mean()
    south_mean = y_pred[south_test].mean()
    signed_gap = north_mean - south_mean

    sweep_results.append({
        "alpha_0": alpha_0,
        "RMSE": rmse,
        "MAE": mae,
        "North_Mean": north_mean,
        "South_Mean": south_mean,
        "Signed_Gap": signed_gap,
        "Abs_Gap": abs(signed_gap)
    })

    print(f"Alpha_0: {alpha_0:.3f} | RMSE: {rmse:.4f} | North: {north_mean:.2f} | South: {south_mean:.2f} | Gap (N-S): {signed_gap:+.4f}")

sweep_df = pd.DataFrame(sweep_results)
os.makedirs("results", exist_ok=True)
sweep_df.to_csv("results/daes_alpha_sweep.csv", index=False)

# ── 4. PLOT PARETO FRONTIER ───────────────────────────────────────
os.makedirs("figures", exist_ok=True)
fig, ax1 = plt.subplots(figsize=(9, 5))

color = 'tab:blue'
ax1.set_xlabel('Global DAES Intensity Scaling (α₀)', fontsize=12)
ax1.set_ylabel('Geographic Score Gap (North Mean - South Mean)', color=color, fontsize=12)
ax1.plot(sweep_df['alpha_0'], sweep_df['Signed_Gap'], color=color, marker='o', linewidth=2, label='Score Gap (North - South)')
ax1.axhline(0, color='black', linestyle=':', linewidth=1.5, label='Parity Line (0 Gap)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('RMSE against Raw ESG Target', color=color, fontsize=12)
ax2.plot(sweep_df['alpha_0'], sweep_df['RMSE'], color=color, marker='s', linestyle='--', linewidth=2, label='RMSE')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Dynamic DAES Pareto Optimization: Score Parity vs RMSE', fontsize=14, pad=15)
fig.tight_layout()
plt.savefig("figures/daes_pareto_frontier.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nPareto plot saved to 'figures/daes_pareto_frontier.png'")