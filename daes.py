import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap

# ── 1. WORLD BANK DII DATA ────────────────────────────────────
# Disclosure Infrastructure Index (DII) per country/region
# Source: World Bank Governance Indicators 2024
import pandas as pd
wb = pd.read_csv(r'C:\Users\aarya\aiml\data\world_bank_governance.csv')
wb = wb[['Country Name', '2024 [YR2024]']].dropna()
wb.columns = ['Country', 'RegulatoryQuality']
wb['RegulatoryQuality'] = pd.to_numeric(wb['RegulatoryQuality'], errors='coerce')
wb = wb.dropna(subset=['RegulatoryQuality'])
# Normalise -2.5 to +2.5 scale → 0 to 1
wb['DII'] = (wb['RegulatoryQuality'] - wb['RegulatoryQuality'].min()) / \
            (wb['RegulatoryQuality'].max() - wb['RegulatoryQuality'].min())

# Map countries to your dataset's regions
region_country_map = {
    'Europe'       : ['Germany','France','United Kingdom','Sweden','Netherlands'],
    'North America': ['United States','Canada'],
    'Oceania'      : ['Australia','New Zealand'],
    'Latin America': ['Brazil','Mexico','Colombia','Argentina','Chile'],
    'Asia'         : ['China','India','Indonesia','Japan','South Korea'],
    'Middle East'  : ['Saudi Arabia','United Arab Emirates','Turkey','Egypt','Iran, Islamic Rep.'],
    'Africa'       : ['Nigeria','South Africa','Kenya','Ghana','Ethiopia']
}

dii_map = {}
for region, countries in region_country_map.items():
    scores = wb[wb['Country'].isin(countries)]['DII']
    dii_map[region] = round(scores.mean(), 4)

print("── Real World Bank DII Values ──")
for region, score in sorted(dii_map.items(), key=lambda x: x[1], reverse=True):
    print(f"  {region}: {score}")
# ── 2. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('cleaned_esg.csv')

# ── 3. COMPUTE DAES ───────────────────────────────────────────
# Formula: DAES = ESG_Overall × (1 + α × (1 - DII))
# α = correction strength (we test 3 values)
# Logic: weaker disclosure infrastructure = larger upward correction

df['DII'] = df['Region'].map(dii_map)

alphas = [0.1, 0.2, 0.3]
for alpha in alphas:
    col = f'DAES_alpha_{int(alpha*10)}'
    df[col] = df['ESG_Overall'] * (1 + alpha * (1 - df['DII']))

print("── DAES SCORES: Before vs After Correction ──\n")
print("Alpha = 0.1 (conservative correction):")
print(df.groupby('Region')[['ESG_Overall','DAES_alpha_1']].mean().round(2))
print("\nAlpha = 0.2 (moderate correction):")
print(df.groupby('Region')[['ESG_Overall','DAES_alpha_2']].mean().round(2))
print("\nAlpha = 0.3 (aggressive correction):")
print(df.groupby('Region')[['ESG_Overall','DAES_alpha_3']].mean().round(2))

# ── 4. VISUALISE: Before vs After (Alpha = 0.2) ───────────────
region_comparison = df.groupby('Region').agg(
    Original_ESG =('ESG_Overall',   'mean'),
    DAES_Score   =('DAES_alpha_2',  'mean')
).sort_values('Original_ESG')

north = ['Europe', 'North America', 'Oceania']
colors_orig = ['#c0392b' if r not in north else '#2980b9'
               for r in region_comparison.index]

x = np.arange(len(region_comparison))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, region_comparison['Original_ESG'],
               width, label='Original ESG Score', color=colors_orig, alpha=0.7)
bars2 = ax.bar(x + width/2, region_comparison['DAES_Score'],
               width, label='DAES Corrected Score', color=colors_orig, alpha=1.0)

ax.set_xlabel('Region')
ax.set_ylabel('Mean ESG Score')
ax.set_title('DAES Correction: Original vs Disclosure-Adjusted ESG Scores\n(Red = Global South, Blue = Global North | α = 0.2)')
ax.set_xticks(x)
ax.set_xticklabels(region_comparison.index, rotation=15)
ax.legend()
ax.axhline(df['ESG_Overall'].mean(), color='black',
           linestyle='--', alpha=0.5, label='Global Mean')
plt.tight_layout()
plt.savefig('figures/daes_before_after.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: figures/daes_before_after.png")

# ── 5. QUANTIFY THE GAP REDUCTION ─────────────────────────────
print("\n── GAP ANALYSIS: How Much Does DAES Reduce the North-South Gap? ──")
for alpha in alphas:
    col = f'DAES_alpha_{int(alpha*10)}'
    north_mean = df[df['Region'].isin(north)][col].mean()
    south_mean = df[~df['Region'].isin(north)][col].mean()
    orig_north = df[df['Region'].isin(north)]['ESG_Overall'].mean()
    orig_south = df[~df['Region'].isin(north)]['ESG_Overall'].mean()
    orig_gap   = orig_north - orig_south
    new_gap    = north_mean - south_mean
    reduction  = ((orig_gap - new_gap) / orig_gap) * 100
    print(f"α={alpha}: Original gap={orig_gap:.2f} → DAES gap={new_gap:.2f} "
          f"(reduced by {reduction:.1f}%)")

# ── 6. RETRAIN MODEL ON DAES SCORES ───────────────────────────
# Key question: does DAES reduce geographic bias in the model too?
print("\n── RETRAINING MODEL ON DAES SCORES (α=0.2) ──")

FEATURES = [
    'Industry', 'Region', 'Year', 'Revenue', 'ProfitMargin',
    'MarketCap', 'GrowthRate', 'CarbonEmissions',
    'WaterUsage', 'EnergyConsumption', 'Market_Type'
]

df_model = df[FEATURES + ['DAES_alpha_2']].copy()
le_dict = {}
for col in ['Industry', 'Region', 'Market_Type']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    le_dict[col] = le

X = df_model[FEATURES]
y = df_model['DAES_alpha_2']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

daes_model = XGBRegressor(n_estimators=200, max_depth=6,
                           learning_rate=0.1, random_state=42)
daes_model.fit(X_train, y_train)
y_pred_daes = daes_model.predict(X_test)

# SHAP on DAES model
print("Running SHAP on DAES model...")
explainer_daes   = shap.TreeExplainer(daes_model)
shap_values_daes = explainer_daes.shap_values(X_test)

mean_shap_daes = pd.DataFrame({
    'Feature'  : FEATURES,
    'Mean_SHAP': np.abs(shap_values_daes).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)

print("\n── DAES Model: Mean |SHAP| per Feature ──")
print(mean_shap_daes.to_string(index=False))

# Compare Market_Type SHAP: before vs after DAES
ablation_shap = pd.read_csv('results/ablation_shap_importance.csv')
orig_mt_shap  = ablation_shap[ablation_shap['Feature']=='Market_Type']['Mean_SHAP'].values[0]
daes_mt_shap  = mean_shap_daes[mean_shap_daes['Feature']=='Market_Type']['Mean_SHAP'].values[0]
reduction_pct = ((orig_mt_shap - daes_mt_shap) / orig_mt_shap) * 100

print(f"\n── KEY FINDING ──")
print(f"Market_Type SHAP before DAES : {orig_mt_shap:.4f}")
print(f"Market_Type SHAP after DAES  : {daes_mt_shap:.4f}")
print(f"Reduction in geographic bias : {reduction_pct:.1f}%")

# SHAP bar comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Before DAES
ax1.barh(ablation_shap['Feature'], ablation_shap['Mean_SHAP'], color='#c0392b')
ax1.set_title('Before DAES\n(Original ESG Score)')
ax1.set_xlabel('Mean |SHAP|')
ax1.invert_yaxis()

# After DAES
ax2.barh(mean_shap_daes['Feature'], mean_shap_daes['Mean_SHAP'], color='#27ae60')
ax2.set_title('After DAES Correction\n(Disclosure-Adjusted Score)')
ax2.set_xlabel('Mean |SHAP|')
ax2.invert_yaxis()

plt.suptitle('SHAP Feature Importance: Before vs After DAES Correction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/daes_shap_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: figures/daes_shap_comparison.png")

mean_shap_daes.to_csv('results/daes_shap_importance.csv', index=False)
print("Saved: results/daes_shap_importance.csv")
print("\n DAES complete.")