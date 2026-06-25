import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('cleaned_esg.csv')

# ── 2. ABLATION: DROP the 3 sub-scores (this is the key change)
# We are forcing the model to predict ESG_Overall from RAW company
# data only — no sub-scores allowed. This is where the real bias shows.
FEATURES = [
    'Industry', 'Region', 'Year', 'Revenue', 'ProfitMargin',
    'MarketCap', 'GrowthRate', 'CarbonEmissions',
    'WaterUsage', 'EnergyConsumption', 'Market_Type'
]
TARGET = 'ESG_Overall'

# ── 3. ENCODE categorical columns (Region, Industry, Market_Type)
df_model = df[FEATURES + [TARGET]].copy()
label_encoders = {}
for col in ['Industry', 'Region', 'Market_Type']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le  # save for reference

# ── 4. TRAIN / TEST SPLIT
X = df_model[FEATURES]
y = df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 5. TRAIN XGBOOST (ablation version)
model = XGBRegressor(n_estimators=200, max_depth=6,
                     learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ── 6. EVALUATE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
print(f"\n── ABLATION MODEL PERFORMANCE ──")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print("(R² will be lower than before — that's expected and good!)\n")

# ── 7. SHAP ANALYSIS
print("Running SHAP... (may take a minute)")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot (Beeswarm)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary — Ablation (No Sub-Scores)")
plt.tight_layout()
plt.savefig('figures/ablation_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/ablation_shap_summary.png")

# SHAP Bar Plot (Mean absolute importance)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.title("SHAP Feature Importance — Ablation")
plt.tight_layout()
plt.savefig('figures/ablation_shap_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/ablation_shap_bar.png")

# ── 8. GEOGRAPHIC FAIRNESS ON ABLATION MODEL
df_test = X_test.copy()
df_test['Actual_ESG']    = y_test.values
df_test['Predicted_ESG'] = y_pred
df_test['Absolute_Error']= np.abs(y_test.values - y_pred)

# Decode Region back to names for readability
df_test['Region_Name'] = label_encoders['Region'].inverse_transform(df_test['Region'])
df_test['Market_Name'] = label_encoders['Market_Type'].inverse_transform(df_test['Market_Type'])

# North vs South grouping
north = ['Europe', 'North America', 'Oceania']
df_test['GeoGroup'] = df_test['Region_Name'].apply(
    lambda x: 'Global North' if x in north else 'Global South'
)

print("\n── ABLATION: ESG Score by Region ──")
region_summary = df_test.groupby('Region_Name').agg(
    Actual_ESG_Mean   =('Actual_ESG',     'mean'),
    Predicted_ESG_Mean=('Predicted_ESG',  'mean'),
    MAE               =('Absolute_Error', 'mean')
).round(4)
print(region_summary)

print("\n── ABLATION: ESG Score by Market Type ──")
market_summary = df_test.groupby('Market_Name').agg(
    Actual_ESG_Mean   =('Actual_ESG',     'mean'),
    Predicted_ESG_Mean=('Predicted_ESG',  'mean'),
    MAE               =('Absolute_Error', 'mean')
).round(4)
print(market_summary)

print("\n── ABLATION: Global North vs Global South ──")
geo_summary = df_test.groupby('GeoGroup').agg(
    Actual_ESG_Mean   =('Actual_ESG',     'mean'),
    Predicted_ESG_Mean=('Predicted_ESG',  'mean'),
    MAE               =('Absolute_Error', 'mean')
).round(4)
print(geo_summary)

# ── 9. MEAN SHAP VALUE PER FEATURE (for paper table)
mean_shap = pd.DataFrame({
    'Feature'   : FEATURES,
    'Mean_SHAP' : np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)
print("\n── ABLATION: Mean |SHAP| per Feature ──")
print(mean_shap.to_string(index=False))
mean_shap.to_csv('results/ablation_shap_importance.csv', index=False)
print("\nSaved: results/ablation_shap_importance.csv")

# ── 10. ESG Gap Bar Chart by Region
region_plot = df_test.groupby('Region_Name')['Predicted_ESG'].mean().sort_values()
plt.figure(figsize=(10, 5))
colors = ['#c0392b' if r not in north else '#2980b9' for r in region_plot.index]
bars = plt.bar(region_plot.index, region_plot.values, color=colors)
plt.axhline(region_plot.mean(), color='black', linestyle='--', label='Global Mean')
plt.title('Ablation Model: Predicted ESG Score by Region\n(Red = Global South, Blue = Global North)')
plt.ylabel('Mean Predicted ESG Score')
plt.xlabel('Region')
plt.legend()
plt.tight_layout()
plt.savefig('figures/ablation_esg_region.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/ablation_esg_region.png")

print("\n✅ Ablation complete. Check figures/ and results/ for outputs.")