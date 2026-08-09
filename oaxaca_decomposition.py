"""
Oaxaca-Blinder Decomposition of the Global North-South ESG Gap
=================================================================
Purpose: DAES's current approach corrects the ENTIRE North-South ESG gap
(15.44 points), implicitly assuming the whole gap is bias. This is an
unfalsifiable evaluation criterion -- any correction that shrinks group
means will "succeed," even if the underlying gap were 100% legitimate.

This script uses the classical (Blinder 1973; Oaxaca 1973) threefold
decomposition to split the gap into:
  1. EXPLAINED   -- the portion attributable to groups having genuinely
                     different covariate levels (e.g. different average
                     Revenue, CarbonEmissions, firm size)
  2. UNEXPLAINED -- the portion attributable to groups being SCORED
                     differently despite having the SAME covariate levels
                     (i.e. same inputs, different ESG_Overall -- this is
                     the part that is defensibly "bias")
  3. INTERACTION -- the portion attributable to differences in
                     covariates AND coefficients jointly

Standard errors for each component are obtained via firm-level
bootstrap resampling (500 iterations), so the confidence intervals
reported are actually computed from data, not asserted.

Only real columns present in cleaned_esg.csv are used. ESG sub-scores
(Environmental/Social/Governance) and Region/Market_Type themselves are
EXCLUDED as regressors: the sub-scores are near-deterministic components
of ESG_Overall (circularity), and Region/Market_Type are the group-
defining variable itself (would be tautological to "explain" the gap
with the variable that defines the gap).
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

RANDOM_SEED = 42
N_BOOTSTRAP = 500
rng = np.random.default_rng(RANDOM_SEED)

# ── 1. LOAD DATA & DEFINE GROUPS ─────────────────────────────────
df = pd.read_csv('cleaned_esg.csv')

NORTH_REGIONS = ['Europe', 'North America', 'Oceania']
df['GeoGroup'] = np.where(df['Region'].isin(NORTH_REGIONS), 'North', 'South')

TARGET = 'ESG_Overall'

# Real, non-circular, non-tautological predictors only.
NUMERIC_FEATURES = [
    'Revenue', 'ProfitMargin', 'MarketCap', 'GrowthRate',
    'CarbonEmissions', 'WaterUsage', 'EnergyConsumption'
]
CATEGORICAL_FEATURES = ['Industry']

# Log-transform heavily skewed financial/emissions variables (common
# practice; these span several orders of magnitude in this dataset).
df_model = df.copy()
for col in ['Revenue', 'MarketCap', 'CarbonEmissions', 'WaterUsage', 'EnergyConsumption']:
    df_model[col] = np.log1p(df_model[col].clip(lower=0))

# One-hot encode Industry (drop first to avoid collinearity with intercept)
industry_dummies = pd.get_dummies(df_model['Industry'], prefix='Industry', drop_first=True)
X_full = pd.concat([df_model[NUMERIC_FEATURES], industry_dummies], axis=1).astype(float)
X_full = sm.add_constant(X_full)
y_full = df_model[TARGET].astype(float)
group = df_model['GeoGroup']

FEATURE_COLS = X_full.columns.tolist()  # includes 'const'


def fit_group_model(X, y):
    """OLS fit for a single group; returns fitted params (Series)."""
    model = sm.OLS(y, X).fit()
    return model.params


def threefold_decomposition(X, y, group):
    """
    Blinder-Oaxaca threefold decomposition of the North-South gap.
    Reference: Blinder (1973), Oaxaca (1973).
    """
    north_mask = group == 'North'
    south_mask = group == 'South'

    X_n, y_n = X[north_mask], y[north_mask]
    X_s, y_s = X[south_mask], y[south_mask]

    beta_n = fit_group_model(X_n, y_n)
    beta_s = fit_group_model(X_s, y_s)

    xbar_n = X_n.mean()
    xbar_s = X_s.mean()

    total_gap = float(y_n.mean() - y_s.mean())

    explained   = float((xbar_n - xbar_s) @ beta_s)
    unexplained = float(xbar_s @ (beta_n - beta_s))
    interaction = float((xbar_n - xbar_s) @ (beta_n - beta_s))

    # sanity check: components must sum to total gap
    reconstructed = explained + unexplained + interaction
    assert abs(reconstructed - total_gap) < 1e-6, \
        f"Decomposition does not sum to total gap: {reconstructed} vs {total_gap}"

    return {
        'total_gap': total_gap,
        'explained': explained,
        'unexplained': unexplained,
        'interaction': interaction,
    }


# ── 2. POINT ESTIMATE ────────────────────────────────────────────
print("── Fitting group-specific OLS models (North vs South) ──")
point_estimate = threefold_decomposition(X_full, y_full, group)

for k, v in point_estimate.items():
    print(f"{k:12s}: {v:+.4f}")

# ── 3. BOOTSTRAP FOR CONFIDENCE INTERVALS ────────────────────────
print(f"\n── Bootstrapping ({N_BOOTSTRAP} iterations, firm-level resampling) ──")

boot_results = {'total_gap': [], 'explained': [], 'unexplained': [], 'interaction': []}
n = len(df_model)
idx_north = np.where((group == 'North').values)[0]
idx_south = np.where((group == 'South').values)[0]

for b in range(N_BOOTSTRAP):
    # Resample within each group to preserve group sizes (standard for
    # two-sample bootstrap; avoids degenerate single-group draws)
    boot_idx_n = rng.choice(idx_north, size=len(idx_north), replace=True)
    boot_idx_s = rng.choice(idx_south, size=len(idx_south), replace=True)
    boot_idx = np.concatenate([boot_idx_n, boot_idx_s])

    X_b = X_full.iloc[boot_idx].reset_index(drop=True)
    y_b = y_full.iloc[boot_idx].reset_index(drop=True)
    g_b = group.iloc[boot_idx].reset_index(drop=True)

    try:
        res_b = threefold_decomposition(X_b, y_b, g_b)
        for k in boot_results:
            boot_results[k].append(res_b[k])
    except Exception:
        continue  # skip degenerate resamples (e.g. singular design matrix)

    if (b + 1) % 100 == 0:
        print(f"  {b + 1}/{N_BOOTSTRAP} bootstrap iterations complete")

boot_df = pd.DataFrame(boot_results)
print(f"\nValid bootstrap iterations: {len(boot_df)} / {N_BOOTSTRAP}")

# ── 4. SUMMARY TABLE WITH REAL 95% CIs ───────────────────────────
summary_rows = []
for component in ['total_gap', 'explained', 'unexplained', 'interaction']:
    point = point_estimate[component]
    ci_low, ci_high = np.percentile(boot_df[component], [2.5, 97.5])
    pct_of_gap = 100 * point / point_estimate['total_gap']
    summary_rows.append({
        'Component': component,
        'Point_Estimate': round(point, 4),
        'CI_2.5%': round(ci_low, 4),
        'CI_97.5%': round(ci_high, 4),
        'Pct_of_Total_Gap': round(pct_of_gap, 1),
    })

summary = pd.DataFrame(summary_rows)
print("\n── Oaxaca-Blinder Decomposition Summary ──")
print(summary.to_string(index=False))

os.makedirs('results', exist_ok=True)
summary.to_csv('results/oaxaca_decomposition.csv', index=False)
boot_df.to_csv('results/oaxaca_bootstrap_draws.csv', index=False)
print("\nSaved: results/oaxaca_decomposition.csv")
print("Saved: results/oaxaca_bootstrap_draws.csv")

# ── 5. DERIVE A PRINCIPLED ALPHA FROM THE UNEXPLAINED COMPONENT ──
# DAES should only correct the portion of the gap that is NOT explained
# by legitimate covariate differences. We define alpha such that DAES's
# correction, applied only to the South group, closes exactly the
# unexplained gap (not the total gap).
unexplained_gap = point_estimate['unexplained']
total_gap = point_estimate['total_gap']
south_mean = float(y_full[group == 'South'].mean())

# DAES formula: DAES_ESG = ESG * (1 + alpha * (1 - DII))
# We solve for the alpha_0 that, at the AVERAGE South DII, inflates the
# South mean by exactly unexplained_gap points (i.e. closes only the
# bias-attributable portion of the gap, leaving the explained portion
# -- which reflects real covariate differences -- untouched).
#
# DII values are read from data/dii_by_region.csv, computed by
# compute_dii.py from real World Bank Regulatory Quality data (NOT a
# hardcoded dictionary -- see compute_dii.py for the derivation).
# Run `python compute_dii.py` first if this file does not exist yet.
DII_PATH = 'data/dii_by_region.csv'
if not os.path.exists(DII_PATH):
    raise FileNotFoundError(
        f"{DII_PATH} not found. Run `python compute_dii.py` first to "
        "compute region-level DII from data/world_bank_governance.csv."
    )
dii_lookup = pd.read_csv(DII_PATH).set_index('Region')['DII']
df['DII'] = df['Region'].map(dii_lookup)
south_mean_dii = float(df.loc[df['GeoGroup'] == 'South', 'DII'].mean())

# south_mean * alpha * (1 - mean_DII) = unexplained_gap  =>  solve alpha
principled_alpha = unexplained_gap / (south_mean * (1 - south_mean_dii))

print("\n── Principled Alpha Derivation ──")
print(f"Total North-South gap        : {total_gap:.2f} points")
print(f"Unexplained (bias) component : {unexplained_gap:.2f} points ({100*unexplained_gap/total_gap:.1f}% of gap)")
print(f"Explained (legitimate) component: {point_estimate['explained']:.2f} points ({100*point_estimate['explained']/total_gap:.1f}% of gap)")
print(f"Derived alpha_0 (targets unexplained gap only): {principled_alpha:.4f}")

with open('results/oaxaca_derived_alpha.txt', 'w') as f:
    f.write(f"unexplained_gap={unexplained_gap:.4f}\n")
    f.write(f"total_gap={total_gap:.4f}\n")
    f.write(f"explained_gap={point_estimate['explained']:.4f}\n")
    f.write(f"interaction_gap={point_estimate['interaction']:.4f}\n")
    f.write(f"principled_alpha_0={principled_alpha:.4f}\n")
print("Saved: results/oaxaca_derived_alpha.txt")

# ── 6. VISUALIZATION ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
components = ['Explained\n(legitimate covariates)', 'Unexplained\n(bias)', 'Interaction']
values = [point_estimate['explained'], point_estimate['unexplained'], point_estimate['interaction']]
cis = [
    (np.percentile(boot_df['explained'], 2.5), np.percentile(boot_df['explained'], 97.5)),
    (np.percentile(boot_df['unexplained'], 2.5), np.percentile(boot_df['unexplained'], 97.5)),
    (np.percentile(boot_df['interaction'], 2.5), np.percentile(boot_df['interaction'], 97.5)),
]
errors = [[values[i] - cis[i][0] for i in range(3)], [cis[i][1] - values[i] for i in range(3)]]
colors = ['#2980b9', '#c0392b', '#7f8c8d']

bars = ax.bar(components, values, color=colors, alpha=0.85, yerr=errors, capsize=6)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('ESG Points')
ax.set_title(f'Oaxaca-Blinder Decomposition of the North-South ESG Gap\n'
             f'(Total gap = {total_gap:.2f} points, 95% CI from {N_BOOTSTRAP}-iteration bootstrap)')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + (0.3 if val >= 0 else -0.5),
            f'{val:+.2f}', ha='center', fontweight='bold')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/oaxaca_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/oaxaca_decomposition.png")

print("\nOaxaca-Blinder decomposition complete.")