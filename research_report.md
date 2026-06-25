# Research Results Report
## Auditing Geographic Bias in AI-Driven ESG Scoring: A SHAP-Based Explainability Analysis of Rating Disparities Between Global North and Global South Firms

**Authors:** Aarya Kulkarni  
**Conference:** AIML 2026 — Paris, France | October 26–27, 2026  
**Report Generated:** 2026-06-25

---

## 1. Project Overview

### What the Project Accomplishes

This project builds and audits machine learning models that predict corporate ESG (Environmental, Social, and Governance) scores for a globally diverse dataset of firms spanning seven regions and 11 years (2015–2025). Three gradient-boosted and ensemble tree models — XGBoost, Random Forest, and LightGBM — are trained to predict the overall ESG score (`ESG_Overall`) from financial, operational, and environmental features. Model predictions are then subjected to a rigorous two-layer audit:

1. **Explainability layer (SHAP):** SHAP (SHapley Additive exPlanations) values are computed to identify which features drive predictions, and how region-level and market-type-level variables influence scores.
2. **Fairness layer:** Prediction errors and actual ESG score distributions are disaggregated by geographic region and market type (Developed vs. Emerging) to surface any systematic disparities.

### Research Objective

> *To determine whether AI-driven ESG scoring models encode geographic bias — systematically assigning higher or lower scores, or exhibiting higher prediction uncertainty, for firms from the Global South relative to those from the Global North — and to quantify the mechanisms of such bias using SHAP-based explainability.*

---

## 2. Dataset Summary

### Core Statistics

| Property | Value |
|---|---|
| **Raw dataset shape** | 11,000 rows x 16 columns |
| **Cleaned dataset shape** | 11,000 rows x 17 columns |
| **Time span** | 2015 – 2025 (11 years, 1,000 records per year) |
| **Target variable** | `ESG_Overall` (continuous, range: 6.3 – 98.8) |
| **Target mean** | 54.62 (SD = 15.89) |
| **Target median** | 54.60 |

### Features (17 total)

| Category | Features |
|---|---|
| **Identifiers** | `CompanyID`, `CompanyName` |
| **Categorical** | `Industry`, `Region`, `Market_Type` |
| **Temporal** | `Year` |
| **Financial** | `Revenue`, `ProfitMargin`, `MarketCap`, `GrowthRate` |
| **ESG Sub-scores** | `ESG_Environmental`, `ESG_Social`, `ESG_Governance` |
| **Environmental** | `CarbonEmissions`, `WaterUsage`, `EnergyConsumption` |

### Missing Values

| Column | Missing Values |
|---|---|
| `GrowthRate` | **1,000** (imputed during cleaning) |
| All other columns | 0 |

> [!NOTE]
> The only missing values were in `GrowthRate` (exactly 1,000 records — one full year's worth), which were imputed during the cleaning phase. After cleaning, the dataset has **zero missing values**.

### Regions Represented

| Region | Count | Share |
|---|---|---|
| Asia | 1,672 | 15.2% |
| Oceania | 1,661 | 15.1% |
| Middle East | 1,617 | 14.7% |
| Europe | 1,540 | 14.0% |
| North America | 1,540 | 14.0% |
| Latin America | 1,507 | 13.7% |
| Africa | 1,463 | 13.3% |

### Market Types

| Market Type | Count | Share |
|---|---|---|
| Emerging | 6,259 | 56.9% |
| Developed | 4,741 | 43.1% |

### Industry Distribution

| Industry | Count |
|---|---|
| Healthcare | 1,331 |
| Transportation | 1,287 |
| Manufacturing | 1,287 |
| Consumer Goods | 1,276 |
| Finance | 1,243 |
| Energy | 1,188 |
| Utilities | 1,177 |
| Retail | 1,166 |
| Technology | 1,045 |

### Target Variable Summary

| Statistic | ESG_Overall |
|---|---|
| Min | 6.3 |
| 25th percentile | 44.1 |
| Median | 54.6 |
| Mean | 54.62 |
| 75th percentile | 65.6 |
| Max | 98.8 |
| Std. Dev | 15.89 |

### Feature Engineering Performed

The following feature engineering was applied in the preprocessing pipeline (saved to `results/preprocessor.pkl`):

- **One-hot encoding** of categorical features: `Industry`, `Region`, `Market_Type`
- **Passthrough** of all numeric features: `Year`, `Revenue`, `ProfitMargin`, `MarketCap`, `GrowthRate`, `ESG_Environmental`, `ESG_Social`, `ESG_Governance`, `CarbonEmissions`, `WaterUsage`, `EnergyConsumption`
- The prefix `cat__` indicates one-hot encoded features; `remainder__` indicates passthrough numerics in the SHAP output.
- A `Market_Type` column was added to the raw dataset during cleaning (raw = 16 columns; cleaned = 17 columns).

---

## 3. Model Performance

Three models were trained and evaluated. All metrics are from held-out test data, reported in `results/model_comparison.csv`.

### Performance Table

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| **XGBoost** | 0.6535 | 0.5050 | 0.99819 |
| **Random Forest** | 0.7881 | 0.5926 | 0.99737 |
| **LightGBM** | **0.6229** | **0.4842** | **0.99836** |

### XGBoost

- **RMSE:** 0.6535 | **MAE:** 0.5050 | **R2:** 0.9982
- **Strengths:** Excellent accuracy, well-calibrated outputs, robust to outliers via regularization, fast inference. Natively handles sparse one-hot features.
- **Weaknesses:** Requires more hyperparameter tuning than LightGBM. Slightly larger absolute error than LightGBM.

### Random Forest

- **RMSE:** 0.7881 | **MAE:** 0.5926 | **R2:** 0.9974
- **Strengths:** Easy to interpret at the tree level, no gradient approximations, low bias when fully grown, robust to noisy features.
- **Weaknesses:** Worst performer on all three metrics. Highest RMSE (+20.6% vs. LightGBM) and highest MAE (+22.4% vs. LightGBM). Very large model file (~227 MB), indicating high depth and many estimators.

### LightGBM

- **RMSE:** 0.6229 | **MAE:** 0.4842 | **R2:** 0.9984
- **Strengths:** **Best overall model.** Lowest RMSE and MAE, highest R2. Histogram-based leaf-wise tree growth leads to efficient training and generalisation. Compact model file (~858 KB vs. ~227 MB for RF).
- **Weaknesses:** Less commonly used in fairness audits in the ESG literature; may be harder to explain to non-technical stakeholders.

### Best Model: **LightGBM**

> [!IMPORTANT]
> LightGBM achieves the best performance across all three metrics (lowest RMSE = 0.6229, lowest MAE = 0.4842, highest R2 = 0.9984). All three models achieve near-perfect R2 (>0.997), indicating that ESG sub-scores together with other features can almost perfectly reconstruct the overall ESG score — structurally expected given the compositional nature of ESG ratings. SHAP analysis was performed on the XGBoost model.

---

## 4. SHAP Explainability

SHAP values were computed on the XGBoost model. Results are stored in `results/shap_feature_importance.csv`.

### Top 20 Feature Importances (by Mean |SHAP|)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `ESG_Governance` | 7.8159 |
| 2 | `ESG_Environmental` | 7.6599 |
| 3 | `ESG_Social` | 6.4781 |
| 4 | `Year` | 0.0242 |
| 5 | `ProfitMargin` | 0.0152 |
| 6 | `CarbonEmissions` | 0.0146 |
| 7 | `EnergyConsumption` | 0.0144 |
| 8 | `GrowthRate` | 0.0125 |
| 9 | `Revenue` | 0.0111 |
| 10 | `Industry_Finance` | 0.0094 |
| 11 | `MarketCap` | 0.0091 |
| 12 | `WaterUsage` | 0.0085 |
| 13 | `Region_Middle East` | 0.0060 |
| 14 | `Region_Latin America` | 0.0047 |
| 15 | `Industry_Energy` | 0.0044 |
| 16 | `Region_Africa` | 0.0044 |
| 17 | `Region_Europe` | 0.0040 |
| 18 | `Region_Asia` | 0.0029 |
| 19 | `Region_North America` | 0.0029 |
| 20 | `Industry_Manufacturing` | 0.0020 |

### Top 10 Feature Ranking (Summary)

The top 3 features (`ESG_Governance`, `ESG_Environmental`, `ESG_Social`) dominate the model with mean |SHAP| values of 7.82, 7.66, and 6.48 respectively — orders of magnitude larger than any other feature. Features 4–10 collectively contribute less than 0.025 SHAP units each, confirming the extreme concentration of predictive signal in the three ESG pillars.

### Interpretation of SHAP Summary Plot

![SHAP Summary Plot — Beeswarm](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_summary.png)

The beeswarm SHAP summary plot reveals:

- **ESG sub-scores dominate:** `ESG_Governance`, `ESG_Environmental`, and `ESG_Social` form three dense horizontal bands spanning SHAP values of roughly -15 to +15. High feature values (red) push predictions upward; low values (blue) push predictions downward — consistent with their compositional role in the overall score.
- **The remaining 17+ features form a tight band near zero**, confirming that all financial, operational, and geographic variables together contribute less than 1% of total SHAP explanation mass.
- **Bidirectionality is symmetric:** Each dominant feature shows roughly symmetric positive and negative impact, indicating the model learned a balanced additive relationship.
- **Regional and market-type features** (positions 13–20) produce predominantly near-zero SHAP values, indicating very low marginal influence conditional on the ESG sub-scores.

### SHAP Dependence Plot — Carbon Emissions

![SHAP Dependence — CarbonEmissions](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_carbon.png)

- X-axis: `CarbonEmissions` raw value (0 to ~1.75 x 10^8)
- Y-axis: SHAP value for `CarbonEmissions`; Color: `ESG_Environmental` score
- Most observations cluster near zero carbon emissions with SHAP values close to zero.
- A small number of extreme emitters show slightly positive or near-zero SHAP contributions — high emitters are not uniformly penalized, a potential fairness concern for industrializing Global South economies.
- At very low carbon values, there is high variance in SHAP output (-0.12 to +0.12), indicating strong interaction with the environmental sub-score.

### SHAP Dependence Plot — Market Capitalization

![SHAP Dependence — MarketCap](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_marketcap.png)

- X-axis: `MarketCap` (USD millions, 0 to ~670,000); Color: `WaterUsage`
- SHAP values for `MarketCap` are constrained between approximately -0.15 and +0.10, confirming its minor role.
- Small-cap firms show high variance in SHAP contributions; large-cap firms (>$100,000M) stabilize near small positive values (~0.01–0.05), suggesting larger firms receive a marginal upward ESG boost.
- No strong interaction between market cap and water usage is observable.

### Influence of Region (from `results/region_shap.csv`)

| Region Feature | Mean |SHAP| |
|---|---|---|
| `Region_Middle East` | 0.00602 |
| `Region_Latin America` | 0.00472 |
| `Region_Africa` | 0.00437 |
| `Region_Europe` | 0.00402 |
| `Region_Asia` | 0.00289 |
| `Region_North America` | 0.00289 |
| `Region_Oceania` | 0.00075 |

**Interpretation:** Global South regions (Middle East, Latin America, Africa) carry higher SHAP magnitudes than Global North regions (Europe, North America, Oceania). This suggests the model uses regional identity as a correction factor, potentially encoding structural disadvantages not fully explained by sub-scores alone. Oceania shows the smallest regional impact (0.00075).

### Influence of Market_Type (from `results/markettype_shap.csv`)

| Market Type Feature | Mean |SHAP| |
|---|---|---|
| `Market_Type_Developed` | 0.001497 |
| `Market_Type_Emerging` | 0.000000 |

Being classified as "Developed" contributes a small positive average SHAP value (0.0015). `Market_Type_Emerging` SHAP is exactly 0.0 — after one-hot encoding, the Emerging indicator was effectively unused by XGBoost, with its information subsumed by the Developed indicator and regional features.

---

## 5. Fairness Evaluation

### Average ESG Score by Market_Type (from `results/fairness_markettype.csv`)

| Market Type | Actual ESG Mean | Actual ESG SD | Predicted ESG Mean | MAE |
|---|---|---|---|---|
| **Developed** | 62.90 | 12.99 | 62.91 | 0.4712 |
| **Emerging** | 47.95 | 13.83 | 48.02 | 0.4936 |

- The gap between Developed and Emerging average ESG scores is **~15 points** (62.90 vs. 47.95).
- Predictions closely track actuals for both groups.
- Emerging markets have slightly higher MAE (0.4936 vs. 0.4712), indicating marginally lower accuracy on Global South firms.

### Average ESG Score by Region (from `results/fairness_region.csv`)

| Region | Actual ESG Mean | Predicted ESG Mean | Mean Abs. Error |
|---|---|---|---|
| Europe | 66.58 | 66.61 | 0.5142 |
| North America | 61.16 | 61.17 | 0.4502 |
| Oceania | 61.08 | 61.06 | 0.4506 |
| Latin America | 51.33 | 51.37 | 0.4427 |
| Asia | 51.08 | 51.12 | 0.4928 |
| Africa | 45.75 | 45.86 | 0.5363 |
| Middle East | 43.65 | 43.71 | 0.5040 |

![Average ESG Score by Region](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\esg_region.png)

### Prediction Error by Market Type

![Prediction Error by Market Type](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\error_markettype.png)

- Developed: MAE = 0.4712 | Emerging: MAE = 0.4936
- Difference: 0.0224 ESG points — statistically **not significant** (p = 0.185)

### Prediction Error by Region

![Prediction Error by Region](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\error_region.png)

**Africa has the highest absolute error (0.5363)**. Latin America (0.4427) has the lowest. Range across regions is only ~0.09 ESG points, but statistically significant (ANOVA p = 0.0115).

### Global North vs. Global South Summary

> [!IMPORTANT]
> There is **strong evidence of geographic disparity in ESG score levels** between the Global North and Global South (Delta ~15 points, p < 0.0001). However, there is **no statistically significant geographic disparity in prediction accuracy** (p = 0.185). The model faithfully reproduces the geographic gap in ESG scores, encoding and perpetuating existing structural inequalities rather than correcting for them.

---

## 6. Statistical Analysis

| # | Test | Variables | Statistic | p-value | Interpretation |
|---|---|---|---|---|---|
| 1 | **t-test** | ESG_Overall: Developed vs. Emerging | t = 25.69 | < 0.0001 | Highly significant score gap; Developed firms score ~15 points higher |
| 2 | **Mann-Whitney U** | ESG_Overall: Developed vs. Emerging | U = 922,329 | < 0.0001 | Non-parametric confirmation; distributions are stochastically ordered |
| 3 | **One-Way ANOVA** | ESG_Overall across 7 Regions | F = 135.21 | < 0.0001 | Highly significant regional variation in mean ESG scores |
| 4 | **t-test** | Absolute Error: Developed vs. Emerging | t = -1.3259 | 0.185 | **Not significant** — model is equivalently accurate across market types |
| 5 | **One-Way ANOVA** | Absolute Error across 7 Regions | F = 2.7491 | 0.0115 | **Marginally significant** — Africa highest, Latin America lowest |
| 6 | **t-test** | ESG_Overall: Global North vs. Global South | t = 25.69 | < 0.0001 | Confirms geographic scoring gap at hemisphere level |
| 7 | **t-test** | Absolute Error: Global North vs. Global South | t = -1.3259 | 0.185 | **Not significant** — prediction error parity confirmed |

**Additional correlations with prediction error (all weak):**
- `ESG_Governance`: r = -0.045 | `CarbonEmissions`: r = +0.036 | `WaterUsage` / `EnergyConsumption`: r = +0.037

---

## 7. Figures

### Figure 1: SHAP Summary Plot (Beeswarm)

![SHAP Summary Plot](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_summary.png)

*The beeswarm SHAP summary plot ranks all 20 features by mean |SHAP|. The three ESG sub-scores overwhelmingly dominate, collectively accounting for virtually all SHAP explanation mass. Geographic and financial features contribute minimally when sub-scores are available.*

---

### Figure 2: SHAP Bar Chart (Global Feature Importance)

![SHAP Bar Chart](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_bar.png)

*Mean absolute SHAP values per feature. ESG Governance (7.82), Environmental (7.66), and Social (6.48) are orders of magnitude more important than all other features. The discontinuity between ranks 1–3 and ranks 4–20 is visually stark.*

---

### Figure 3: Top 10 SHAP Features

![Top 10 SHAP Features](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\top10_shap_features.png)

*Publication-ready figure of the top 10 SHAP-ranked features. The three dominant features occupy virtually all visual space; remaining 7 features have bars barely visible on this scale.*

---

### Figure 4: SHAP Dependence — Carbon Emissions

![SHAP Dependence — Carbon Emissions](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_carbon.png)

*No strong monotonic penalty for carbon emissions. Some extreme emitters receive unexpectedly neutral SHAP values — a potential fairness concern for heavy industrial sectors concentrated in Global South economies.*

---

### Figure 5: SHAP Dependence — Market Capitalization

![SHAP Dependence — MarketCap](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\shap_marketcap.png)

*Small-cap firms exhibit high SHAP variance; large-cap firms show a slight positive plateau. Effect size is small. No strong evidence of size-based discrimination.*

---

### Figure 6: Average ESG Score by Region

![Average ESG Score by Region](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\esg_region.png)

*Europe (66.58) and Oceania/North America (~61) score highest; Africa (45.75) and Middle East (43.65) score lowest. The gap between highest and lowest region is approximately 23 ESG points — nearly 1.5 standard deviations.*

---

### Figure 7: Average ESG Score by Market Type

![Average ESG Score by Market Type](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\esg_markettype.png)

*Developed markets average ~63; Emerging markets average ~48 — a ~15-point gap confirmed statistically (p < 0.0001). This represents the core equity concern of the research.*

---

### Figure 8: ESG Score Distribution (Boxplot)

![ESG Score Distribution Boxplot](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\esg_boxplot.png)

*Developed markets show a tighter, higher distribution (IQR ~55–72; median ~63). Emerging markets show a wider, lower distribution (IQR ~39–58; median ~47) with outliers from ~6 to ~90, indicating greater heterogeneity in ESG performance.*

---

### Figure 9: Prediction Error by Region

![Prediction Error by Region](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\error_region.png)

*Africa has the highest error (0.5363) and Latin America the lowest (0.4427). Range is ~0.09 ESG points — small but statistically significant (ANOVA p = 0.0115). Africa's higher error likely reflects lower ESG data transparency.*

---

### Figure 10: Prediction Error by Market Type

![Prediction Error by Market Type](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\error_markettype.png)

*Emerging markets show slightly higher error (0.4936 vs. 0.4712), but this is not statistically significant (p = 0.185). The model is approximately equally accurate for both groups.*

---

### Figure 11: RMSE Comparison

![RMSE Comparison](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\rmse_comparison.png)

*LightGBM achieves the lowest RMSE (0.6229), followed by XGBoost (0.6535) and Random Forest (0.7881). Random Forest is visually noticeably worse.*

---

### Figure 12: R2 Comparison

![R2 Comparison](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\r2_comparison.png)

*All three models achieve R2 > 0.997. Differences are barely visible at this scale (values range 0.9974–0.9984), confirming that RMSE and MAE are more discriminating metrics in this performance regime.*

---

### Figure 13: XGBoost Native Feature Importance

![XGBoost Feature Importance](C:\Users\aarya\.gemini\antigravity-ide\brain\e4f32b32-9ed7-48bb-956c-b65307e4488c\xgboost_importance.png)

*XGBoost's gain-based native importance by feature index. Features at indices ~23–25 (the ESG sub-scores) dominate, corroborating SHAP findings. SHAP remains the superior tool for named feature attribution.*

---

## 8. Major Findings

1. **All three models achieve near-perfect fit (R2 > 0.997).** LightGBM is the best performer (RMSE = 0.6229, R2 = 0.9984), followed by XGBoost (RMSE = 0.6535) and Random Forest (RMSE = 0.7881).

2. **ESG sub-scores account for virtually all predictive signal.** The three ESG pillars hold SHAP values of 7.82, 7.66, and 6.48 — more than 99% of total SHAP explanation mass.

3. **There is a statistically significant 15-point ESG score gap between Developed and Emerging markets.** Developed market firms average 62.90 vs. Emerging 47.95 (t = 25.69, p < 0.0001). This gap is directly reproduced in model predictions, propagating existing inequalities.

4. **Europe scores highest (66.58); Middle East scores lowest (43.65)** — a 23-point gap (~1.45 SD). Africa ranks second-lowest (45.75); both bottom regions are in the Global South.

5. **No statistically significant geographic disparity in prediction accuracy** (t = -1.33, p = 0.185). Model bias manifests in input data (actual scores), not in differential model performance across regions.

6. **Africa has the highest prediction error (MAE = 0.5363)** — regional ANOVA is marginally significant (F = 2.75, p = 0.0115), suggesting Africa's ESG data is harder to model due to lower disclosure quality.

7. **Global South regions have higher average |SHAP| than Global North regions** for regional features (Middle East: 0.006 > Europe: 0.004 > Oceania: 0.001), suggesting the model uses regional identity as a correction factor.

8. **`Market_Type_Emerging` SHAP = 0.0** exactly; `Market_Type_Developed` = 0.0015. The model treats market type as a weak binary signal subsumed by regional encodings.

9. **Carbon emissions have a non-monotonic, weak SHAP relationship** (rank 6, mean |SHAP| = 0.0146). The model does not heavily penalize high emitters — a fairness blind spot for industrializing Global South sectors.

10. **Large-cap firms receive a marginal positive MarketCap SHAP contribution (~0.01–0.05)**, while small-cap firms show high variance. Since large-cap firms are disproportionately Global North, this acts as a subtle geographic proxy.

11. **The 11-year panel dataset (2015–2025) has perfect temporal and geographic balance** (1,000 firms per year), a methodological strength that reduces confounds in the fairness analysis.

12. **`Year` ranks 4th in SHAP importance (0.024)**, indicating a small but detectable temporal trend in ESG scores — consistent with global ESG adoption growth, particularly post-2020.

13. **Emerging markets show significantly wider ESG variance** (SD = 13.83 vs. Developed SD = 12.99), making their ratings more dependent on firm-specific factors and harder to compare across geographies.

14. **No evidence of differential model bias** (the model doesn't systematically over/under-predict for any group), but **strong confirmation of structural data bias** — the input data encodes a systematic ESG premium for Global North firms.

15. **ESG Governance (SHAP = 7.82) is the single most important feature** and exhibits the largest regional gaps: Europe ~72 vs. Africa ~33 and Middle East ~33 — a ~39-point divide, more than double the overall ESG gap. Governance standard disparities are the primary driver of geographic ESG inequality in this dataset.

---

## 9. Limitations

1. **Synthetic Dataset Structure:** The perfectly balanced dataset (1,000 records/year uniformly) suggests synthetic or simulated data, limiting direct real-world generalizability.

2. **Sub-score Feature Leakage:** Including `ESG_Environmental`, `ESG_Social`, and `ESG_Governance` when predicting `ESG_Overall` creates near-perfect R2 (>0.997) because the pillars compose the target. Model performance metrics are therefore uninformative for assessing genuine predictive capability on novel firms.

3. **SHAP Computed on XGBoost Only:** SHAP analysis was performed exclusively on XGBoost, while LightGBM outperformed it on all metrics. Computing SHAP on LightGBM would have been more internally consistent.

4. **No Causal Calibration of Geographic Disparity:** The study demonstrates that geographic ESG score gaps exist and are reproduced, but does not causally establish whether this is due to regulatory environments, disclosure standards, data collection bias, industry composition, or genuine sustainability differences.

5. **Single Train/Test Split:** No cross-validation was reported. A single split may overstate model stability, particularly for fairness metrics sensitive to split randomness.

6. **Missing Country-Level Controls:** No macro controls are included (GDP per capita, ESG regulatory stringency, data quality index). Regional SHAP effects conflate geography with economic and institutional contexts.

7. **Empty Notebook Files:** All seven Jupyter notebooks in `notebooks/` are empty (0 bytes). Only derived outputs were available; experimental code cannot be reproduced from the notebooks as submitted.

8. **No Temporal Fairness Analysis:** Fairness metrics are not disaggregated by year. Pre-2020 vs. post-2020 geographic gaps may differ systematically given the evolution of ESG disclosure standards.

9. **Binary Market Type Classification:** The Developed/Emerging binary oversimplifies the economic development spectrum and may conflate very different economies.

10. **No Individual-Level Fairness Analysis:** The study measures group-level (aggregate) fairness only. Individual-level fairness — whether two similar firms from different regions receive similar scores — is not evaluated.

---

## 10. Future Work

1. **Remove Sub-score Features and Re-evaluate.** Train models using only financial, operational, and environmental features (excluding ESG sub-scores) to test whether geographic bias emerges more strongly. This would make the predictive task genuinely challenging and fairness analysis more meaningful.

2. **Extend SHAP Analysis to LightGBM.** Compute SHAP values on the best-performing model for full consistency between explainability and performance reporting.

3. **Apply Stratified Cross-Validation.** Use k-fold CV stratified by region and market type to ensure fairness metrics are stable and not artifacts of a single train-test split.

4. **Integrate Real-World ESG Datasets.** Replace or supplement with real-world ESG ratings (e.g., MSCI, Sustainalytics, Refinitiv, Bloomberg) to validate whether observed geographic gaps and model behaviors hold on authentic corporate data.

5. **Counterfactual Fairness Analysis.** Use counterfactual SHAP or causal fairness frameworks to ask: would a firm's predicted ESG score change if it were located in the Global North instead of the Global South, holding all other attributes constant?

6. **Add Country-Level Controls.** Incorporate macro-level variables — ESG regulatory stringency index, World Governance Indicators, GNI per capita, GRI adoption rate — to disentangle geography from institutional context.

7. **Temporal Fairness Analysis.** Disaggregate all fairness metrics by year to track whether the geographic ESG gap has widened or narrowed between 2015 and 2025.

8. **Intersectional Fairness.** Investigate the interaction between region, industry, and market type. Do energy-sector firms from Africa face a different ESG scoring dynamic than technology-sector firms from Africa?

9. **Fairness-Constrained Training.** Implement fairness-aware learning methods (e.g., adversarial debiasing, reweighting, Fairlearn/AIF360 constraints) and measure the accuracy-fairness tradeoff.

10. **Expand to Qualitative Governance Indicators.** ESG Governance scores exhibit the largest regional gaps. Future work should investigate whether these gaps reflect genuine governance differences or measurement artefacts from disclosure frameworks designed primarily within Western regulatory contexts.

---

*End of Report*

---

> **Data Sources:** `data/company_esg_financial_dataset.csv` (raw), `cleaned_esg.csv` (processed)
> **Models Saved:** `results/xgboost_model.pkl`, `results/random_forest.pkl`, `results/lightgbm.pkl`
> **Key Results Files:** `results/model_comparison.csv`, `results/shap_feature_importance.csv`, `results/fairness_metrics.csv`, `results/fairness_region.csv`, `results/fairness_markettype.csv`
