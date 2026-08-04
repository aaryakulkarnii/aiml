<div align="center">

# 🌍 Auditing Geographic Bias in AI-Driven ESG Scoring

### A SHAP-Based Explainability Analysis and Disclosure-Adjusted ESG Scoring (DAES) Framework for Rating Parity Between Global North and Global South Firms

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Research](https://img.shields.io/badge/Research-AIML%202026-8B5CF6?style=for-the-badge)](https://aiml-conf.org/)
[![XAI](https://img.shields.io/badge/Explainable-AI-EC4899?style=for-the-badge)](https://shap.readthedocs.io/)
[![ESG](https://img.shields.io/badge/Domain-ESG%20Fairness-0EA5E9?style=for-the-badge)]()

**Submitted to the AIML 2026 Conference — Paris, France | October 26–27, 2026**

*Aarya Kulkarni · Aarya Patankar*

</div>

---

## 📌 Project Overview

This repository contains the full end-to-end research pipeline for our paper submitted to **AIML 2026**. The project investigates whether AI-based ESG (Environmental, Social, and Governance) scoring systems encode and perpetuate geographic bias against firms from the **Global South** — particularly those in Africa, Asia, Latin America, and the Middle East — relative to firms from the **Global North** (Europe, North America, Oceania).

Beyond auditing baseline model disparities, this repository implements **Disclosure-Adjusted ESG Scoring (DAES)**, a dynamic, macro-institutional framework that leverages the World Bank Regulatory Quality Index to compute a firm's **Disclosure Infrastructure Index ($DII$)**. By scaling targets dynamically according to regional disclosure deficits, DAES systematically mitigates geographic bias while preserving predictive integrity.

---

## 💡 Research Motivation

ESG scores have become a dominant mechanism through which investors, regulators, and stakeholders assess corporate sustainability. However, the frameworks and data used to generate these scores are predominantly developed by Western rating agencies, calibrated to Western regulatory standards, and evaluated primarily on Global North firms.

This creates a structural risk: AI models trained on such scores may **learn, encode, and amplify** the geographic inequalities baked into the underlying data — not because the models are inherently biased, but because the input signals themselves carry historical and institutional disclosure disadvantages for Global South firms.

Understanding and auditing this bias is critical for:
- **Equitable capital allocation** — ensuring Global South firms are not systematically under-financed due to structural reporting deficits.
- **ESG policy design** — informing how rating agencies should adapt their methodologies for emerging markets.
- **AI accountability & Bias Mitigation** — demonstrating how XAI tools (SHAP) combined with institutional target-scaling (DAES) can audit and correct geographic bias.

---

## ❓ Research Questions

> 1. *Do AI-driven ESG scoring models exhibit geographic bias by systematically predicting lower scores or producing higher prediction errors for firms from the Global South compared to those from the Global North?*
> 2. *How do feature importances shift during sub-pillar ablation when raw operational signals replace aggregated scores?*
> 3. *Can a Disclosure-Adjusted ESG Scoring (DAES) framework dynamically shrink the Global North vs. Global South rating gap without distorting global model performance?*

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Prediction** | Baseline ESG score prediction using XGBoost, Random Forest, and LightGBM |
| 📊 **Sub-Pillar Ablation Study** | `ablation.py` pipeline removing aggregated ESG sub-scores to audit raw feature reliance |
| 🌐 **Institutional DII Integration** | Firm-level Disclosure Infrastructure Index ($DII$) derived from World Bank Governance indicators |
| ⚙️ **DAES Mitigation Framework** | `daes.py` dynamic target scaling to correct institutional disclosure deficits |
| 📈 **Pareto Hyperparameter Sweep** | `daes_alpha_sweep.py` grid search determining optimal trade-off ($\alpha_0 = 0.275$) |
| 🔍 **SHAP Explainability** | Global feature importance (beeswarm, bar) and before/after SHAP shift analysis |
| ⚖️ **Geographic Fairness Audit** | Disaggregated predictions across 7 regions, 2 market types, and Global North vs. South |
| 📐 **Statistical Hypothesis Testing** | `stat_analysis.py` execution of t-tests, Mann-Whitney U, and one-way ANOVA |

---

## 🗂️ Repository Structure

aiml/
│
├── 📓 notebooks/                   # Jupyter notebooks — run in order 01 → 07
│   ├── 01_data_exploration.ipynb   # Dataset inspection, distributions, missing values
│   ├── 02_data_cleaning.ipynb      # Imputation, type casting, Market_Type engineering
│   ├── 03_feature_engineering.ipynb# One-hot encoding, preprocessor pipeline
│   ├── 04_modeling_xgboost.ipynb   # XGBoost training, evaluation, feature importance
│   ├── 05_modeling_rf_lgbm.ipynb   # Random Forest and LightGBM training and comparison
│   ├── 06_shap_analysis.ipynb      # SHAP value computation, summary & dependence plots
│   └── 07_fairness_evaluation.ipynb# Fairness audit by region and market type
│
├── 📁 data/                        # Dataset files
│   ├── company_esg_financial_dataset.csv  # Raw dataset (11,000 firms × 16 features)
│   ├── cleaned_esg.csv                    # Cleaned dataset (11,000 firms × 17 features)
│   └── world_bank_governance.csv          # Macro-institutional regulatory quality indicators
│
├── 📁 results/                     # Model outputs, ablation metrics, and DAES predictions
│   ├── model_comparison.csv        # RMSE, MAE, R² for all three models
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── random_forest.pkl           # Trained Random Forest model
│   ├── lightgbm.pkl                # Trained LightGBM model
│   ├── preprocessor.pkl            # Fitted ColumnTransformer (encoder + scaler)
│   ├── shap_feature_importance.csv # Baseline SHAP values
│   ├── ablation_metrics.csv        # Performance metrics without ESG sub-pillars
│   ├── ablation_shap_importance.csv# SHAP ranking under sub-pillar ablation
│   ├── daes_alpha_sweep.csv        # Pareto sweep grid search log (alpha_0 = 0.00 to 0.50)
│   ├── daes_metrics.csv            # Final DAES performance metrics (alpha_0 = 0.275)
│   ├── daes_predictions.csv        # Test-set predictions under DAES regime
│   ├── daes_shap_importance.csv    # SHAP rankings under DAES regime
│   ├── fairness_metrics.csv        # Per-observation predictions and group errors
│   ├── fairness_region.csv         # Region-level ESG score and error aggregates
│   └── fairness_markettype.csv     # Market-type-level ESG score and error aggregates
│
├── 📁 figures/                     # Generated publication figures (PNG)
│   ├── shap_summary.png            # SHAP beeswarm summary plot
│   ├── shap_bar.png                # SHAP global bar chart
│   ├── top10_shap_features.png     # Top 10 baseline SHAP features
│   ├── shap_carbon.png             # SHAP dependence plot — CarbonEmissions
│   ├── shap_marketcap.png          # SHAP dependence plot — MarketCap
│   ├── esg_region.png              # Baseline ESG score by region
│   ├── esg_markettype.png          # Baseline ESG score by market type
│   ├── esg_boxplot.png             # ESG score distribution (boxplot)
│   ├── error_region.png            # Prediction error by region
│   ├── error_markettype.png        # Prediction error by market type
│   ├── rmse_comparison.png         # Model benchmark RMSE comparison
│   ├── r2_comparison.png           # Model benchmark R² comparison
│   ├── ablation_shap_bar.png       # Feature importance when sub-pillars are ablated
│   ├── daes_pareto_frontier.png    # Pareto trade-off plot (Gap vs. RMSE over alpha_0)
│   ├── daes_before_after.png       # Regional ESG score lift before vs. after DAES
│   └── daes_shap_comparison.png    # Structural SHAP importance shift (Baseline vs. DAES)
│
├── 📄 ablation.py                  # Sub-pillar ablation modeling script
├── 📄 daes_alpha_sweep.py          # Grid search script for DAES Pareto optimization
├── 📄 daes.py                      # Production execution script for DAES framework
├── 📄 stat_analysis.py             # Statistical testing script (scipy)
├── 📄 research_report.md           # Full research paper draft and results report
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
└── 📄 README.md                    # Repository documentation


---

## 🔄 Workflow

Raw Dataset (11,000 firms × 16 features)
                                 │
                                 ▼
                      01–03 · Data & Preprocessing
                 Imputation, one-hot encoding, feature scaling
                                 │
                                 ▼
                     04–05 · Baseline ML Modeling
                 XGBoost, Random Forest, LightGBM benchmark
                                 │
                                 ▼
                     06–07 · Explainability & Fairness
                 SHAP value audit & regional disparity testing
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
📄 ablation.py                                 📄 daes_alpha_sweep.py
Sub-Pillar Ablation                             Dynamic Alpha Grid Search
Audit feature reliance without                 Find Pareto knee point between
E, S, G sub-scores                              Gap reduction & target RMSE
│                                                 │
│                                                 ▼
│                                           📄 daes.py
│                                     Execute DAES (α₀ = 0.275)
│                                     Generate final SHAP & predictions
└────────────────────────┬────────────────────────┘
▼
stat_analysis.py & Paper


---

## ⚖️ Disclosure-Adjusted ESG Scoring (DAES)

### Framework Formulation

To prevent models from penalizing Global South firms for macro-institutional reporting deficits, DAES dynamically scales training targets $y_i$ using a firm-specific **Disclosure Infrastructure Index ($DII_i$)**:

$$y_{\text{DAES}, i} = y_i \times \left(1 + \alpha_0 \cdot (1 - DII_i)\right)$$

Where:
* $DII_i \in [0, 1]$ represents regional regulatory quality and disclosure infrastructure derived from World Bank indicators (Global North $\approx 0.82\text{--}0.85$, Global South $\approx 0.28\text{--}0.45$).
* $(1 - DII_i)$ represents the geographic disclosure deficit.
* $\alpha_0$ is the global adjustment strength hyperparameter.

### Pareto Optimization Results ($\alpha_0$ Sweep)

Running `daes_alpha_sweep.py` across $\alpha_0 \in [0.00, 0.50]$ yielded the following Pareto frontier (`figures/daes_pareto_frontier.png`):

| Hyperparameter ($\alpha_0$) | RMSE ↑ | North Mean | South Mean | Signed Gap (N-S) ↓ | Bias Reduction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.000 (Baseline)** | 8.66 | 62.95 | 47.51 | +15.44 | 0.0% |
| **0.150** | 9.29 | 64.55 | 51.82 | +12.73 | -17.6% |
| **0.275 (Optimal Knee)** ⭐ | **10.57** | **65.88** | **55.51** | **+10.38** | **-32.8%** |
| **0.500 (Aggressive)** | 14.41 | 68.40 | 62.08 | +6.32 | -59.1% |

**⭐ Optimal Crossover Point ($\alpha_0 = 0.275$):**
* **Geographic Gap Reduction:** Shrinks the North-South disparity from **$+15.44$** to **$+10.38$** (a **~33% reduction in geographic bias**).
* **Global South Lift:** Global South scores receive a substantial **$+8.00$ point average boost** ($47.51 \rightarrow 55.51$) compared to a modest **$+2.93$ point adjustment** for Global North firms ($62.95 \rightarrow 65.88$).

---

## 📊 Summary of Key Findings

1. **LightGBM performs best on raw targets** (RMSE = 0.6229, R² = 0.9984), slightly outperforming XGBoost and Random Forest.
2. **A 15-point baseline gap exists** between Developed (62.90) and Emerging (47.95) markets ($p < 0.0001$).
3. **Sub-pillar scores mask geographic proxies:** In baseline models, aggregated ESG sub-scores account for >99% of SHAP weight. When sub-scores are ablated (`ablation.py`), model reliance shifts heavily onto geographic, financial, and operational proxies.
4. **DAES successfully closes the disparity gap:** Setting $\alpha_0 = 0.275$ in `daes.py` closes the geographic score gap by 32.8% while preserving model stability.

---

## 🚀 Quick Start & Script Execution

### 1. Installation

```bash
git clone [https://github.com/aaryakulkarnii/aiml.git](https://github.com/aaryakulkarnii/aiml.git)
cd aiml
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
2. Running Core Scripts
Bash
# Step 1: Run baseline statistical hypothesis testing
python stat_analysis.py

# Step 2: Run sub-pillar ablation study
python ablation.py

# Step 3: Run DAES hyperparameter grid search (Generates Pareto plot)
python daes_alpha_sweep.py

# Step 4: Execute DAES framework at optimal setting (alpha_0 = 0.275)
python daes.py
📝 Citation
Code snippet
@inproceedings{kulkarni2026esgbias,
  title     = {Auditing Geographic Bias in AI-Driven ESG Scoring: A SHAP-Based
               Explainability Analysis and Disclosure-Adjusted ESG Scoring (DAES) Framework},
  author    = {Patankar, Aarya and Kulkarni, Aarya},
 
  url       = {[https://github.com/aaryakulkarnii/aiml](https://github.com/aaryakulkarnii/aiml)}
}
👥 Authors
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

ICAIF'26 , Milan

Built with ❤️ for open and equitable AI research