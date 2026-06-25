import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv('results/fairness_metrics.csv')

# Statistical test: Developed vs Emerging
dev = df[df['Market_Type'] == 'Developed']['Actual_ESG']
emg = df[df['Market_Type'] == 'Emerging']['Actual_ESG']
t_stat, p_val = stats.ttest_ind(dev, emg)
print(f'T-test (Developed vs Emerging ESG): t={t_stat:.4f}, p={p_val:.10f}')

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dev, emg, alternative='two-sided')
print(f'Mann-Whitney U (Developed vs Emerging): U={u_stat:.4f}, p={p_mw:.10f}')

# ANOVA across regions
groups = [group['Actual_ESG'].values for name, group in df.groupby('Region')]
f_stat, p_anova = stats.f_oneway(*groups)
print(f'One-Way ANOVA (Region): F={f_stat:.4f}, p={p_anova:.10f}')

# Error differences
dev_err = df[df['Market_Type'] == 'Developed']['Absolute_Error']
emg_err = df[df['Market_Type'] == 'Emerging']['Absolute_Error']
t_err, p_err = stats.ttest_ind(dev_err, emg_err)
print(f'T-test (Error Developed vs Emerging): t={t_err:.4f}, p={p_err:.6f}')

# ANOVA on error across regions
err_groups = [group['Absolute_Error'].values for name, group in df.groupby('Region')]
f_err, p_err_anova = stats.f_oneway(*err_groups)
print(f'One-Way ANOVA (Error by Region): F={f_err:.4f}, p={p_err_anova:.6f}')

# Error by region
print('\nAbsolute Error by Region:')
for region, group in df.groupby('Region'):
    print(f'  {region}: mean={group["Absolute_Error"].mean():.4f}, std={group["Absolute_Error"].std():.4f}')

# Prediction error by market type
print('\nAbsolute Error by Market_Type:')
for mt, group in df.groupby('Market_Type'):
    print(f'  {mt}: mean={group["Absolute_Error"].mean():.4f}, std={group["Absolute_Error"].std():.4f}')

# Correlation
print('\nCorrelation of features with Absolute_Error:')
num_cols = ['Revenue','ProfitMargin','MarketCap','GrowthRate','ESG_Environmental','ESG_Social','ESG_Governance',
            'CarbonEmissions','WaterUsage','EnergyConsumption','Actual_ESG']
corr = df[num_cols + ['Absolute_Error']].corr()['Absolute_Error'].drop('Absolute_Error')
print(corr.round(4))

# North vs South: define based on region
north_regions = ['Europe','North America','Oceania']
south_regions = ['Africa','Asia','Latin America','Middle East']
df['GeoGroup'] = df['Region'].apply(lambda x: 'Global North' if x in north_regions else 'Global South')
north_esg = df[df['GeoGroup'] == 'Global North']['Actual_ESG']
south_esg = df[df['GeoGroup'] == 'Global South']['Actual_ESG']
t_ns, p_ns = stats.ttest_ind(north_esg, south_esg)
print(f'\nT-test (Global North vs Global South ESG): t={t_ns:.4f}, p={p_ns:.10f}')
print(f'North ESG mean: {north_esg.mean():.4f}, South ESG mean: {south_esg.mean():.4f}')

north_err = df[df['GeoGroup'] == 'Global North']['Absolute_Error']
south_err = df[df['GeoGroup'] == 'Global South']['Absolute_Error']
t_ns_err, p_ns_err = stats.ttest_ind(north_err, south_err)
print(f'T-test (Error: Global North vs South): t={t_ns_err:.4f}, p={p_ns_err:.6f}')
print(f'North error mean: {north_err.mean():.4f}, South error mean: {south_err.mean():.4f}')
