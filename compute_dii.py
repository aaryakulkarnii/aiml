"""
Compute region-level DII (Disclosure/Institutional Index) from REAL
World Bank Regulatory Quality data, instead of the hand-picked constants
previously hardcoded in daes.py.

Why this exists: the original daes_region_map in daes.py was a manually
chosen 7-value dictionary that was NOT derived from world_bank_governance.csv,
despite that file being present in the repo. This script closes that gap.

Method:
  1. Load real 2024 Regulatory Quality estimates for 222 countries
     (World Bank Worldwide Governance Indicators, -2.5 to +2.5 scale)
  2. Map each country to one of the dataset's 7 regions using standard
     continent classification (pycountry_convert), with a manual
     override list for the Middle East (which is not its own continent
     code, so it must be carved out of Asia/Africa by name) and for
     Latin America vs. North America (both share continent code 'NA'
     in pycountry_convert's schema, so the US/Canada must be split out)
  3. Average the real regulatory quality estimate within each region
  4. Min-max normalize the 7 regional averages to a 0-1 DII scale

This mapping is a defensible approximation, not a perfect one -- exact
country-to-region assignment for firms in cleaned_esg.csv is unknown
(the dataset only records broad Region, not country), so this produces
the best available REGION-level proxy from real data, replacing values
that were previously asserted rather than computed.
"""
import pandas as pd
import numpy as np
import pycountry_convert as pc

wb = pd.read_csv('data/world_bank_governance.csv')
wb = wb.rename(columns={'2024 [YR2024]': 'RegQuality'})
wb['RegQuality'] = pd.to_numeric(wb['RegQuality'], errors='coerce')
wb = wb.dropna(subset=['RegQuality'])

# Countries considered "Middle East" for this taxonomy (not a standard
# continent code -- must be carved out manually by ISO3 code)
MIDDLE_EAST_ISO3 = {
    'BHR', 'CYP', 'EGY', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN',
    'OMN', 'PSE', 'QAT', 'SAU', 'SYR', 'TUR', 'ARE', 'YEM'
}
# North America proper (rest of continent code 'NA' becomes Latin America)
NORTH_AMERICA_ISO3 = {'USA', 'CAN'}
# Oceania restricted to Australia/NZ: pycountry_convert's 'OC' continent
# code includes ~18 small Pacific island states (Vanuatu, Kiribati, etc.)
# that never appear as HQ countries in a corporate ESG dataset. A dataset
# labeled "Oceania" for listed companies is, in practice, ASX/NZX firms.
OCEANIA_ISO3 = {'AUS', 'NZL'}

CONTINENT_TO_REGION = {
    'EU': 'Europe',
    'AF': 'Africa',
    'AS': 'Asia',
    'SA': 'Latin America',   # South America -> Latin America
    'NA': 'Latin America',   # default; overridden to North America below
    # 'OC' deliberately excluded here: only AUS/NZL (via OCEANIA_ISO3
    # above) count as "Oceania" for this corporate dataset. Other
    # Pacific island states are excluded rather than dragging the
    # Oceania average down with countries that never appear as firm
    # headquarters in this kind of data.
}


def classify_region(iso3):
    if iso3 in MIDDLE_EAST_ISO3:
        return 'Middle East'
    if iso3 in NORTH_AMERICA_ISO3:
        return 'North America'
    if iso3 in OCEANIA_ISO3:
        return 'Oceania'
    try:
        iso2 = pc.country_alpha3_to_country_alpha2(iso3)
        continent_code = pc.country_alpha2_to_continent_code(iso2)
    except (KeyError, Exception):
        return None
    region = CONTINENT_TO_REGION.get(continent_code)
    if continent_code == 'NA' and iso3 not in NORTH_AMERICA_ISO3:
        return 'Latin America'  # Mexico, Central America, Caribbean
    return region


wb['Region'] = wb['Country Code'].apply(classify_region)
wb_matched = wb.dropna(subset=['Region'])

print(f"Matched {len(wb_matched)} / {len(wb)} countries to a region")
print(f"Unmatched countries (excluded): {len(wb) - len(wb_matched)}")

region_avg = wb_matched.groupby('Region')['RegQuality'].agg(['mean', 'count']).round(4)
print("\n── Real Regulatory Quality by Region (raw WGI scale, -2.5 to +2.5) ──")
print(region_avg)

# Min-max normalize to 0-1 DII scale
raw_means = region_avg['mean']
dii_normalized = (raw_means - raw_means.min()) / (raw_means.max() - raw_means.min())
dii_normalized = dii_normalized.round(4)

print("\n── Computed DII (0-1 scale, from real data) ──")
print(dii_normalized.sort_values(ascending=False))

# Save for daes.py to consume
out = dii_normalized.reset_index()
out.columns = ['Region', 'DII']
out.to_csv('data/dii_by_region.csv', index=False)
print("\nSaved: data/dii_by_region.csv")

# Compare against the old hardcoded values for transparency
old_hardcoded = {
    'Europe': 0.82, 'North America': 0.85, 'Oceania': 0.83,
    'Asia': 0.45, 'Latin America': 0.41, 'Middle East': 0.38, 'Africa': 0.28
}
print("\n── Old (hardcoded) vs New (computed from real World Bank data) ──")
comparison = pd.DataFrame({
    'Old_Hardcoded': pd.Series(old_hardcoded),
    'New_Computed': dii_normalized
}).round(4)
print(comparison)
comparison.to_csv('results/dii_old_vs_new_comparison.csv')
print("\nSaved: results/dii_old_vs_new_comparison.csv")