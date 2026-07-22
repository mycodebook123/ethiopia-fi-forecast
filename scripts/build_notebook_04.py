import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Task 4: Forecasting Access and Usage
## Ethiopia Financial Inclusion Forecasting

Forecasts Account Ownership Rate (Access) and Digital Payment Adoption Rate (Usage) for
2025-2027, using trend regression, event-augmented modeling with the Task 3 calibrated
association matrix, and optimistic/base/pessimistic scenarios.
"""
))

cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv('../data/processed/ethiopia_fi_unified_data_enriched.csv')
df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')

obs = df[df['record_type'] == 'observation'].copy()
events = df[df['record_type'] == 'event'].copy()
links = df[df['record_type'] == 'impact_link'].copy()
targets = df[df['record_type'] == 'target'].copy()

links_full = links.merge(
    events[['record_id', 'indicator', 'observation_date']].rename(
        columns={'record_id': 'parent_id', 'indicator': 'event_name', 'observation_date': 'event_date'}
    ),
    on='parent_id', how='left'
)

print(f"Observations: {len(obs)}")
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 1. Define Targets

- **Access**: `ACC_OWNERSHIP` - Account Ownership Rate, 4 real Findex survey points (2014, 2017, 2021, 2024)
- **Usage**: `USG_DIGITAL_PAYMENT` - Digital Payment Adoption Rate, 1 real Findex point (2024, ~35%)

Usage has only a single data point, so no trend can be fit to it alone - its forecast relies
entirely on the event-augmented model plus scenario bounds, not a fitted regression line.
"""
))

cells.append(nbf.v4.new_code_cell(
"""access = obs[(obs['indicator_code'] == 'ACC_OWNERSHIP') & (obs['gender'] == 'all')].sort_values('observation_date').copy()
access['year'] = access['observation_date'].dt.year

usage = obs[obs['indicator_code'] == 'USG_DIGITAL_PAYMENT'].sort_values('observation_date').copy()
usage['year'] = usage['observation_date'].dt.year

print("Access (ACC_OWNERSHIP):")
print(access[['year', 'value_numeric']])
print()
print("Usage (USG_DIGITAL_PAYMENT):")
print(usage[['year', 'value_numeric']])
"""
))

cells.append(nbf.v4.new_markdown_cell("## 2. Trend Regression (Baseline) - Access"))

cells.append(nbf.v4.new_code_cell(
"""X = sm.add_constant(access['year'])
y = access['value_numeric']
trend_model = sm.OLS(y, X).fit()

# Print key stats directly rather than .summary() - .summary() calls Jarque-Bera which
# requires >=2 residuals and can error out on very small samples like ours (n=4).
print(f"Intercept: {trend_model.params['const']:.3f}")
print(f"Slope (pp/year): {trend_model.params['year']:.3f}")
print(f"R-squared: {trend_model.rsquared:.3f}")
print(f"N observations: {int(trend_model.nobs)}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""forecast_years = [2025, 2026, 2027]
X_pred = sm.add_constant(pd.Series(forecast_years, name='year'), has_constant='add')
pred = trend_model.get_prediction(X_pred)
pred_summary = pred.summary_frame(alpha=0.20)  # 80% CI, appropriate given n=4
pred_summary.index = forecast_years
pred_summary['mean'] = pred_summary['mean'].clip(upper=100, lower=0)
pred_summary
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""**Data gap found during modeling:** none of the 14 impact_link records built in Task 1/3
directly target `USG_DIGITAL_PAYMENT` (they target related Usage-pillar indicators like
`ACC_MM_ACCOUNT` and `USG_P2P_COUNT` instead). Forecasting Usage with zero direct event links
would make all three scenarios identical - defeating the purpose of scenario analysis. As a
documented workaround, when an indicator has no direct impact_links, we proxy its event-driven
growth using the **average relative (%) growth** modeled for other Usage-pillar indicators that
do have direct links, scaled onto the target indicator's own baseline value. This is an explicit
modeling assumption, not a validated relationship - it assumes Usage-pillar indicators move
together proportionally, which is plausible but untested here.
"""
))

cells.append(nbf.v4.new_code_cell(
"""def relative_effect_proxy(target_indicator, proxy_indicators, baseline_value, baseline_date, forecast_dates, mag_table=None):
    \"\"\"For indicators with no direct impact_links, proxy event-driven growth using the average
    relative growth of related indicators that do have direct links.\"\"\"
    results = {}
    for d in forecast_dates:
        rel_changes = []
        for pi in proxy_indicators:
            pi_baseline = obs[obs['indicator_code'] == pi].sort_values('observation_date')
            if len(pi_baseline) == 0:
                continue
            pi_base_val = pi_baseline['value_numeric'].iloc[-1]
            pi_base_date = pi_baseline['observation_date'].iloc[-1]
            if pi_base_val == 0:
                continue
            base_eff = total_effect(pi, pi_base_date, mag_table)
            fwd_eff = total_effect(pi, d, mag_table)
            rel_changes.append((fwd_eff - base_eff) / pi_base_val)
        avg_rel_change = np.mean(rel_changes) if rel_changes else 0.0
        results[d.year] = baseline_value * (1 + avg_rel_change)
    return results

USAGE_PROXY_INDICATORS = ['ACC_MM_ACCOUNT', 'USG_P2P_COUNT']

has_direct_links = 'USG_DIGITAL_PAYMENT' in links_full['related_indicator'].values
print(f"USG_DIGITAL_PAYMENT has direct impact_links: {has_direct_links}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""MAGNITUDE_PP = {'negligible': 0.5, 'low': 1.5, 'medium': 3.5, 'high': 6.0}
DIRECTION_SIGN = {'increase': 1, 'decrease': -1, 'mixed': 0, 'stabilize': 0}
RAMP_MONTHS = 12

# Re-derive the same calibration factor as Task 3, using the one validated relationship
# (Telebirr -> mobile money accounts), so this notebook is self-consistent and reproducible
# without depending on a hardcoded constant.
mm = obs[obs['indicator_code'] == 'ACC_MM_ACCOUNT'].sort_values('observation_date')

def event_effect(event_date, lag_months, magnitude, direction, evidence_basis, eval_date,
                  mag_table_empirical=None, ramp_months=RAMP_MONTHS):
    if pd.isna(event_date) or pd.isna(lag_months):
        return 0.0
    start = event_date + pd.DateOffset(months=int(lag_months))
    if eval_date < start:
        return 0.0
    elapsed_months = (eval_date.year - start.year) * 12 + (eval_date.month - start.month)
    frac = min(max(elapsed_months / ramp_months, 0), 1.0)
    pp_table = mag_table_empirical if (evidence_basis == 'empirical' and mag_table_empirical) else MAGNITUDE_PP
    pp = pp_table.get(magnitude, 0.0)
    sign = DIRECTION_SIGN.get(direction, 0)
    return sign * pp * frac

def total_effect(indicator_code, eval_date, mag_table_empirical=None, links_df=links_full):
    rows = links_df[links_df['related_indicator'] == indicator_code]
    return sum(
        event_effect(r['event_date'], r['lag_months'], r['impact_magnitude'], r['impact_direction'],
                     r['evidence_basis'], eval_date, mag_table_empirical)
        for _, r in rows.iterrows()
    )

if len(mm) >= 2:
    baseline_date = mm['observation_date'].iloc[0]
    end_date = mm['observation_date'].iloc[-1]
    actual_change = mm['value_numeric'].iloc[-1] - mm['value_numeric'].iloc[0]
    predicted_change = total_effect('ACC_MM_ACCOUNT', end_date) - total_effect('ACC_MM_ACCOUNT', baseline_date)
    calibration_factor = actual_change / predicted_change if predicted_change != 0 else 1.0
else:
    calibration_factor = 1.0

MAGNITUDE_PP_CALIBRATED = {k: v * calibration_factor for k, v in MAGNITUDE_PP.items()}
print(f"Calibration factor (re-derived): {calibration_factor:.3f}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""def event_augmented_forecast(indicator_code, baseline_value, baseline_date, forecast_dates):
    results = {}
    base_effect = total_effect(indicator_code, baseline_date, MAGNITUDE_PP_CALIBRATED)
    for d in forecast_dates:
        eff = total_effect(indicator_code, d, MAGNITUDE_PP_CALIBRATED) - base_effect
        results[d.year] = baseline_value + eff
    return results

access_baseline_date = access['observation_date'].iloc[-1]
access_baseline_value = access['value_numeric'].iloc[-1]
forecast_dates = [pd.Timestamp(f'{y}-12-31') for y in forecast_years]

access_event_augmented = event_augmented_forecast('ACC_OWNERSHIP', access_baseline_value, access_baseline_date, forecast_dates)
print("Access - event-augmented forecast:", access_event_augmented)

usage_baseline_date = usage['observation_date'].iloc[-1]
usage_baseline_value = usage['value_numeric'].iloc[-1]

if has_direct_links:
    usage_event_augmented = event_augmented_forecast('USG_DIGITAL_PAYMENT', usage_baseline_value, usage_baseline_date, forecast_dates)
else:
    usage_event_augmented = relative_effect_proxy('USG_DIGITAL_PAYMENT', USAGE_PROXY_INDICATORS,
                                                    usage_baseline_value, usage_baseline_date, forecast_dates,
                                                    MAGNITUDE_PP_CALIBRATED)
print("Usage - event-augmented forecast:", usage_event_augmented)
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 4. Scenario Analysis

- **Pessimistic**: trend-only (no additional event effects materialize beyond historical trend)
- **Base**: event-augmented model (calibrated)
- **Optimistic**: event-augmented model with effects at their original (uncalibrated) magnitude -
  i.e. assumes events perform at their full assumed impact rather than the calibrated, more
  conservative estimate
"""
))

cells.append(nbf.v4.new_code_cell(
"""access_optimistic = event_augmented_forecast('ACC_OWNERSHIP', access_baseline_value, access_baseline_date, forecast_dates)
# recompute optimistic with uncalibrated (original) magnitudes
def event_augmented_forecast_uncalibrated(indicator_code, baseline_value, baseline_date, forecast_dates):
    results = {}
    base_effect = total_effect(indicator_code, baseline_date)
    for d in forecast_dates:
        eff = total_effect(indicator_code, d) - base_effect
        results[d.year] = baseline_value + eff
    return results

access_optimistic = event_augmented_forecast_uncalibrated('ACC_OWNERSHIP', access_baseline_value, access_baseline_date, forecast_dates)
if has_direct_links:
    usage_optimistic = event_augmented_forecast_uncalibrated('USG_DIGITAL_PAYMENT', usage_baseline_value, usage_baseline_date, forecast_dates)
else:
    usage_optimistic = relative_effect_proxy('USG_DIGITAL_PAYMENT', USAGE_PROXY_INDICATORS,
                                              usage_baseline_value, usage_baseline_date, forecast_dates, None)

access_pessimistic = {y: pred_summary.loc[y, 'mean'] for y in forecast_years}
# Usage has no fitted trend (1 data point) - pessimistic case is "no further growth"
usage_pessimistic = {y: usage_baseline_value for y in forecast_years}

access_forecast_table = pd.DataFrame({
    'pessimistic (trend-only)': access_pessimistic,
    'base (calibrated event-augmented)': access_event_augmented,
    'optimistic (uncalibrated event-augmented)': access_optimistic,
})
access_forecast_table.index.name = 'year'

usage_forecast_table = pd.DataFrame({
    'pessimistic (no further growth)': usage_pessimistic,
    'base (calibrated event-augmented)': usage_event_augmented,
    'optimistic (uncalibrated event-augmented)': usage_optimistic,
})
usage_forecast_table.index.name = 'year'

print("ACCESS (Account Ownership Rate, %) forecast:")
print(access_forecast_table.round(1))
print()
print("USAGE (Digital Payment Adoption Rate, %) forecast:")
print(usage_forecast_table.round(1))
"""
))

cells.append(nbf.v4.new_markdown_cell("## 5. Forecast Visualization"))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(access['year'], access['value_numeric'], marker='o', color='black', linewidth=2, label='Historical (Findex)')
for col, style in zip(access_forecast_table.columns, ['--', '-', ':']):
    ax.plot(access_forecast_table.index, access_forecast_table[col], marker='s', linestyle=style, label=col)
ax.fill_between(pred_summary.index, pred_summary['obs_ci_lower'].clip(lower=0), pred_summary['obs_ci_upper'].clip(upper=100),
                alpha=0.15, color='gray', label='80% CI (trend)')
ax.axhline(70, color='red', linestyle=':', alpha=0.5, label='NFIS-II target (70%)')
ax.set_title('Access: Account Ownership Rate Forecast')
ax.set_ylabel('% of adults')
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(usage['year'], usage['value_numeric'], color='black', s=60, zorder=3, label='Historical (Findex, 1 point)')
for col, style in zip(usage_forecast_table.columns, ['--', '-', ':']):
    ax.plot(usage_forecast_table.index, usage_forecast_table[col], marker='s', linestyle=style, label=col)
ax.set_title('Usage: Digital Payment Adoption Rate Forecast')
ax.set_ylabel('% of adults')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../reports/figures/10_forecast_scenarios.png', bbox_inches='tight')
plt.show()
"""
))

cells.append(nbf.v4.new_markdown_cell(
"""## 6. Interpretation

**Access (Account Ownership):** The trend-only (pessimistic) path continues the historical
slowdown seen since 2021, landing well short of the NFIS-II 70% target by 2027. The calibrated
event-augmented (base) forecast adds a modest lift from Fayda ID rollout, EthioPay, and NFIS-II's
own coordinating effect, but even this base case falls short of 70% by 2027 given the size of the
gap and the conservative calibration derived from the one validated Ethiopian relationship. Only
the optimistic scenario (uncalibrated event magnitudes) approaches the target range, which
should be read as an upper bound requiring events to perform at their full assumed impact with
no diminishing returns - a strong assumption.

**Usage (Digital Payment Adoption):** With only one real Findex data point, this forecast is
structurally weaker than Access - there is no historical trend to validate against, so all
three scenarios depend entirely on assumptions carried over from the (mostly comparable-country,
`theoretical`-basis) event effects. This should be treated as the most uncertain forecast in
this project and flagged clearly to the consortium as such.

**Events with largest potential impact:** Per the Task 3 association matrix, NFIS-II (long lag,
medium magnitude, coordinating effect) and EthioPay (medium magnitude, direct infrastructure)
contribute the most to Access; M-Pesa/EthSwitch interoperability and Telebirr contribute most to
Usage-adjacent indicators.

**Key uncertainties:**
- Calibration is derived from a single validated relationship (n=1); applying it broadly assumes
  that relationship generalizes to other event-indicator pairs, which is untested.
- Usage forecasts rest on a single real data point and mostly comparable-country evidence.
- The linear-ramp functional form and additive combination of multiple events (documented in
  Task 3) carry forward into these forecasts and remain unvalidated assumptions.
"""
))

nb['cells'] = cells

with open('notebooks/04_forecasting.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/04_forecasting.ipynb")
