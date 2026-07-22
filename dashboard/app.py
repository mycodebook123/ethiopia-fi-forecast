import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm

st.set_page_config(page_title="Ethiopia Financial Inclusion Dashboard", layout="wide")

# ---------- Data loading ----------

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/ethiopia_fi_unified_data_enriched.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    obs = df[df["record_type"] == "observation"].copy()
    obs["year"] = obs["observation_date"].dt.year
    events = df[df["record_type"] == "event"].copy()
    links = df[df["record_type"] == "impact_link"].copy()
    targets = df[df["record_type"] == "target"].copy()
    links_full = links.merge(
        events[["record_id", "indicator", "observation_date"]].rename(
            columns={"record_id": "parent_id", "indicator": "event_name", "observation_date": "event_date"}
        ),
        on="parent_id", how="left"
    )
    return df, obs, events, links_full, targets

df, obs, events, links_full, targets = load_data()

# ---------- Forecasting logic (mirrors notebooks/04_forecasting.ipynb) ----------

MAGNITUDE_PP = {"negligible": 0.5, "low": 1.5, "medium": 3.5, "high": 6.0}
DIRECTION_SIGN = {"increase": 1, "decrease": -1, "mixed": 0, "stabilize": 0}
RAMP_MONTHS = 12

def event_effect(event_date, lag_months, magnitude, direction, evidence_basis, eval_date, mag_table_empirical=None):
    if pd.isna(event_date) or pd.isna(lag_months):
        return 0.0
    start = event_date + pd.DateOffset(months=int(lag_months))
    if eval_date < start:
        return 0.0
    elapsed_months = (eval_date.year - start.year) * 12 + (eval_date.month - start.month)
    frac = min(max(elapsed_months / RAMP_MONTHS, 0), 1.0)
    pp_table = mag_table_empirical if (evidence_basis == "empirical" and mag_table_empirical) else MAGNITUDE_PP
    pp = pp_table.get(magnitude, 0.0)
    sign = DIRECTION_SIGN.get(direction, 0)
    return sign * pp * frac

def total_effect(indicator_code, eval_date, mag_table_empirical=None):
    rows = links_full[links_full["related_indicator"] == indicator_code]
    return sum(
        event_effect(r["event_date"], r["lag_months"], r["impact_magnitude"], r["impact_direction"],
                     r["evidence_basis"], eval_date, mag_table_empirical)
        for _, r in rows.iterrows()
    )

@st.cache_data
def compute_calibration():
    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")
    if len(mm) >= 2:
        baseline_date = mm["observation_date"].iloc[0]
        end_date = mm["observation_date"].iloc[-1]
        actual_change = mm["value_numeric"].iloc[-1] - mm["value_numeric"].iloc[0]
        predicted_change = total_effect("ACC_MM_ACCOUNT", end_date) - total_effect("ACC_MM_ACCOUNT", baseline_date)
        factor = actual_change / predicted_change if predicted_change != 0 else 1.0
    else:
        factor = 1.0
    return factor

calibration_factor = compute_calibration()
MAGNITUDE_PP_CALIBRATED = {k: v * calibration_factor for k, v in MAGNITUDE_PP.items()}

FORECAST_YEARS = [2025, 2026, 2027]
FORECAST_DATES = [pd.Timestamp(f"{y}-12-31") for y in FORECAST_YEARS]

def event_augmented_forecast(indicator_code, baseline_value, baseline_date, calibrated=True):
    mag_table = MAGNITUDE_PP_CALIBRATED if calibrated else None
    base_effect = total_effect(indicator_code, baseline_date, mag_table)
    return {d.year: baseline_value + (total_effect(indicator_code, d, mag_table) - base_effect) for d in FORECAST_DATES}

def relative_effect_proxy(target_indicator, proxy_indicators, baseline_value, baseline_date, calibrated=True):
    mag_table = MAGNITUDE_PP_CALIBRATED if calibrated else None
    results = {}
    for d in FORECAST_DATES:
        rel_changes = []
        for pi in proxy_indicators:
            pi_obs = obs[obs["indicator_code"] == pi].sort_values("observation_date")
            if len(pi_obs) == 0:
                continue
            pi_base_val = pi_obs["value_numeric"].iloc[-1]
            pi_base_date = pi_obs["observation_date"].iloc[-1]
            if pi_base_val == 0:
                continue
            base_eff = total_effect(pi, pi_base_date, mag_table)
            fwd_eff = total_effect(pi, d, mag_table)
            rel_changes.append((fwd_eff - base_eff) / pi_base_val)
        avg_rel = np.mean(rel_changes) if rel_changes else 0.0
        results[d.year] = baseline_value * (1 + avg_rel)
    return results

@st.cache_data
def compute_forecasts():
    access = obs[(obs["indicator_code"] == "ACC_OWNERSHIP") & (obs["gender"] == "all")].sort_values("observation_date")
    usage = obs[obs["indicator_code"] == "USG_DIGITAL_PAYMENT"].sort_values("observation_date")

    result = {"access_hist": access, "usage_hist": usage}

    if len(access) >= 3:
        X = sm.add_constant(access["year"])
        y = access["value_numeric"]
        model = sm.OLS(y, X).fit()
        Xp = sm.add_constant(pd.Series(FORECAST_YEARS, name="year"), has_constant="add")
        pred = model.get_prediction(Xp).summary_frame(alpha=0.20)
        pred.index = FORECAST_YEARS
        result["access_trend"] = pred
    else:
        result["access_trend"] = None

    if len(access) > 0:
        a_base_val = access["value_numeric"].iloc[-1]
        a_base_date = access["observation_date"].iloc[-1]
        result["access_base"] = event_augmented_forecast("ACC_OWNERSHIP", a_base_val, a_base_date, calibrated=True)
        result["access_optimistic"] = event_augmented_forecast("ACC_OWNERSHIP", a_base_val, a_base_date, calibrated=False)
        result["access_pessimistic"] = {y: result["access_trend"].loc[y, "mean"] if result["access_trend"] is not None else a_base_val for y in FORECAST_YEARS}

    has_direct_links = "USG_DIGITAL_PAYMENT" in links_full["related_indicator"].values
    if len(usage) > 0:
        u_base_val = usage["value_numeric"].iloc[-1]
        u_base_date = usage["observation_date"].iloc[-1]
        proxy_ind = ["ACC_MM_ACCOUNT", "USG_P2P_COUNT"]
        if has_direct_links:
            result["usage_base"] = event_augmented_forecast("USG_DIGITAL_PAYMENT", u_base_val, u_base_date, calibrated=True)
            result["usage_optimistic"] = event_augmented_forecast("USG_DIGITAL_PAYMENT", u_base_val, u_base_date, calibrated=False)
        else:
            result["usage_base"] = relative_effect_proxy("USG_DIGITAL_PAYMENT", proxy_ind, u_base_val, u_base_date, calibrated=True)
            result["usage_optimistic"] = relative_effect_proxy("USG_DIGITAL_PAYMENT", proxy_ind, u_base_val, u_base_date, calibrated=False)
        result["usage_pessimistic"] = {y: u_base_val for y in FORECAST_YEARS}

    return result

forecasts = compute_forecasts()

# ---------- Sidebar navigation ----------

st.sidebar.title("Ethiopia Financial Inclusion")
page = st.sidebar.radio("Navigate", ["Overview", "Trends", "Forecasts", "Inclusion Projections"])
st.sidebar.markdown("---")
st.sidebar.caption("Selam Analytics | Week 11 Challenge | Data: enriched Findex + operator/regulator sources")

# ---------- Overview Page ----------

if page == "Overview":
    st.title("Ethiopia Financial Inclusion — Overview")

    access = obs[(obs["indicator_code"] == "ACC_OWNERSHIP") & (obs["gender"] == "all")].sort_values("observation_date")
    mm = obs[obs["indicator_code"] == "ACC_MM_ACCOUNT"].sort_values("observation_date")
    usage = obs[obs["indicator_code"] == "USG_DIGITAL_PAYMENT"].sort_values("observation_date")
    p2p = obs[obs["indicator_code"] == "USG_P2P_COUNT"].sort_values("observation_date")
    atm = obs[obs["indicator_code"] == "USG_ATM_COUNT"].sort_values("observation_date")

    col1, col2, col3, col4 = st.columns(4)
    if len(access) > 0:
        latest = access.iloc[-1]
        prev = access.iloc[-2] if len(access) > 1 else None
        delta = f"{latest['value_numeric'] - prev['value_numeric']:+.1f}pp since {int(prev['year'])}" if prev is not None else None
        col1.metric("Account Ownership (Access)", f"{latest['value_numeric']:.0f}%", delta)
    if len(mm) > 0:
        latest = mm.iloc[-1]
        prev = mm.iloc[-2] if len(mm) > 1 else None
        delta = f"{latest['value_numeric'] - prev['value_numeric']:+.2f}pp since {int(prev['year'])}" if prev is not None else None
        col2.metric("Mobile Money Accounts", f"{latest['value_numeric']:.2f}%", delta)
    if len(usage) > 0:
        col3.metric("Digital Payment Adoption (Usage)", f"{usage.iloc[-1]['value_numeric']:.0f}%")
    if len(p2p) > 0 and len(atm) > 0:
        ratio = p2p.iloc[-1]["value_numeric"] / atm.iloc[-1]["value_numeric"] if atm.iloc[-1]["value_numeric"] else None
        if ratio:
            col4.metric("P2P / ATM Crossover Ratio", f"{ratio:.2f}x", "P2P > ATM" if ratio > 1 else "ATM > P2P")

    st.markdown("---")
    st.subheader("Records by Type")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df["record_type"].value_counts().reset_index(), x="record_type", y="count",
                     labels={"record_type": "Record Type", "count": "Count"}, color="record_type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig2 = px.pie(obs, names="pillar", title="Observations by Pillar")
        st.plotly_chart(fig2, width='stretch')

    st.markdown("---")
    st.subheader("Growth Rate Highlights")
    if len(access) >= 2:
        access_sorted = access.reset_index(drop=True)
        access_sorted["pp_change"] = access_sorted["value_numeric"].diff()
        access_sorted["years_elapsed"] = access_sorted["year"].diff()
        access_sorted["pp_per_year"] = access_sorted["pp_change"] / access_sorted["years_elapsed"]
        st.dataframe(access_sorted[["year", "value_numeric", "pp_change", "pp_per_year"]].rename(
            columns={"year": "Year", "value_numeric": "Account Ownership %", "pp_change": "pp Change", "pp_per_year": "pp/Year"}
        ), width='stretch', hide_index=True)

# ---------- Trends Page ----------

elif page == "Trends":
    st.title("Trends Explorer")

    all_indicators = sorted(obs["indicator_code"].dropna().unique())
    default_idx = all_indicators.index("ACC_OWNERSHIP") if "ACC_OWNERSHIP" in all_indicators else 0
    selected = st.multiselect("Select indicator(s) to compare", all_indicators, default=[all_indicators[default_idx]])

    min_year, max_year = int(obs["year"].min()), int(obs["year"].max())
    year_range = st.slider("Date range", min_year, max_year, (min_year, max_year))

    show_events = st.checkbox("Show event markers", value=True)

    filtered = obs[(obs["indicator_code"].isin(selected)) & (obs["year"] >= year_range[0]) & (obs["year"] <= year_range[1])]

    if len(filtered) > 0:
        fig = px.line(filtered.sort_values("observation_date"), x="observation_date", y="value_numeric",
                       color="indicator_code", markers=True,
                       labels={"observation_date": "Date", "value_numeric": "Value", "indicator_code": "Indicator"})
        if show_events:
            for _, ev in events.iterrows():
                if pd.notna(ev["observation_date"]) and year_range[0] <= ev["observation_date"].year <= year_range[1]:
                    fig.add_vline(x=ev["observation_date"].timestamp() * 1000, line_dash="dash", line_color="gray", opacity=0.4)
        st.plotly_chart(fig, width='stretch')

        st.download_button(
            "Download this data as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="trends_data.csv",
            mime="text/csv"
        )
        st.dataframe(filtered[["indicator_code", "indicator", "year", "value_numeric", "source_name", "confidence"]],
                     width='stretch', hide_index=True)
    else:
        st.info("No data for the selected indicator(s) and date range.")

# ---------- Forecasts Page ----------

elif page == "Forecasts":
    st.title("Forecasts: Access and Usage (2025-2027)")

    target_choice = st.selectbox("Select target", ["Access (Account Ownership)", "Usage (Digital Payment Adoption)"])
    model_choice = st.radio("Scenario", ["Pessimistic", "Base", "Optimistic", "Compare all"], horizontal=True)

    if target_choice.startswith("Access"):
        hist = forecasts["access_hist"]
        scenarios = {
            "Pessimistic": forecasts.get("access_pessimistic", {}),
            "Base": forecasts.get("access_base", {}),
            "Optimistic": forecasts.get("access_optimistic", {}),
        }
        title = "Account Ownership Rate Forecast"
    else:
        hist = forecasts["usage_hist"]
        scenarios = {
            "Pessimistic": forecasts.get("usage_pessimistic", {}),
            "Base": forecasts.get("usage_base", {}),
            "Optimistic": forecasts.get("usage_optimistic", {}),
        }
        title = "Digital Payment Adoption Rate Forecast"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["year"], y=hist["value_numeric"], mode="lines+markers",
                              name="Historical (Findex)", line=dict(color="black", width=3)))

    colors = {"Pessimistic": "orange", "Base": "steelblue", "Optimistic": "seagreen"}
    to_plot = scenarios.keys() if model_choice == "Compare all" else [model_choice]
    for s in to_plot:
        vals = scenarios.get(s, {})
        if vals:
            years = sorted(vals.keys())
            fig.add_trace(go.Scatter(x=years, y=[vals[y] for y in years], mode="lines+markers",
                                      name=s, line=dict(color=colors.get(s), dash="dash")))

    if target_choice.startswith("Access") and forecasts.get("access_trend") is not None:
        pt = forecasts["access_trend"]
        fig.add_trace(go.Scatter(x=list(pt.index) + list(pt.index)[::-1],
                                  y=list(pt["obs_ci_upper"]) + list(pt["obs_ci_lower"])[::-1],
                                  fill="toself", fillcolor="rgba(150,150,150,0.15)", line=dict(color="rgba(0,0,0,0)"),
                                  name="80% CI (trend)", showlegend=True))

    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="% of adults")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Forecast Table")
    table = pd.DataFrame(scenarios)
    table.index.name = "Year"
    st.dataframe(table.round(1), width='stretch')

    st.download_button("Download forecast table as CSV", table.to_csv().encode("utf-8"),
                        file_name="forecast_table.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Key Projected Milestones")
    if target_choice.startswith("Access"):
        base_2027 = forecasts.get("access_base", {}).get(2027)
        if base_2027 is not None:
            gap = 70 - base_2027
            st.write(f"Base-case 2027 forecast: **{base_2027:.1f}%** — NFIS-II target is 70%, "
                     f"leaving a projected gap of **{gap:.1f}pp** under the base scenario.")
    else:
        base_2027 = forecasts.get("usage_base", {}).get(2027)
        if base_2027 is not None:
            st.write(f"Base-case 2027 forecast: **{base_2027:.1f}%** for digital payment adoption.")

# ---------- Inclusion Projections Page ----------

elif page == "Inclusion Projections":
    st.title("Inclusion Projections & Policy Questions")

    scenario = st.selectbox("Scenario", ["Pessimistic", "Base", "Optimistic"])
    access_val = forecasts.get(f"access_{scenario.lower()}", {}).get(2027)
    nfis_target = 70

    st.subheader("Progress Toward NFIS-II 70% Account Ownership Target")
    if access_val is not None:
        progress = min(access_val / nfis_target, 1.0)
        st.progress(progress, text=f"{access_val:.1f}% of 70% target ({scenario} scenario, 2027)")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=access_val,
            delta={"reference": nfis_target},
            gauge={"axis": {"range": [0, 100]}, "threshold": {"line": {"color": "red", "width": 4}, "value": nfis_target},
                   "bar": {"color": "steelblue"}},
            title={"text": f"Account Ownership vs. NFIS-II Target ({scenario}, 2027)"}
        ))
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("Answers to the Consortium's Key Questions")

    st.markdown("**What drives financial inclusion in Ethiopia?**")
    st.write("Mobile money product launches (Telebirr, M-Pesa) are the strongest empirically "
             "validated drivers of Access-adjacent growth; policy coordination (NFIS-II) and "
             "infrastructure (Fayda ID, EthioPay) contribute with longer lags and more uncertainty, "
             "based on comparable-country evidence rather than direct Ethiopian validation.")

    st.markdown("**How do events like product launches, policy changes, and infrastructure "
                "investments affect inclusion outcomes?**")
    st.write("See the Forecasts page for the event-augmented model, and the event-indicator "
             "association matrix built in Task 3 for the full effect-size breakdown by event.")

    st.markdown("**How did financial inclusion change in 2025, and how will it look in 2026-2027?**")
    if access_val is not None:
        st.write(f"Under the {scenario.lower()} scenario, Account Ownership is projected to reach "
                 f"**{access_val:.1f}%** by 2027, against an NFIS-II target of 70%.")

    st.caption("Forecasts carry substantial uncertainty given limited historical Findex survey "
               "points (4 for Access, 1 for Usage) - see notebooks/04_forecasting.ipynb for full "
               "methodology, assumptions, and limitations.")
