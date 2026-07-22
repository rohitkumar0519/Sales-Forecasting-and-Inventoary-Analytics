"""
Project FORESIGHT
AI-Powered Demand & Inventory Intelligence Platform
Streamlit dashboard entry point.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import load_all_data
from forecasting import forecast_multiple, FORECAST_HORIZON_WEEKS
from inventory import build_inventory_intelligence, portfolio_kpis

# ----------------------------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT | Inventory Control Tower",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLORS = {
    "Stockout Risk": "#F97066",
    "Overstock Risk": "#A78BFA",
    "Healthy": "#34D399",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, #12192C 0%, #0A0F1C 45%, #070B14 100%);
    color: #E4E9F2;
}

section[data-testid="stSidebar"] {
    background: #0B1120;
    border-right: 1px solid #1E293B;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

.foresight-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 4px 0 18px 0;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 22px;
}
.foresight-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.02em;
}
.foresight-title span { color: #38BDF8; }
.foresight-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #64748B;
    margin-top: 2px;
}
.foresight-live {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: #34D399;
    border: 1px solid #1E4033;
    background: #0D1F17;
    padding: 5px 12px;
    border-radius: 20px;
}
.foresight-live::before {
    content: "● ";
}

.kpi-card {
    background: linear-gradient(180deg, #121A2E 0%, #0D1424 100%);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px 18px;
    height: 100%;
}
.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1.1;
}
.kpi-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin-top: 5px;
}
.kpi-delta.warn { color: #F97066; }
.kpi-delta.info { color: #A78BFA; }
.kpi-delta.good { color: #34D399; }

.risk-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 20px;
    display: inline-block;
}

.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 2px;
}

div[data-testid="stMetric"] {
    background: #121A2E;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 10px 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    color: #94A3B8;
    padding: 10px 18px;
}
.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
}

.dataframe, .stDataFrame { font-family: 'JetBrains Mono', monospace !important; font-size: 12.5px; }

hr { border-color: #1E293B; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#CBD5E1", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ----------------------------------------------------------------------------
# DATA LAYER (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_base_data():
    return load_all_data()


@st.cache_data(show_spinner=False)
def get_all_forecasts(_sales_history: pd.DataFrame, combos: tuple):
    forecasts = {}
    for sku_id, store_id in combos:
        forecasts[(sku_id, store_id)] = forecast_multiple(_sales_history, sku_id, store_id)
    return forecasts


@st.cache_data(show_spinner=False)
def get_intelligence(_inventory_master: pd.DataFrame, _forecasts: dict):
    intel = build_inventory_intelligence(_inventory_master, _forecasts)
    kpis = portfolio_kpis(intel)
    return intel, kpis


sales_history, sku_master, inventory_master = get_base_data()
combos = tuple(zip(inventory_master["sku_id"], inventory_master["store_id"]))

with st.spinner("Training XGBoost + LightGBM ensemble across the SKU-store portfolio..."):
    forecasts = get_all_forecasts(sales_history, combos)

intel_df, kpis = get_intelligence(inventory_master, forecasts)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
header_l, header_r = st.columns([5, 1])
with header_l:
    st.markdown(
        """
        <div class="foresight-header">
            <div>
                <div class="foresight-title">◈ FORESIGHT <span>Control Tower</span></div>
                <div class="foresight-subtitle">AI-Powered Demand Forecasting &amp; Inventory Intelligence Platform</div>
            </div>
            <div class="foresight-live">LIVE PORTFOLIO VIEW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Filters")
region_options = ["All"] + sorted(inventory_master["region"].unique().tolist())
category_options = ["All"] + sorted(inventory_master["category"].unique().tolist())
risk_options = ["All", "Stockout Risk", "Overstock Risk", "Healthy"]

sel_region = st.sidebar.selectbox("Region", region_options)
sel_category = st.sidebar.selectbox("Category", category_options)
sel_risk = st.sidebar.selectbox("Risk status", risk_options)
search_term = st.sidebar.text_input("Search SKU name or ID").strip().lower()

filtered = intel_df.copy()
if sel_region != "All":
    filtered = filtered[filtered["region"] == sel_region]
if sel_category != "All":
    filtered = filtered[filtered["category"] == sel_category]
if sel_risk != "All":
    filtered = filtered[filtered["risk_status"] == sel_risk]
if search_term:
    filtered = filtered[
        filtered["sku_name"].str.lower().str.contains(search_term)
        | filtered["sku_id"].str.lower().str.contains(search_term)
    ]

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#64748B;line-height:1.7;">
    PORTFOLIO<br>
    {kpis['total_sku_locations']} SKU-locations<br>
    {sku_master.shape[0]} SKUs · {inventory_master['store_id'].nunique()} sites<br>
    {FORECAST_HORIZON_WEEKS}-week rolling forecast<br>
    Model: XGBoost + LightGBM ensemble
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# KPI STRIP
# ----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
kpi_specs = [
    (k1, "Inventory Value", f"${kpis['total_inventory_value']:,.0f}", "across active portfolio", "good"),
    (k2, "Stockout Risk", f"{kpis['stockout_count']} lines", f"{kpis['stockout_pct']}% of portfolio", "warn"),
    (k3, "Overstock Risk", f"{kpis['overstock_count']} lines", f"{kpis['overstock_pct']}% of portfolio", "info"),
    (k4, "Est. Lost Sales Risk", f"${kpis['total_lost_sales_risk']:,.0f}", "if stockouts unresolved", "warn"),
    (k5, "Markdown Exposure", f"${kpis['total_markdown_exposure']:,.0f}", "tied up in excess stock", "info"),
]
for col, label, value, delta, tone in kpi_specs:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-delta {tone}">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_radar, tab_forecast, tab_reco, tab_about = st.tabs(
    ["Risk Radar", "Demand Forecast", "Recommendations", "Methodology"]
)

# --- TAB 1: RISK RADAR (signature grid) --------------------------------------
with tab_radar:
    st.markdown('<div class="section-eyebrow">Inventory Radar Grid</div>', unsafe_allow_html=True)
    st.markdown("Every SKU × site combination, colored by real-time risk status. Scan for red before it becomes a stockout.")

    radar_df = filtered.copy()
    radar_df["y_label"] = radar_df["sku_name"] + " · " + radar_df["sku_id"]

    risk_rank = {"Stockout Risk": 0, "Healthy": 1, "Overstock Risk": 2}
    radar_df["risk_rank"] = radar_df["risk_status"].map(risk_rank)

    fig = go.Figure()
    for risk_status, color in RISK_COLORS.items():
        subset = radar_df[radar_df["risk_status"] == risk_status]
        fig.add_trace(go.Scatter(
            x=subset["store_id"],
            y=subset["y_label"],
            mode="markers",
            marker=dict(size=22, color=color, symbol="square", line=dict(width=1, color="#0A0F1C")),
            name=risk_status,
            customdata=subset[["current_stock", "weeks_of_cover", "forecast_avg_weekly", "region"]],
            hovertemplate=(
                "<b>%{y}</b><br>Site: %{x} (%{customdata[3]})<br>"
                "Stock on hand: %{customdata[0]}<br>"
                "Weeks of cover: %{customdata[1]}<br>"
                "Avg forecast demand/wk: %{customdata[2]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(420, 26 * radar_df["y_label"].nunique()),
        xaxis=dict(title="Site", showgrid=False, side="top"),
        yaxis=dict(title="", showgrid=False, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Stockout Risk = projected cover falls below lead time before the next replenishment can land. "
        "Overstock Risk = cover exceeds ~4x lead time, tying up capital and markdown exposure."
    )

# --- TAB 2: DEMAND FORECAST EXPLORER -----------------------------------------
with tab_forecast:
    st.markdown('<div class="section-eyebrow">Demand Forecast Explorer</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([2, 1])
    with fc1:
        sku_choice = st.selectbox(
            "SKU",
            sku_master["sku_id"] + " — " + sku_master["sku_name"],
            index=0,
        )
        chosen_sku_id = sku_choice.split(" — ")[0]
    with fc2:
        store_choice = st.selectbox("Site", inventory_master["store_id"].unique())

    hist = sales_history[
        (sales_history["sku_id"] == chosen_sku_id) & (sales_history["store_id"] == store_choice)
    ].sort_values("week_start").tail(52)
    fc = forecasts.get((chosen_sku_id, store_choice))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist["week_start"], y=hist["units_sold"], mode="lines", name="Actual sales",
        line=dict(color="#94A3B8", width=1.6),
    ))
    if fc is not None and not fc.empty:
        fig2.add_trace(go.Scatter(
            x=pd.concat([fc["week_start"], fc["week_start"][::-1]]),
            y=pd.concat([fc["forecast_upper"], fc["forecast_lower"][::-1]]),
            fill="toself", fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence band", showlegend=True,
        ))
        fig2.add_trace(go.Scatter(
            x=fc["week_start"], y=fc["forecast"], mode="lines+markers", name="FORESIGHT forecast",
            line=dict(color="#38BDF8", width=2.4), marker=dict(size=5),
        ))

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="Units / week", showgrid=True, gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    row = intel_df[(intel_df["sku_id"] == chosen_sku_id) & (intel_df["store_id"] == store_choice)]
    if not row.empty:
        r = row.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current stock", f"{r['current_stock']:.0f} units")
        m2.metric("Forecast avg demand/wk", f"{r['forecast_avg_weekly']:.1f}")
        m3.metric("Weeks of cover", f"{r['weeks_of_cover']:.1f} wk")
        m4.metric("Risk status", r["risk_status"])

# --- TAB 3: RECOMMENDATIONS ---------------------------------------------------
with tab_reco:
    st.markdown('<div class="section-eyebrow">Actionable Recommendations</div>', unsafe_allow_html=True)

    reco_tab1, reco_tab2 = st.tabs(["Reorder Now (Stockout Risk)", "Markdown Candidates (Overstock Risk)"])

    with reco_tab1:
        stockouts = filtered[filtered["risk_status"] == "Stockout Risk"].sort_values(
            "stockout_lost_sales_estimate", ascending=False
        )
        if stockouts.empty:
            st.info("No stockout-risk lines match the current filters.")
        else:
            st.dataframe(
                stockouts[[
                    "sku_id", "sku_name", "category", "store_id", "region", "current_stock",
                    "weeks_of_cover", "lead_time_days", "recommended_order_qty", "stockout_lost_sales_estimate"
                ]].rename(columns={
                    "sku_id": "SKU", "sku_name": "Name", "category": "Category", "store_id": "Site",
                    "region": "Region", "current_stock": "On Hand", "weeks_of_cover": "Weeks Cover",
                    "lead_time_days": "Lead Time (d)", "recommended_order_qty": "Recommended Order Qty",
                    "stockout_lost_sales_estimate": "Est. Lost Sales ($)",
                }),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download reorder plan (CSV)",
                stockouts.to_csv(index=False).encode("utf-8"),
                "foresight_reorder_plan.csv",
                "text/csv",
            )

    with reco_tab2:
        overstocks = filtered[filtered["risk_status"] == "Overstock Risk"].sort_values(
            "potential_markdown_value", ascending=False
        )
        if overstocks.empty:
            st.info("No overstock-risk lines match the current filters.")
        else:
            st.dataframe(
                overstocks[[
                    "sku_id", "sku_name", "category", "store_id", "region", "current_stock",
                    "weeks_of_cover", "excess_units", "potential_markdown_value"
                ]].rename(columns={
                    "sku_id": "SKU", "sku_name": "Name", "category": "Category", "store_id": "Site",
                    "region": "Region", "current_stock": "On Hand", "weeks_of_cover": "Weeks Cover",
                    "excess_units": "Excess Units", "potential_markdown_value": "Capital Tied Up ($)",
                }),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download markdown candidates (CSV)",
                overstocks.to_csv(index=False).encode("utf-8"),
                "foresight_markdown_candidates.csv",
                "text/csv",
            )

# --- TAB 4: METHODOLOGY -------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-eyebrow">How FORESIGHT works</div>', unsafe_allow_html=True)
    st.markdown(
        """
**1. Forecasting engine.** Each SKU-site combination is treated as its own weekly time series.
FORESIGHT engineers lag features (1–12 weeks), rolling mean/std windows, and seasonal encodings,
then trains an **XGBoost + LightGBM ensemble** per series. The two models are averaged, and their
disagreement (plus recent volatility) drives the confidence band shown in the Demand Forecast tab.
Forecasts are generated recursively, 12 weeks ahead.

**2. Risk classification.** Weeks of cover (current stock ÷ average forecast demand) is compared
against each SKU's supplier lead time:
- **Stockout Risk** — cover falls below ~1.1x lead time; stock will likely run out before a
  replenishment order can land.
- **Overstock Risk** — cover exceeds ~4x lead time; capital is tied up in excess stock with
  markdown/write-off exposure.
- **Healthy** — everything in between.

**3. Recommendations.** For stockout-risk lines, FORESIGHT computes an order-up-to quantity that
restores cover for lead time + a buffer, plus safety stock. For overstock-risk lines, it estimates
excess units and the capital value tied up, to support markdown or reallocation decisions.

*Note: this demo runs on synthetically generated sales history so the full pipeline can be explored
without a live data connection. Swap in real POS/ERP extracts via `data_generator.py` to go to production.*
        """
    )
