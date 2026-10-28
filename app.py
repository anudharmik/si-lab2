"""
FreshTrack — Food Waste Intelligence & Analytics System
Streamlit Production Application
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="FreshTrack | Food Waste Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Design Palette Constants (High-Contrast Quiet-Luxury Theme)
FOREST = "#194830"
MINT = "#225c3d"
GOLD = "#68450e"
SLATE = "#1a465e"
BG_BASE = "#faf8f5"
BG_SURFACE = "#ffffff"
BG_ELEVATED = "#f4f1ec"
TEXT_DARK = "#0a130e"
TEXT_MUTED = "#2e4034"
BORDER = "rgba(25, 45, 32, 0.14)"

# Custom Luxury & Cursive CSS Injection
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Alex+Brush&family=Cormorant+Garamond:ital,wght@0,500;0,600;1,400;1,600&family=Outfit:wght@400;500;600;700;800&family=Pinyon+Script&family=Playfair+Display:ital,wght@0,500;0,600;1,400;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background-color: {BG_BASE};
    color: {TEXT_DARK};
}}

[data-testid="stSidebar"] {{
    background-color: {BG_ELEVATED};
    border-right: 1px solid {BORDER};
}}

[data-testid="stSidebar"] * {{
    color: {TEXT_DARK} !important;
}}

[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] {{
    color: #2e4034 !important;
    font-weight: 700 !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Outfit', sans-serif;
    color: {TEXT_DARK} !important;
    font-weight: 800 !important;
}}

p, span, label {{
    color: #1a2920;
}}

button p, button span {{
    color: #ffffff !important;
}}

label {{
    color: #0a130e !important;
    font-weight: 700 !important;
}}

div[data-testid="stCaptionContainer"] {{
    color: #2e4034 !important;
    font-weight: 600 !important;
}}

.hero-container {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 34px 40px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(25, 38, 30, 0.05);
    position: relative;
    overflow: hidden;
}}

.hero-container::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {FOREST}, {GOLD}, {SLATE});
}}

.hero-kicker {{
    font-family: 'Pinyon Script', cursive;
    font-size: 2.2rem;
    color: {GOLD};
    font-weight: 600;
    margin-bottom: 4px;
    line-height: 1.2;
}}

.hero-title {{
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: {TEXT_DARK} !important;
    margin: 0 0 10px 0;
}}

.hero-title span.serif-italic {{
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    color: {FOREST};
    border-bottom: 2px solid {GOLD};
}}

.hero-desc {{
    color: #1a2920 !important;
    font-size: 1.05rem;
    font-weight: 500;
    max-width: 820px;
    line-height: 1.6;
    margin: 0;
}}

.section-eyebrow {{
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1a2920 !important;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-script {{
    font-family: 'Pinyon Script', cursive;
    font-size: 1.65rem;
    color: {GOLD} !important;
    font-weight: 600;
    text-transform: none;
    letter-spacing: 0.02em;
}}

.kpi-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 2px 10px rgba(25, 38, 30, 0.04);
    height: 100%;
}}

.kpi-label {{
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #1a2920;
    margin-bottom: 8px;
}}

.kpi-value {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.15rem;
    font-weight: 800;
    color: {TEXT_DARK} !important;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}}

.kpi-cursive-sub {{
    font-family: 'Pinyon Script', cursive;
    font-size: 1.45rem;
    color: {GOLD};
    font-weight: 600;
    margin-top: 4px;
    display: block;
}}

.kpi-note {{
    font-size: 0.8rem;
    color: #2e4034;
    font-weight: 600;
    margin-top: 6px;
}}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, {FOREST} 0%, #102e1f 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 9999px !important;
    padding: 8px 16px !important;
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(25, 72, 48, 0.25) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button *,
.stButton > button p,
.stButton > button span,
.stButton > button div,
div[data-testid="stFormSubmitButton"] > button *,
div[data-testid="stFormSubmitButton"] > button p,
div[data-testid="stFormSubmitButton"] > button span,
div[data-testid="stFormSubmitButton"] > button div,
div[data-testid="stDownloadButton"] > button *,
div[data-testid="stDownloadButton"] > button p,
div[data-testid="stDownloadButton"] > button span,
div[data-testid="stDownloadButton"] > button div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #225c3d 0%, #153a27 100%) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(25, 72, 48, 0.35) !important;
}

.stButton > button:hover *,
.stButton > button:hover p,
.stButton > button:hover span,
div[data-testid="stFormSubmitButton"] > button:hover *,
div[data-testid="stFormSubmitButton"] > button:hover p,
div[data-testid="stFormSubmitButton"] > button:hover span,
div[data-testid="stDownloadButton"] > button:hover *,
div[data-testid="stDownloadButton"] > button:hover p,
div[data-testid="stDownloadButton"] > button:hover span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton > button:active,
.stButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:active,
div[data-testid="stFormSubmitButton"] > button:focus,
div[data-testid="stDownloadButton"] > button:active,
div[data-testid="stDownloadButton"] > button:focus {
    color: #ffffff !important;
    border-color: rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.35) !important;
}

.stButton > button:active *,
.stButton > button:focus *,
div[data-testid="stFormSubmitButton"] > button:active *,
div[data-testid="stFormSubmitButton"] > button:focus *,
div[data-testid="stDownloadButton"] > button:active *,
div[data-testid="stDownloadButton"] > button:focus * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-testid="stMetric"] {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(25, 38, 30, 0.03);
}}

div[data-testid="stMetricLabel"] p {{
    color: #1a2920 !important;
    font-weight: 800 !important;
    font-size: 0.84rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

div[data-testid="stMetricValue"] {{
    color: {TEXT_DARK} !important;
    font-weight: 800 !important;
    font-family: 'Outfit', sans-serif !important;
}}

/* ---- Multiselect: tag chips ---- */
[data-baseweb="tag"] {{
    background-color: {FOREST} !important;
    border-radius: 6px !important;
    border: none !important;
}}

[data-baseweb="tag"] span {{
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}}

[data-baseweb="tag"] [aria-label="Delete"],
[data-baseweb="tag"] button {{
    color: rgba(255,255,255,0.8) !important;
}}

[data-baseweb="tag"] [aria-label="Delete"]:hover,
[data-baseweb="tag"] button:hover {{
    background: rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
}}

/* ---- Multiselect: input text typed in the box ---- */
[data-baseweb="select"] input {{
    color: {TEXT_DARK} !important;
    caret-color: {FOREST} !important;
    -webkit-text-fill-color: {TEXT_DARK} !important;
    font-weight: 600 !important;
}}

/* ---- Multiselect: dropdown container ---- */
[data-baseweb="select"] > div {{
    background-color: {BG_SURFACE} !important;
    border-color: rgba(25, 72, 48, 0.25) !important;
}}

/* ---- Multiselect: dropdown option list ---- */
[data-baseweb="popover"] li {{
    color: {TEXT_DARK} !important;
    font-weight: 500 !important;
    background-color: {BG_SURFACE} !important;
}}

[data-baseweb="popover"] li:hover {{
    background-color: rgba(25, 72, 48, 0.08) !important;
}}

/* ---- Sidebar multiselect label ---- */
[data-testid="stSidebar"] [data-baseweb="tag"] {{
    background-color: {FOREST} !important;
}}

[data-testid="stSidebar"] [data-baseweb="tag"] span {{
    color: #ffffff !important;
}}

/* ---- Text inputs (search boxes) ---- */
[data-testid="stTextInput"] input {{
    color: {TEXT_DARK} !important;
    -webkit-text-fill-color: {TEXT_DARK} !important;
    font-weight: 600 !important;
    caret-color: {FOREST} !important;
}}

[data-testid="stTextInput"] input::placeholder {{
    color: #3f5246 !important;
    -webkit-text-fill-color: #3f5246 !important;
    opacity: 0.8 !important;
}}
</style>
""", unsafe_allow_html=True)

# Data Path & Features
DATA_PATH = Path(__file__).parent / "food_waste_data.csv"
FEATURES = [
    "combined figures (kg/capita/year)",
    "Household estimate (kg/capita/year)",
    "Retail estimate (kg/capita/year)",
    "Food service estimate (kg/capita/year)",
    "Confidence in estimate",
    "Region",
]
TARGET = "Household estimate (tonnes/year)"

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()
    return data

@st.cache_resource
def train_model(data):
    needed = FEATURES + [TARGET]
    clean = data[needed].copy()
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    for c in FEATURES[:4]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=needed)
    if len(clean) < 10:
        raise ValueError("At least 10 complete records are needed to train the model.")
    
    X, y = clean[FEATURES], clean[TARGET]
    categorical = ["Confidence in estimate", "Region"]
    numeric = [c for c in FEATURES if c not in categorical]
    
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2)),
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "test_actual": y_test.tolist(),
        "test_pred": predictions.tolist()
    }
    # Fit full model on all records for live simulations
    model.fit(X, y)
    return model, metrics, len(clean)

def main():
    # Sidebar Navigation & Controls
    with st.sidebar:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: {FOREST}; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white;">
                🌱
            </div>
            <div>
                <span style="font-family: 'Outfit'; font-size: 1.25rem; font-weight: 800; color: {TEXT_DARK};">FreshTrack</span>
                <span style="font-family: 'Pinyon Script'; font-size: 1.4rem; color: {GOLD}; margin-left: 4px;">Observatory</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("UNITED NATIONS FOOD WASTE INTELLIGENCE")
        st.markdown("---")
        
        page = st.radio("OBSERVATORY CHAPTERS", [
            "Executive Overview",
            "Country Rankings",
            "Waste Sources Analysis",
            "Regional Benchmarks",
            "ML Prediction Studio",
            "Model Evaluation",
            "Data Registry"
        ])
        
        st.markdown("---")
        st.markdown("**Dataset Controls**")
        uploaded = st.file_uploader("Upload custom CSV (Optional)", type=["csv"], help="Use official UNEP food-waste CSV schema")

    df = load_data(uploaded)
    if df.empty:
        st.error("Dataset not found. Please ensure `food_waste_data.csv` is present in the repository root.")
        st.stop()

    required_cols = [c for c in FEATURES + [TARGET, "Country", "Region"] if c not in df.columns]
    if required_cols:
        st.error("The dataset is missing required columns: " + ", ".join(required_cols))
        st.stop()

    for col in FEATURES[:4] + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Country", "Region", TARGET])
    df["Country"] = df["Country"].astype(str)
    df["Region"] = df["Region"].astype(str)

    # Sidebar Filter Controls
    with st.sidebar:
        regions_list = sorted(df["Region"].dropna().unique().tolist())
        chosen_regions = st.multiselect("Filter Regions", regions_list, default=regions_list)
        conf_col = "Confidence in estimate"
        conf_list = sorted(df[conf_col].dropna().unique().tolist())
        chosen_conf = st.multiselect("Confidence Level", conf_list, default=conf_list)

    filtered = df[df["Region"].isin(chosen_regions) & df[conf_col].isin(chosen_conf)].copy()
    if filtered.empty:
        st.warning("No records match the selected filters. Please adjust the sidebar filter options.")
        st.stop()

    # Hero Banner
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-kicker">A Comprehensive Planetary Food Loss & Preservation Dossier</div>
        <h1 class="hero-title">Understanding Global Food Waste: <span class="serif-italic">931 Million Tonnes</span> Lost Annually</h1>
        <p class="hero-desc">
            Empirical baseline derived from the official United Nations Environment Programme (UNEP) Food Waste Index. Explore country-level estimates across 214 nations, examine sectoral breakdowns, and run client-side Random Forest machine learning predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Executive Overview
    if page == "Executive Overview":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Territorial Registry</span> • Global Pulse</div>', unsafe_allow_html=True)
        st.subheader("Global Food Waste Pulse & Executive Metrics")
        
        total_tonnes = filtered[TARGET].sum() + filtered["Retail estimate (tonnes/year)"].sum() + filtered["Food service estimate (tonnes/year)"].sum()
        hh_tonnes = filtered[TARGET].sum()
        hh_share = (hh_tonnes / total_tonnes * 100) if total_tonnes > 0 else 0
        avg_combined = filtered["combined figures (kg/capita/year)"].mean()
        
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Annual Waste</div>
                <div class="kpi-value">{total_tonnes/1e6:,.1f}M t</div>
                <span class="kpi-cursive-sub">aggregate planetary loss</span>
                <div class="kpi-note">Sum of household, retail & service</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Household Share</div>
                <div class="kpi-value">{hh_share:.1f}%</div>
                <span class="kpi-cursive-sub">private kitchens & tables</span>
                <div class="kpi-note">{hh_tonnes/1e6:,.1f}M tonnes domestic discard</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Per-Person Average</div>
                <div class="kpi-value">{avg_combined:.1f} kg</div>
                <span class="kpi-cursive-sub">individual yearly discard</span>
                <div class="kpi-note">Combined annual footprint per person</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Coverage</div>
                <div class="kpi-value">{filtered['Country'].nunique()}</div>
                <span class="kpi-cursive-sub">UN M49 territories</span>
                <div class="kpi-note">Active nations in current view</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Overview Charts
        col_left, col_right = st.columns([1.1, 0.9])
        with col_left:
            st.markdown("#### Top 10 Nations by Household Volume (Tonnes/Year)")
            top_tonnes = filtered.nlargest(10, TARGET).sort_values(TARGET, ascending=True)
            fig = px.bar(
                top_tonnes,
                x=TARGET,
                y="Country",
                orientation="h",
                color_discrete_sequence=[FOREST],
                text_auto=".2s"
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=420,
                font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
                xaxis_title="Household Waste (Tonnes/Year)",
                xaxis=dict(tickfont=dict(color=TEXT_DARK, size=12), title_font=dict(color=TEXT_DARK)),
                yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12, weight="bold")),
                yaxis_title="",
                margin=dict(l=10, r=20, t=10, b=10)
            )
            fig.update_traces(textfont_color="white", textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.markdown("#### Sector Discard Proportions")
            sector_sums = {
                "Household": filtered[TARGET].sum(),
                "Food Service": filtered["Food service estimate (tonnes/year)"].sum(),
                "Retail": filtered["Retail estimate (tonnes/year)"].sum()
            }
            fig_pie = px.pie(
                names=list(sector_sums.keys()),
                values=list(sector_sums.values()),
                color_discrete_sequence=[FOREST, SLATE, GOLD],
                hole=0.62
            )
            fig_pie.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=420,
                font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                            font=dict(color=TEXT_DARK, size=13))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # 2. Country Rankings
    elif page == "Country Rankings":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">National Dispatches</span> • Rankings</div>', unsafe_allow_html=True)
        st.subheader("Country Rankings & Explorer")
        
        metric_choice = st.radio(
            "Ranking Metric",
            ["combined figures (kg/capita/year)", TARGET, "Household estimate (kg/capita/year)"],
            format_func=lambda m: "Combined (kg/capita/year)" if "combined" in m else ("Household Volume (Tonnes/year)" if "tonnes" in m else "Household (kg/capita/year)"),
            horizontal=True
        )
        
        top15 = filtered.nlargest(15, metric_choice).sort_values(metric_choice, ascending=True)
        fig_rank = px.bar(
            top15,
            x=metric_choice,
            y="Country",
            orientation="h",
            color=metric_choice,
            color_continuous_scale=[MINT, FOREST],
            text_auto=".2s" if "tonnes" in metric_choice else ".1f"
        )
        fig_rank.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=500,
            font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
            xaxis=dict(tickfont=dict(color=TEXT_DARK, size=12), title_font=dict(color=TEXT_DARK)),
            yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12, weight="bold")),
            coloraxis_showscale=False,
            margin=dict(l=10, r=20, t=10, b=10)
        )
        fig_rank.update_traces(textfont_color=TEXT_DARK, textfont_size=12)
        st.plotly_chart(fig_rank, use_container_width=True)

    # 3. Waste Sources Analysis
    elif page == "Waste Sources Analysis":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Empirical Distribution</span> • Sector Breakdown</div>', unsafe_allow_html=True)
        st.subheader("Where Does Food Waste Originate?")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid {FOREST};">
                <div class="kpi-label">Household Waste</div>
                <div class="kpi-value">{filtered['Household estimate (kg/capita/year)'].mean():.1f} kg</div>
                <span class="kpi-cursive-sub">per person / year</span>
                <div class="kpi-note">Represents ~61.2% of global volume</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid {SLATE};">
                <div class="kpi-label">Food Service</div>
                <div class="kpi-value">{filtered['Food service estimate (kg/capita/year)'].mean():.1f} kg</div>
                <span class="kpi-cursive-sub">per person / year</span>
                <div class="kpi-note">Catering, dining & hospitality (~25.7%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid {GOLD};">
                <div class="kpi-label">Retail Groceries</div>
                <div class="kpi-value">{filtered['Retail estimate (kg/capita/year)'].mean():.1f} kg</div>
                <span class="kpi-cursive-sub">per person / year</span>
                <div class="kpi-note">Supermarkets & market distribution (~13.1%)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        source_df = pd.DataFrame([
            {"Sector": "Household", "Average kg/capita": filtered['Household estimate (kg/capita/year)'].mean(), "Total Tonnes": filtered[TARGET].sum()},
            {"Sector": "Food Service", "Average kg/capita": filtered['Food service estimate (kg/capita/year)'].mean(), "Total Tonnes": filtered["Food service estimate (tonnes/year)"].sum()},
            {"Sector": "Retail", "Average kg/capita": filtered['Retail estimate (kg/capita/year)'].mean(), "Total Tonnes": filtered["Retail estimate (tonnes/year)"].sum()},
        ])
        
        b1, b2 = st.columns(2)
        with b1:
            fig_b1 = px.bar(source_df, x="Sector", y="Average kg/capita", color="Sector", color_discrete_sequence=[FOREST, SLATE, GOLD], text_auto=".1f")
            fig_b1.update_layout(
                plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False,
                font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
                xaxis=dict(tickfont=dict(color=TEXT_DARK, size=13, weight="bold")),
                yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12))
            )
            fig_b1.update_traces(textfont_color="white", textfont_size=13)
            st.plotly_chart(fig_b1, use_container_width=True)
        with b2:
            fig_b2 = px.bar(source_df, x="Sector", y="Total Tonnes", color="Sector", color_discrete_sequence=[FOREST, SLATE, GOLD], text_auto=".2s")
            fig_b2.update_layout(
                plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False,
                font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
                xaxis=dict(tickfont=dict(color=TEXT_DARK, size=13, weight="bold")),
                yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12))
            )
            fig_b2.update_traces(textfont_color="white", textfont_size=13)
            st.plotly_chart(fig_b2, use_container_width=True)

    # 4. Regional Benchmarks
    elif page == "Regional Benchmarks":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Continental Benchmarks</span> • Regional Comparisons</div>', unsafe_allow_html=True)
        st.subheader("Regional Waste Comparison across 17 Global Zones")
        
        regional_summary = filtered.groupby("Region").agg(
            Countries=("Country", "nunique"),
            Total_Household=(TARGET, "sum"),
            Avg_Combined=("combined figures (kg/capita/year)", "mean"),
            Avg_Household=("Household estimate (kg/capita/year)", "mean")
        ).reset_index().sort_values("Avg_Combined", ascending=True)
        
        fig_reg = px.bar(
            regional_summary,
            x="Avg_Combined",
            y="Region",
            orientation="h",
            color="Avg_Combined",
            color_continuous_scale=[MINT, FOREST],
            text_auto=".1f"
        )
        fig_reg.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=540,
            font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
            coloraxis_showscale=False,
            xaxis_title="Combined Waste (kg per capita per year)",
            xaxis=dict(tickfont=dict(color=TEXT_DARK, size=12), title_font=dict(color=TEXT_DARK)),
            yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12, weight="bold")),
            yaxis_title=""
        )
        fig_reg.update_traces(textfont_color="white", textfont_size=12)
        st.plotly_chart(fig_reg, use_container_width=True)

    # 5. ML Prediction Studio
    elif page == "ML Prediction Studio":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Predictive Intelligence</span> • Random Forest 200 Trees</div>', unsafe_allow_html=True)
        st.subheader("Household Food Waste Machine Learning Estimator")
        st.markdown("Direct implementation of the Random Forest ensemble trained on official UNEP records.")
        
        try:
            model, metrics, n_rows = train_model(df)
            
            # Preset buttons
            st.markdown("**Quick Preset Scenarios:**")
            p_cols = st.columns(4)
            preset = "custom"
            if p_cols[0].button("🇦🇺 Australia (Notebook Ex.)", use_container_width=True):
                preset = "australia"
            if p_cols[1].button("🌍 Global Average", use_container_width=True):
                preset = "global"
            if p_cols[2].button("📈 High Waste Profile", use_container_width=True):
                preset = "high"
            if p_cols[3].button("📉 Low Waste Profile", use_container_width=True):
                preset = "low"
                
            defaults = {
                "australia": {"comb": 133.0, "hh": 102.0, "ret": 9.0, "fs": 22.0, "conf": "High Confidence", "reg": "Australia and New Zealand"},
                "global": {"comb": 126.8, "hh": 84.3, "ret": 15.1, "fs": 27.4, "conf": "Medium Confidence", "reg": "Western Europe"},
                "high": {"comb": 185.0, "hh": 120.0, "ret": 28.0, "fs": 37.0, "conf": "High Confidence", "reg": "Northern America"},
                "low": {"comb": 78.0, "hh": 52.0, "ret": 8.0, "fs": 18.0, "conf": "High Confidence", "reg": "Southern Asia"},
                "custom": {"comb": 133.0, "hh": 102.0, "ret": 9.0, "fs": 22.0, "conf": "High Confidence", "reg": "Australia and New Zealand"}
            }[preset]
            
            with st.form("ml_estimator_form"):
                f1, f2 = st.columns(2)
                with f1:
                    comb_in = st.slider("Combined Figures (kg/capita/year)", 40.0, 260.0, defaults["comb"], 1.0)
                    hh_in = st.slider("Household Estimate (kg/capita/year)", 20.0, 190.0, defaults["hh"], 1.0)
                    ret_in = st.slider("Retail Estimate (kg/capita/year)", 2.0, 60.0, defaults["ret"], 1.0)
                with f2:
                    fs_in = st.slider("Food Service Estimate (kg/capita/year)", 2.0, 90.0, defaults["fs"], 1.0)
                    conf_in = st.selectbox("Estimate Confidence", sorted(df[conf_col].dropna().unique()), index=sorted(df[conf_col].dropna().unique()).index(defaults["conf"]) if defaults["conf"] in df[conf_col].values else 0)
                    reg_in = st.selectbox("Region", sorted(df["Region"].dropna().unique()), index=sorted(df["Region"].dropna().unique()).index(defaults["reg"]) if defaults["reg"] in df["Region"].values else 0)
                
                submitted = st.form_submit_button("⚡ Compute Machine Learning Estimate", use_container_width=True)
            
            input_df = pd.DataFrame([{
                FEATURES[0]: comb_in,
                FEATURES[1]: hh_in,
                FEATURES[2]: ret_in,
                FEATURES[3]: fs_in,
                FEATURES[4]: conf_in,
                FEATURES[5]: reg_in
            }], columns=FEATURES)
            
            prediction = float(model.predict(input_df)[0])
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f7f9f7 0%, #f0f5f2 100%); border: 1px solid rgba(37, 90, 64, 0.18); border-radius: 16px; padding: 26px; text-align: center; margin: 20px 0;">
                <span style="font-family: 'Pinyon Script', cursive; font-size: 1.6rem; color: {GOLD}; display: block;">Random Forest 200-Tree Projection</span>
                <span style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: {TEXT_MUTED};">Estimated National Household Waste</span>
                <div style="font-family: 'Playfair Display', serif; font-style: italic; font-size: 2.85rem; font-weight: 700; color: {FOREST}; margin: 6px 0;">
                    {prediction:,.0f} tonnes / year
                </div>
                <span style="font-size: 0.85rem; color: #526359;">Trained on {n_rows} standardized UNEP country records</span>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error training Random Forest model: {e}")

    # 6. Model Evaluation
    elif page == "Model Evaluation":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Statistical Rigor</span> • Evaluation Metrics</div>', unsafe_allow_html=True)
        st.subheader("Model Validation & Test Performance")
        try:
            model, metrics, n_rows = train_model(df)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Mean Absolute Error (MAE)", f"{metrics['MAE']/1e6:.2f}M t", help="Mean absolute prediction discrepancy on test set")
            with m2:
                st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']/1e6:.2f}M t", help="Weighted penalty for larger deviations")
            with m3:
                st.metric("Coefficient of Determination (R²)", f"{metrics['R2']:.3f}", help="Variance accounted for by features")
                
            st.markdown("#### Test Split: Actual vs. Predicted Parity (Million Tonnes)")
            test_df = pd.DataFrame({
                "Actual_M": [v / 1e6 for v in metrics["test_actual"]],
                "Predicted_M": [v / 1e6 for v in metrics["test_pred"]]
            })
            max_v = max(test_df["Actual_M"].max(), test_df["Predicted_M"].max()) * 1.1
            
            fig_eval = px.scatter(test_df, x="Actual_M", y="Predicted_M", color_discrete_sequence=[FOREST])
            fig_eval.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", line=dict(dash="dash", color=GOLD), name="Ideal Parity Line"))
            fig_eval.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color=TEXT_DARK, family="Plus Jakarta Sans, sans-serif"),
                xaxis_title="Actual Test Waste (Million tonnes/year)",
                xaxis=dict(tickfont=dict(color=TEXT_DARK, size=12), title_font=dict(color=TEXT_DARK)),
                yaxis_title="Predicted Test Waste (Million tonnes/year)",
                yaxis=dict(tickfont=dict(color=TEXT_DARK, size=12), title_font=dict(color=TEXT_DARK)),
                legend=dict(font=dict(color=TEXT_DARK)),
                height=450
            )
            st.plotly_chart(fig_eval, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load evaluation metrics: {e}")

    # 7. Data Registry
    elif page == "Data Registry":
        st.markdown(f'<div class="section-eyebrow"><span class="section-script">Archival Repository</span> • Complete Records</div>', unsafe_allow_html=True)
        st.subheader("Official Country Dataset Registry")
        
        search_query = st.text_input("Search nation name...", placeholder="Type to filter countries...")
        view_data = filtered.copy()
        if search_query:
            view_data = view_data[view_data["Country"].str.contains(search_query, case=False, na=False)]
            
        st.dataframe(view_data, use_container_width=True, hide_index=True)
        
        c_csv, c_full = st.columns(2)
        with c_csv:
            st.download_button(
                "⬇ Export Filtered Dataset (CSV)",
                data=view_data.to_csv(index=False).encode("utf-8"),
                file_name="freshtrack_filtered_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_full:
            st.download_button(
                "⬇ Export Full UNEP Dataset (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="freshtrack_complete_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 12px 0;">
        <span style="font-family: 'Pinyon Script', cursive; font-size: 1.6rem; color: {GOLD};">"Every harvest preserved is a world sustained."</span>
        <span style="font-size: 0.8rem; color: {TEXT_MUTED};">UNEP Food Waste Index Repository • Production Streamlit Edition</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
