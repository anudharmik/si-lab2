import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

st.set_page_config(page_title="FreshTrack | Food Waste Intelligence", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")

GREEN = "#176B45"
DARK = "#12372A"
MINT = "#E7F5EC"
GOLD = "#E6B655"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#F6F8F5;color:#172B22;}
[data-testid="stSidebar"] {background:#12372A;}
[data-testid="stSidebar"] * {color:#F4FBF6 !important;}
h1,h2,h3 {font-family:'Manrope',sans-serif;color:#12372A;}
.hero {background:linear-gradient(120deg,#12372A 0%,#176B45 68%,#2C8A5D 100%);padding:30px 34px;border-radius:20px;color:white;margin-bottom:20px;}
.hero h1 {color:white;font-size:2.25rem;margin:0 0 8px 0;}
.hero p {color:#E2F2E7;font-size:1rem;margin:0;max-width:780px;}
.eyebrow {text-transform:uppercase;letter-spacing:2px;font-size:.72rem;font-weight:700;color:#BFE5CB;margin-bottom:10px;}
.kpi {background:white;border:1px solid #E3EAE4;border-radius:16px;padding:18px 20px;min-height:120px;box-shadow:0 3px 12px rgba(18,55,42,.035);}
.kpi-label {color:#66766C;font-size:.82rem;font-weight:600;margin-bottom:10px;}
.kpi-value {color:#12372A;font-family:'Manrope',sans-serif;font-size:1.65rem;font-weight:800;line-height:1.2;}
.kpi-note {color:#738278;font-size:.75rem;margin-top:8px;}
.section-note {color:#617268;font-size:.9rem;}
div[data-testid="stMetric"] {background:white;padding:16px 18px;border:1px solid #E3EAE4;border-radius:14px;}
.stButton>button {background:#176B45;color:white;border-radius:10px;border:0;font-weight:700;}
.stButton>button:hover {background:#12372A;color:white;}
</style>
""", unsafe_allow_html=True)

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
        "R²": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
    }
    # Fit final estimator on all rows for the interactive prediction page.
    model.fit(X, y)
    return model, metrics, len(clean)

def fmt_num(value):
    return f"{value:,.0f}"

uploaded = None
with st.sidebar:
    st.markdown("## 🌱 FreshTrack")
    st.caption("FOOD WASTE INTELLIGENCE")
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Overview", "Country Analysis", "Regional Analysis", "Waste Sources",
        "ML Prediction", "Model Performance", "Dataset Explorer", "About Project"
    ])
    st.markdown("---")
    st.markdown("**Data source**")
    st.caption("Food Waste Index country estimates. Use the dataset uploader below if you want to analyse another compatible CSV.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="The CSV should contain the FreshTrack food-waste columns.")

df = load_data(uploaded)
if df.empty:
    st.error("Dataset not found. Place food_waste_data.csv next to app.py or upload a CSV in the sidebar.")
    st.stop()

required = [c for c in FEATURES + [TARGET, "Country", "Region"] if c not in df.columns]
if required:
    st.error("The dataset is missing required columns: " + ", ".join(required))
    st.stop()

for col in FEATURES[:4] + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["Country", "Region", TARGET])
df["Country"] = df["Country"].astype(str)
df["Region"] = df["Region"].astype(str)

with st.sidebar:
    st.markdown("### Filter dashboard")
    regions = sorted(df["Region"].dropna().unique().tolist())
    chosen_regions = st.multiselect("Region", regions, default=regions)
    confidence_col = "Confidence in estimate"
    confidence_values = sorted(df[confidence_col].dropna().unique().tolist())
    chosen_confidence = st.multiselect("Estimate confidence", confidence_values, default=confidence_values)

filtered = df[df["Region"].isin(chosen_regions) & df[confidence_col].isin(chosen_confidence)].copy()
if filtered.empty:
    st.warning("No rows match the selected filters. Select at least one region and confidence level.")
    st.stop()

st.markdown('<div class="hero"><div class="eyebrow">DATA FOR A LESS WASTEFUL WORLD</div><h1>FreshTrack</h1><p>Food Waste Analytics & Prediction System — explore country estimates, compare waste sources, and try a machine-learning estimate in one interactive dashboard.</p></div>', unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Overview dashboard")
    st.markdown('<p class="section-note">A quick snapshot of the currently filtered dataset.</p>', unsafe_allow_html=True)
    total_household = filtered[TARGET].sum()
    avg_per_capita = filtered["combined figures (kg/capita/year)"].mean()
    top_country = filtered.loc[filtered[TARGET].idxmax(), "Country"]
    top_region = filtered.groupby("Region")[TARGET].sum().idxmax()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi"><div class="kpi-label">Countries covered</div><div class="kpi-value">{filtered["Country"].nunique()}</div><div class="kpi-note">In selected filters</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><div class="kpi-label">Household waste estimate</div><div class="kpi-value">{total_household/1e6:,.2f}M</div><div class="kpi-note">Tonnes per year, summed estimates</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi"><div class="kpi-label">Average combined waste</div><div class="kpi-value">{avg_per_capita:,.1f}</div><div class="kpi-note">kg per capita per year</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><div class="kpi-label">Highest household estimate</div><div class="kpi-value">{top_country}</div><div class="kpi-note">Within current filters</div></div>', unsafe_allow_html=True)
    st.write("")
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("#### Highest household waste estimates")
        top = filtered.nlargest(12, TARGET).sort_values(TARGET)
        fig = px.bar(top, x=TARGET, y="Country", orientation="h", color=TARGET, color_continuous_scale=["#B9DEC7",GREEN], labels={TARGET:"Household estimate (tonnes/year)"})
        fig.update_layout(height=470, coloraxis_showscale=False, margin=dict(l=0,r=10,t=15,b=0), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Waste by source")
        source_df = pd.DataFrame({"Source":["Household","Retail","Food service"],"Tonnes/year":[filtered[TARGET].sum(),filtered["Retail estimate (tonnes/year)"].sum(),filtered["Food service estimate (tonnes/year)"].sum()]})
        fig = px.pie(source_df, names="Source", values="Tonnes/year", hole=.62, color="Source", color_discrete_map={"Household":GREEN,"Retail":GOLD,"Food service":"#80B8D2"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=470, margin=dict(l=0,r=0,t=15,b=0), paper_bgcolor="white", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Regional overview")
    region_df = filtered.groupby("Region", as_index=False)[TARGET].sum().sort_values(TARGET, ascending=False)
    fig = px.bar(region_df, x="Region", y=TARGET, color=TARGET, color_continuous_scale=["#B9DEC7",GREEN], labels={TARGET:"Household estimate (tonnes/year)"})
    fig.update_layout(height=350, coloraxis_showscale=False, xaxis_tickangle=-25, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Country Analysis":
    st.subheader("Country analysis")
    countries = sorted(filtered["Country"].unique())
    selected = st.selectbox("Choose a country", countries)
    row = filtered[filtered["Country"] == selected].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Household estimate", f'{row[TARGET]:,.0f} tonnes/year')
    b.metric("Combined waste", f'{row["combined figures (kg/capita/year)"]:,.0f} kg/capita/year')
    c.metric("Region", row["Region"])
    st.markdown("#### Waste source comparison")
    source = pd.DataFrame({"Source":["Household","Retail","Food service"],"kg/capita/year":[row["Household estimate (kg/capita/year)"],row["Retail estimate (kg/capita/year)"],row["Food service estimate (kg/capita/year)"]],"tonnes/year":[row[TARGET],row["Retail estimate (tonnes/year)"],row["Food service estimate (tonnes/year)"]]})
    fig = px.bar(source, x="Source", y="kg/capita/year", color="Source", color_discrete_sequence=[GREEN,GOLD,"#80B8D2"], text_auto=True)
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Country record")
    st.dataframe(row.to_frame("Value"), use_container_width=True)

elif page == "Regional Analysis":
    st.subheader("Regional analysis")
    regional = filtered.groupby("Region", as_index=False).agg(
        Countries=("Country","nunique"),
        **{"Household tonnes/year":(TARGET,"sum"),
           "Average combined kg/capita/year":("combined figures (kg/capita/year)","mean"),
           "Retail tonnes/year":("Retail estimate (tonnes/year)","sum"),
           "Food service tonnes/year":("Food service estimate (tonnes/year)","sum")}
    ).sort_values("Household tonnes/year", ascending=False)
    st.dataframe(regional, use_container_width=True, hide_index=True)
    fig = px.bar(regional.sort_values("Household tonnes/year"), x="Household tonnes/year", y="Region", orientation="h", color="Household tonnes/year", color_continuous_scale=["#B9DEC7",GREEN])
    fig.update_layout(height=max(400, len(regional)*30), coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Average per-capita waste by region")
    fig = px.scatter(filtered, x="combined figures (kg/capita/year)", y="Household estimate (kg/capita/year)", color="Region", hover_name="Country", size=TARGET, size_max=26)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Waste Sources":
    st.subheader("Waste source analysis")
    st.markdown('<p class="section-note">Compare household, retail and food-service estimates for the selected records.</p>', unsafe_allow_html=True)
    source_cols = {
        "Household":("Household estimate (kg/capita/year)",TARGET),
        "Retail":("Retail estimate (kg/capita/year)","Retail estimate (tonnes/year)"),
        "Food service":("Food service estimate (kg/capita/year)","Food service estimate (tonnes/year)")
    }
    source_summary = pd.DataFrame([{"Source":name,"Total tonnes/year":filtered[cols[1]].sum(),"Average kg/capita/year":filtered[cols[0]].mean()} for name,cols in source_cols.items()])
    a,b = st.columns(2)
    with a:
        fig=px.bar(source_summary,x="Source",y="Total tonnes/year",color="Source",color_discrete_sequence=[GREEN,GOLD,"#80B8D2"],text_auto=".3s")
        fig.update_layout(showlegend=False,plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.bar(source_summary,x="Source",y="Average kg/capita/year",color="Source",color_discrete_sequence=[GREEN,GOLD,"#80B8D2"],text_auto=".1f")
        fig.update_layout(showlegend=False,plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    st.dataframe(source_summary,use_container_width=True,hide_index=True)
    st.info("These are published estimates in the dataset, not live measurements from individual waste collection systems.")

elif page == "ML Prediction":
    st.subheader("Household food-waste prediction")
    st.markdown("Enter the country-level features to estimate **Household estimate (tonnes/year)**. The model is trained from the included dataset; this is a demonstration estimate, not an official forecast.")
    try:
        model, metrics, n_rows = train_model(df)
        st.caption(f"Random Forest Regressor • 200 trees • trained from {n_rows} complete dataset rows")
        with st.form("prediction_form"):
            left,right=st.columns(2)
            with left:
                combined=st.number_input("Combined figures (kg/capita/year)",min_value=0.0,max_value=1000.0,value=126.0,step=1.0)
                household=st.number_input("Household estimate (kg/capita/year)",min_value=0.0,max_value=1000.0,value=82.0,step=1.0)
                retail=st.number_input("Retail estimate (kg/capita/year)",min_value=0.0,max_value=1000.0,value=16.0,step=1.0)
            with right:
                food_service=st.number_input("Food service estimate (kg/capita/year)",min_value=0.0,max_value=1000.0,value=28.0,step=1.0)
                confidence=st.selectbox("Confidence in estimate",sorted(df[confidence_col].dropna().unique()))
                region=st.selectbox("Region",sorted(df["Region"].dropna().unique()))
            submitted=st.form_submit_button("Predict household waste")
        if submitted:
            input_row=pd.DataFrame([{
                FEATURES[0]:combined,FEATURES[1]:household,FEATURES[2]:retail,
                FEATURES[3]:food_service,FEATURES[4]:confidence,FEATURES[5]:region
            }],columns=FEATURES)
            prediction=float(model.predict(input_row)[0])
            st.success(f"Estimated household food waste: {prediction:,.0f} tonnes/year")
            st.caption("This prediction reflects patterns learned from the dataset. It should not be treated as a measured real-world value.")
        with st.expander("What features does the model use?"):
            st.write("Combined, household, retail and food-service waste per capita, estimate confidence, and region.")
    except Exception as e:
        st.error(f"Model could not be trained from the current data: {e}")

elif page == "Model Performance":
    st.subheader("Model performance")
    st.markdown("Metrics below are calculated on a held-out 20% test split (random_state=42). The model is then refitted on all rows for the prediction demo.")
    try:
        model, metrics, n_rows = train_model(df)
        a,b,c=st.columns(3)
        a.metric("MAE",f'{metrics["MAE"]:,.0f}',help="Mean Absolute Error: average absolute prediction error in tonnes/year.")
        b.metric("RMSE",f'{metrics["RMSE"]:,.0f}',help="Root Mean Squared Error: gives larger errors more weight.")
        c.metric("R²",f'{metrics["R²"]:.3f}',help="Coefficient of determination; may be negative on a weak test fit.")
        st.write(f"Training rows: **{metrics['train_rows']}** · Test rows: **{metrics['test_rows']}** · Complete rows: **{n_rows}**")
        st.info("These values are computed when the app runs on the current CSV. They are not copied from a report and may change if the dataset changes.")
    except Exception as e:
        st.error(f"Could not calculate metrics: {e}")

elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    st.write(f"Showing **{len(filtered):,}** rows after filters.")
    search = st.text_input("Search country name", placeholder="Type a country...")
    view = filtered.copy()
    if search:
        view=view[view["Country"].str.contains(search,case=False,na=False)]
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("⬇ Download filtered data as CSV",data=view.to_csv(index=False).encode("utf-8"),file_name="freshtrack_filtered_data.csv",mime="text/csv")
    st.download_button("⬇ Download full dataset as CSV",data=df.to_csv(index=False).encode("utf-8"),file_name="freshtrack_full_data.csv",mime="text/csv")

else:
    st.subheader("About FreshTrack")
    st.markdown("""
    ### Project objective
    FreshTrack is a food-waste analytics and prediction website designed to make country-level food-waste estimates easier to explore and compare.

    ### Technology stack
    - **Python** — application logic
    - **Streamlit** — interactive website and navigation
    - **Pandas / NumPy** — data cleaning and analysis
    - **Plotly** — interactive charts
    - **scikit-learn** — Random Forest regression model and evaluation
    - **joblib** — available for model persistence/integration

    ### Main modules
    1. Overview dashboard and key indicators
    2. Country and regional analysis
    3. Household, retail and food-service comparisons
    4. Machine-learning prediction demonstration
    5. Model performance metrics
    6. Dataset exploration and CSV export

    ### Limitations
    The dataset contains estimates, not real-time sensor data. A prediction is only as reliable as the data and validation process behind it. The test-set metrics should be interpreted carefully because the dataset is relatively small and country-level estimates can vary in quality.

    ### Future scope
    Add updated official datasets, more robust cross-validation, model versioning, and a hosted dashboard for wider access.
    """)
    st.markdown("**Project name:** FreshTrack — Food Waste Analytics and Prediction System")
    st.markdown("**Data file:** `food_waste_data.csv`")
