from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------------------------
# AQUifERGUARD | FRONTEND REDESIGN
# Existing model, features, target and project threshold are preserved.
# -------------------------------------------------------------------

st.set_page_config(
    page_title="AquiferGuard | Groundwater Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aquiferguard_model.pkl"
DATA_PATH = ROOT / "dataset" / "raw" / "cgwb_punjab_groundwater_quality_raw.csv"

THRESHOLD = 0.03
FEATURES = [
    "Potential of Hydrogen (pH)",
    "Electric Conductivity (μS/cm)",
    "Bicarbonate (mg/L)",
    "Chloride (mg/L)",
    "Sodium (mg/L)",
]
FEATURE_LABELS = ["pH", "Electrical Conductivity", "Bicarbonate", "Chloride", "Sodium"]


# ----------------------------- DATA ---------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["Uranium(mg/L)"]).drop_duplicates().copy()
    data["Uranium_Risk"] = (data["Uranium(mg/L)"] > THRESHOLD).astype(int)
    data["Risk Category"] = data["Uranium_Risk"].map(
        {1: "Elevated", 0: "Not Elevated"}
    )
    if "District" in data.columns:
        data["District"] = data["District"].fillna("Unknown").astype(str)
    return data


try:
    model = load_model()
    df = load_dataset()
except Exception as error:
    st.error("AquiferGuard could not load a required project file.")
    st.code(str(error))
    st.info(
        "Check that the model and dataset folders are present in the repository "
        "and that Streamlit is launched from the project structure."
    )
    st.stop()


# ------------------------------ STYLE -------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --panel: #0d1b2b;
        --panel-soft: #102338;
        --border: rgba(148, 177, 201, 0.14);
        --text: #edf7ff;
        --muted: #91a9bd;
        --teal: #20d6c2;
        --cyan: #42c9f5;
        --green: #59dfad;
        --amber: #f6bd60;
        --red: #ff7b8b;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 78% 2%, rgba(20, 164, 169, 0.13), transparent 28%),
            radial-gradient(circle at 4% 28%, rgba(43, 112, 173, 0.10), transparent 25%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.78);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1929 0%, #091522 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }

    p, li, label, .stCaption {
        color: #bfd0df;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 39, 59, 0.96), rgba(12, 29, 45, 0.96));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 18px 20px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.13);
    }

    [data-testid="stMetricLabel"] {
        color: #a8bfd1 !important;
        font-size: 0.86rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #f3fbff !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }

    div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border) !important;
    }

    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border: 1px solid rgba(32, 214, 194, 0.42);
        border-radius: 11px;
        color: #04151b;
        background: linear-gradient(100deg, #20d6c2, #42c9f5);
        font-weight: 700;
        min-height: 2.7rem;
        transition: all 160ms ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: #7af1e4;
        color: #04151b;
        box-shadow: 0 0 22px rgba(32, 214, 194, 0.20);
        transform: translateY(-1px);
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMultiSelect"] div[data-baseweb="select"] {
        background: #102338;
        color: var(--text);
        border-radius: 10px;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 2px 18px 2px;
    }

    .brand-mark {
        width: 43px;
        height: 43px;
        display: grid;
        place-items: center;
        border-radius: 14px;
        font-size: 23px;
        background: linear-gradient(145deg, rgba(32,214,194,.24), rgba(66,201,245,.12));
        border: 1px solid rgba(32,214,194,.35);
        box-shadow: 0 0 24px rgba(32,214,194,.10);
    }

    .brand-name {
        color: #f2fbff;
        font: 800 1.12rem 'Manrope', sans-serif;
        letter-spacing: -0.04em;
    }

    .brand-caption {
        color: #85a1b7;
        font-size: .69rem;
        letter-spacing: .11em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    .eyebrow {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .16em;
        margin-bottom: 8px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        border: 1px solid rgba(67, 201, 213, .22);
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 20%, rgba(32,214,194,.19), transparent 27%),
            linear-gradient(112deg, #10283b 0%, #102538 48%, #0c2732 100%);
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.12;
        margin: 0 0 12px 0;
    }

    .hero p {
        max-width: 720px;
        color: #afc8d8;
        font-size: .98rem;
        line-height: 1.65;
        margin: 0;
    }

    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 20px;
        padding: 7px 11px;
        border: 1px solid rgba(32,214,194,.28);
        border-radius: 999px;
        background: rgba(32,214,194,.08);
        color: #8ef2e4;
        font-size: .75rem;
        font-weight: 600;
    }

    .section-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 22px 0 13px 0;
        color: #eaf6ff;
        font: 700 1.08rem 'Manrope', sans-serif;
    }

    .section-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 12px rgba(32,214,194,.65);
    }

    .panel {
        background: linear-gradient(145deg, rgba(16, 35, 54, .96), rgba(11, 27, 43, .96));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 18px 19px;
        height: 100%;
    }

    .panel-label {
        color: #91aabd;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .11em;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .panel-value {
        color: #f4fbff;
        font: 800 1.75rem 'Manrope', sans-serif;
        letter-spacing: -.04em;
    }

    .panel-note {
        color: #91aabd;
        font-size: .79rem;
        margin-top: 5px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 700;
        background: rgba(32,214,194,.12);
        border: 1px solid rgba(32,214,194,.25);
        color: #7eeadd;
    }

    .status-pill.warning {
        background: rgba(246,189,96,.12);
        border-color: rgba(246,189,96,.28);
        color: #ffd28b;
    }

    .result-card {
        padding: 23px;
        border-radius: 17px;
        border: 1px solid rgba(32,214,194,.28);
        background: linear-gradient(135deg, rgba(18,67,68,.62), rgba(12,34,50,.95));
        margin-top: 16px;
    }

    .result-card.elevated {
        border-color: rgba(255,123,139,.36);
        background: linear-gradient(135deg, rgba(91,36,55,.48), rgba(28,27,46,.96));
    }

    .result-title {
        color: #f4fbff;
        font: 800 1.55rem 'Manrope', sans-serif;
        margin: 8px 0;
    }

    .muted-small {
        color: #91aabd;
        font-size: .82rem;
        line-height: 1.6;
    }

    .footer {
        border-top: 1px solid var(--border);
        padding-top: 18px;
        margin-top: 38px;
        color: #718ba1;
        font-size: .76rem;
        line-height: 1.6;
    }

    @media (max-width: 760px) {
        .block-container { padding: 1.2rem 1rem 2rem 1rem; }
        .hero { padding: 23px 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- HELPERS --------------------------------

def section_heading(title):
    st.markdown(
        f'<div class="section-head"><span class="section-dot"></span>{title}</div>',
        unsafe_allow_html=True,
    )


def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#b9ccdc", size=12),
        margin=dict(l=10, r=10, t=25, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#b9ccdc"),
        ),
        hoverlabel=dict(
            bgcolor="#10283b",
            bordercolor="#20d6c2",
            font=dict(color="#f3fbff"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(145,169,189,.12)",
        zerolinecolor="rgba(145,169,189,.12)",
        linecolor="rgba(145,169,189,.12)",
    )
    fig.update_yaxes(
        gridcolor="rgba(145,169,189,.12)",
        zerolinecolor="rgba(145,169,189,.12)",
        linecolor="rgba(145,169,189,.12)",
    )
    return fig


def get_feature_importance():
    tree_model = model
    if hasattr(model, "steps"):
        tree_model = model.steps[-1][1]

    if not hasattr(tree_model, "feature_importances_"):
        return None

    values = tree_model.feature_importances_
    if len(values) != len(FEATURE_LABELS):
        return None

    result = pd.DataFrame(
        {"Parameter": FEATURE_LABELS, "Importance": values}
    ).sort_values("Importance", ascending=True)
    return result


def elevated_probability(input_data):
    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
    if 1 in classes:
        return float(probabilities[classes.index(1)])
    return 0.0


def render_hero(eyebrow, title, description, chip="CGWB groundwater quality dataset"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="hero-chip">✦ {chip}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------- SIDEBAR -------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <div class="brand-mark">💧</div>
            <div>
                <div class="brand-name">AquiferGuard</div>
                <div class="brand-caption">Groundwater intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="eyebrow" style="margin-top:14px;">Workspace</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Overview", "Risk Explorer", "Predict Risk", "Prediction History", "About"],
        label_visibility="collapsed",
        format_func=lambda item: {
            "Overview": "◫   Overview",
            "Risk Explorer": "⌕   Risk Explorer",
            "Predict Risk": "✧   Predict Risk",
            "Prediction History": "◷   Prediction History",
            "About": "ⓘ   About Project",
        }[item],
    )
    st.markdown("---")
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">Model status</div>
            <div style="color:#69e2bd;font-weight:700;">● Model loaded</div>
            <div class="muted-small" style="margin-top:8px;">
                Decision Tree classification<br>
                Five groundwater parameters
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "[View GitHub repository ↗](https://github.com/jschouhan007/AquiferGuard)"
    )
    st.markdown(
        "[Open live dashboard ↗](https://aquifera.streamlit.app/)"
    )
    st.markdown(
        '<div class="muted-small" style="margin-top:20px;">ACADEMIC PROJECT · 2026</div>',
        unsafe_allow_html=True,
    )


# -------------------------- DATA SUMMARY ----------------------------

total_samples = len(df)
elevated_samples = int(df["Uranium_Risk"].sum())
not_elevated_samples = total_samples - elevated_samples
risk_share = (elevated_samples / total_samples * 100) if total_samples else 0
district_count = (
    int(df["District"].nunique()) if "District" in df.columns else 0
)
average_uranium = float(df["Uranium(mg/L)"].mean()) if total_samples else 0


# ----------------------------- OVERVIEW -----------------------------

if page == "Overview":
    render_hero(
        "Environmental data • predictive analytics",
        "Groundwater intelligence, <span style='color:#48dfd0;'>made clearer.</span>",
        "Explore uranium measurements from Punjab groundwater samples, understand district-level patterns, and use a machine learning model to estimate the elevated-risk category.",
        "Interactive groundwater risk dashboard",
    )

    metrics = st.columns(4)
    metric_items = [
        ("Uranium records", f"{total_samples:,}", "Samples with recorded uranium values", "#42c9f5"),
        ("Elevated category", f"{elevated_samples:,}", f"{risk_share:.1f}% of recorded samples", "#ff9a9e"),
        ("Not elevated", f"{not_elevated_samples:,}", "At or below project threshold", "#59dfad"),
        ("Districts covered", f"{district_count:,}", "Districts in available records", "#b69cff"),
    ]
    for column, (label, value, note, accent) in zip(metrics, metric_items):
        with column:
            st.markdown(
                f"""
                <div class="panel" style="border-top:3px solid {accent};">
                    <div class="panel-label">{label}</div>
                    <div class="panel-value">{value}</div>
                    <div class="panel-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_heading("Risk distribution & geographic patterns")
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown('<div class="panel-label">Sample classification</div>', unsafe_allow_html=True)
        risk_df = pd.DataFrame({
            "Category": ["Not Elevated", "Elevated"],
            "Samples": [not_elevated_samples, elevated_samples],
        })
        fig = px.pie(
            risk_df,
            names="Category",
            values="Samples",
            hole=0.70,
            color="Category",
            color_discrete_map={
                "Not Elevated": "#20d6c2",
                "Elevated": "#ff8293",
            },
        )
        fig.update_traces(
            textinfo="percent",
            textfont=dict(size=13, color="#f4fbff"),
            marker=dict(line=dict(color="#0d1b2b", width=4)),
        )
        fig.add_annotation(
            text=f"<b>{total_samples:,}</b><br><span style='font-size:11px'>records</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color="#f4fbff"),
        )
        style_chart(fig, 330)
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="panel-label">Elevated samples by district</div>', unsafe_allow_html=True)
        if "District" in df.columns:
            district_data = (
                df[df["Uranium_Risk"] == 1]
                .groupby("District")
                .size()
                .reset_index(name="Elevated samples")
                .nlargest(10, "Elevated samples")
                .sort_values("Elevated samples", ascending=True)
            )
            if not district_data.empty:
                fig = px.bar(
                    district_data,
                    x="Elevated samples",
                    y="District",
                    orientation="h",
                    text="Elevated samples",
                    color="Elevated samples",
                    color_continuous_scale=["#164e63", "#20d6c2", "#a4f3e5"],
                )
                fig.update_traces(
                    textposition="outside",
                    marker_line_width=0,
                    hovertemplate="%{y}<br>Elevated samples: %{x}<extra></extra>",
                )
                fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Samples")
                style_chart(fig, 330)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("No elevated samples are available for the district chart.")
        else:
            st.info("The dataset does not contain a District column.")

    section_heading("Uranium concentration profile")
    concentration_col, info_col = st.columns([1.55, 1], gap="large")
    with concentration_col:
        histogram = px.histogram(
            df,
            x="Uranium(mg/L)",
            nbins=38,
            color_discrete_sequence=["#36cbbd"],
            labels={"Uranium(mg/L)": "Uranium concentration (mg/L)", "count": "Samples"},
        )
        histogram.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_width=2,
            line_color="#ff9e78",
            annotation_text="Project threshold · 0.03 mg/L",
            annotation_position="top right",
            annotation_font_color="#ffc19f",
        )
        histogram.update_layout(yaxis_title="Number of samples")
        style_chart(histogram, 310)
        st.plotly_chart(histogram, use_container_width=True, config={"displayModeBar": False})

    with info_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">Dataset snapshot</div>
                <div class="panel-value">{average_uranium:.4f} <span style="font-size:.9rem;color:#91aabd;">mg/L</span></div>
                <div class="panel-note">Average uranium concentration in records with a measurement</div>
                <hr style="border-color:rgba(145,169,189,.16);margin:18px 0;">
                <div class="panel-label">Project classification threshold</div>
                <div style="font:800 1.45rem 'Manrope';color:#ffbd91;">0.03 mg/L</div>
                <div class="muted-small">Values greater than this project-defined threshold are labelled Elevated.</div>
                <div style="margin-top:18px;"><span class="status-pill">Exploratory analysis</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    importance_df = get_feature_importance()
    if importance_df is not None:
        section_heading("What influences the model?")
        importance_plot = px.bar(
            importance_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Parameter",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#164e63", "#20d6c2", "#a4f3e5"],
        )
        importance_plot.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Relative feature importance",
            yaxis_title="",
        )
        style_chart(importance_plot, 285)
        st.plotly_chart(importance_plot, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Feature importance describes each input's relative contribution to the Decision Tree's impurity reduction. "
            "It does not show the direction of an effect, establish causation, or certify water safety."
        )


# --------------------------- RISK EXPLORER ---------------------------

elif page == "Risk Explorer":
    render_hero(
        "Explore the source records",
        "Understand the patterns in the data.",
        "Filter the available groundwater measurements by district and project risk category. Inspect the records used for the dashboard summaries.",
        "Dataset explorer",
    )

    filter_cols = st.columns([1, 1, 1.4])
    districts = ["All districts"] + sorted(df["District"].dropna().astype(str).unique().tolist()) if "District" in df.columns else ["All districts"]
    with filter_cols[0]:
        selected_district = st.selectbox("District", districts)
    with filter_cols[1]:
        selected_risk = st.selectbox("Risk category", ["All categories", "Elevated", "Not Elevated"])
    with filter_cols[2]:
        station_query = st.text_input("Search station", placeholder="Type a station name...")

    filtered = df.copy()
    if selected_district != "All districts" and "District" in filtered.columns:
        filtered = filtered[filtered["District"].astype(str) == selected_district]
    if selected_risk != "All categories":
        filtered = filtered[filtered["Risk Category"] == selected_risk]
    if station_query and "Station" in filtered.columns:
        filtered = filtered[
            filtered["Station"].astype(str).str.contains(station_query, case=False, na=False)
        ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Matching records", f"{len(filtered):,}")
    c2.metric("Elevated records", f"{int(filtered['Uranium_Risk'].sum()):,}")
    c3.metric(
        "Average uranium",
        f"{filtered['Uranium(mg/L)'].mean():.4f} mg/L" if len(filtered) else "—",
    )

    section_heading("Filtered groundwater records")
    preferred_columns = [
        "Station", "District", "Potential of Hydrogen (pH)",
        "Electric Conductivity (μS/cm)", "Bicarbonate (mg/L)",
        "Chloride (mg/L)", "Sodium (mg/L)", "Uranium(mg/L)", "Risk Category",
    ]
    display_columns = [column for column in preferred_columns if column in filtered.columns]
    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        height=470,
    )

    csv_records = filtered[display_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered records (CSV)",
        data=csv_records,
        file_name="aquiferguard_filtered_groundwater_records.csv",
        mime="text/csv",
    )


# ---------------------------- PREDICTION -----------------------------

elif page == "Predict Risk":
    render_hero(
        "Machine learning workspace",
        "Estimate a groundwater risk category.",
        "Enter the five water chemistry parameters below. AquiferGuard will pass them to the saved Decision Tree model and display its predicted class and estimated class probability.",
        "Decision Tree classifier • five input parameters",
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        section_heading("Groundwater parameters")
        st.markdown(
            '<div class="muted-small">Enter the laboratory-measured values for one groundwater sample.</div>',
            unsafe_allow_html=True,
        )
        with st.form("aquiferguard_prediction_form"):
            input_col1, input_col2 = st.columns(2)
            with input_col1:
                ph = st.number_input(
                    "pH", min_value=0.0, max_value=14.0,
                    value=8.0, step=0.1, help="Allowed pH input range: 0 to 14.",
                )
                ec = st.number_input(
                    "Electrical Conductivity (μS/cm)",
                    min_value=0.0, value=1000.0, step=50.0,
                )
                bicarbonate = st.number_input(
                    "Bicarbonate (mg/L)",
                    min_value=0.0, value=400.0, step=10.0,
                )
            with input_col2:
                chloride = st.number_input(
                    "Chloride (mg/L)",
                    min_value=0.0, value=100.0, step=10.0,
                )
                sodium = st.number_input(
                    "Sodium (mg/L)",
                    min_value=0.0, value=150.0, step=10.0,
                )
                st.markdown(
                    """
                    <div class="muted-small" style="padding-top:12px;">
                    <b style="color:#c9e5ef;">Model inputs</b><br>
                    Five chemistry values are used by the saved model. Missing-value handling remains inside the existing trained pipeline, if present.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            submitted = st.form_submit_button(
                "✦  Predict uranium risk",
                use_container_width=True,
            )

    with right:
        section_heading("How to read the result")
        st.markdown(
            """
            <div class="panel">
                <div class="panel-label">Prediction guide</div>
                <div style="display:flex;align-items:center;gap:10px;margin:13px 0;">
                    <span class="status-pill">Not Elevated</span>
                    <span class="muted-small">Model predicts class 0</span>
                </div>
                <div style="display:flex;align-items:center;gap:10px;margin:13px 0;">
                    <span class="status-pill warning">Elevated</span>
                    <span class="muted-small">Model predicts class 1</span>
                </div>
                <hr style="border-color:rgba(145,169,189,.16);margin:18px 0;">
                <div class="panel-label">Project threshold</div>
                <div class="panel-value">0.03 <span style="font-size:.9rem;color:#91aabd;">mg/L</span></div>
                <div class="muted-small">This is the project's chosen target-label threshold, not a declaration of regulatory compliance or water safety.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        input_data = pd.DataFrame(
            [[ph, ec, bicarbonate, chloride, sodium]],
            columns=FEATURES,
        )
        prediction = int(model.predict(input_data)[0])
        probability = elevated_probability(input_data)
        result = "Elevated" if prediction == 1 else "Not Elevated"

        record = {
            "pH": ph,
            "Electrical Conductivity": ec,
            "Bicarbonate": bicarbonate,
            "Chloride": chloride,
            "Sodium": sodium,
            "Prediction": result,
            "Elevated-class probability (%)": round(probability * 100, 2),
        }
        st.session_state.setdefault("history", []).append(record)
        st.session_state["latest_prediction"] = record

    latest = st.session_state.get("latest_prediction")
    if latest:
        elevated = latest["Prediction"] == "Elevated"
        card_class = "result-card elevated" if elevated else "result-card"
        pill_class = "status-pill warning" if elevated else "status-pill"
        result_icon = "⚠" if elevated else "✓"
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="eyebrow">Latest model output</div>
                <span class="{pill_class}">{latest['Prediction']}</span>
                <div class="result-title">{result_icon} {latest['Prediction']} uranium risk category</div>
                <div class="muted-small">The saved Decision Tree model predicted this class from the five entered groundwater parameters.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        probability_value = float(latest["Elevated-class probability (%)"])
        result_cols = st.columns([1, 1.4])
        with result_cols[0]:
            st.metric("Elevated-class probability", f"{probability_value:.2f}%")
        with result_cols[1]:
            probability_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability_value,
                    number={"suffix": "%", "font": {"color": "#f3fbff", "size": 30}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#91aabd"},
                        "bar": {"color": "#ff8293" if elevated else "#20d6c2"},
                        "bgcolor": "#102338",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 100], "color": "rgba(145,169,189,.08)"}
                        ],
                    },
                )
            )
            style_chart(probability_fig, 220)
            probability_fig.update_layout(margin=dict(l=15, r=15, t=18, b=0))
            st.plotly_chart(probability_fig, use_container_width=True, config={"displayModeBar": False})

        st.caption(
            "This probability is the model's estimated probability for the Elevated class. "
            "It is not a measured uranium concentration. Actual water quality requires appropriate laboratory testing."
        )


# ------------------------- PREDICTION HISTORY ------------------------

elif page == "Prediction History":
    render_hero(
        "Your current session",
        "Review your prediction activity.",
        "Every prediction made in this browser session is listed here. Export the results for your project documentation or clear the session history when needed.",
        "Session-based prediction log",
    )

    history = st.session_state.get("history", [])
    if history:
        history_df = pd.DataFrame(history)
        elevated_count = int((history_df["Prediction"] == "Elevated").sum())
        a, b, c = st.columns(3)
        a.metric("Predictions made", f"{len(history_df):,}")
        b.metric("Elevated predictions", f"{elevated_count:,}")
        c.metric("Not elevated predictions", f"{len(history_df) - elevated_count:,}")

        section_heading("Prediction log")
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        download_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction history (CSV)",
            data=download_data,
            file_name="aquiferguard_prediction_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history", type="secondary"):
            st.session_state["history"] = []
            st.session_state.pop("latest_prediction", None)
            st.rerun()
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:42px 20px;">
                <div style="font-size:2.2rem;">◷</div>
                <div class="result-title">No predictions yet</div>
                <div class="muted-small">Visit Predict Risk to evaluate a groundwater sample. Your results will appear here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------- ABOUT -------------------------------

elif page == "About":
    render_hero(
        "Project information",
        "A data-driven view of groundwater quality.",
        "AquiferGuard is an academic predictive analytics project that explores groundwater uranium measurements and classifies samples into two project-defined categories.",
        "Academic machine learning project",
    )

    section_heading("Project overview")
    st.markdown(
        """
        <div class="panel">
            <p style="color:#d9eaf5;line-height:1.8;">
            AquiferGuard uses groundwater chemistry parameters to estimate whether a sample belongs to the
            <b style="color:#ff9aa8;">Elevated</b> or
            <b style="color:#69e2bd;">Not Elevated</b> uranium category.
            The dashboard combines dataset exploration, descriptive statistics, model feature importance,
            and interactive sample prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        section_heading("Model and input parameters")
        st.markdown(
            """
            <div class="panel">
                <div class="panel-label">Classification algorithm</div>
                <div class="panel-value">Decision Tree</div>
                <hr style="border-color:rgba(145,169,189,.16);margin:16px 0;">
                <div class="panel-label">Model input features</div>
                <ul style="color:#bfd0df;line-height:2;">
                    <li>Potential of Hydrogen (pH)</li>
                    <li>Electrical Conductivity (μS/cm)</li>
                    <li>Bicarbonate (mg/L)</li>
                    <li>Chloride (mg/L)</li>
                    <li>Sodium (mg/L)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        section_heading("Data and classification")
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">Data source</div>
                <div style="color:#d9eaf5;font-weight:700;margin-bottom:8px;">Central Ground Water Board (CGWB)</div>
                <div class="muted-small">Ground-water quality manual chemical parameters dataset, accessed through India's National Water Data Portal.</div>
                <hr style="border-color:rgba(145,169,189,.16);margin:16px 0;">
                <div class="panel-label">Project-defined target threshold</div>
                <div class="panel-value" style="color:#ffbd91;">&gt; 0.03 mg/L</div>
                <div class="muted-small">Above this value: Elevated. At or below this value: Not Elevated.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Important limitations")
    st.warning(
        "The 0.03 mg/L threshold is the project's chosen classification threshold. "
        "It must not be interpreted as a regulatory safe/unsafe boundary. "
        "Predictions are model estimates, not laboratory measurements, medical advice, "
        "or official water-quality certification."
    )
    st.markdown(
        "[Open the National Water Data Portal dataset ↗](https://nwdp.nwic.gov.in/dataset/ground-water-quality-manual-chemical-parameters-cgwb-f-gfg)"
    )


# ------------------------------- FOOTER ------------------------------

st.markdown(
    """
    <div class="footer">
        <b style="color:#a9c1d2;">AQUIFERGUARD</b> · Groundwater Uranium Risk Classification<br>
        Academic predictive analytics project. Model outputs are estimates only and do not replace laboratory testing or official water-quality assessment.
    </div>
    """,
    unsafe_allow_html=True,
)
