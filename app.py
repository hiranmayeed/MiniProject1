"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 4: Streamlit Dashboard — The Farmer's Interface
=============================================================================
Purpose : Interactive dashboard for farmers and agricultural officers.
          Loads pre-processed CSVs and the trained ML model to display:
          - District rainfall trends vs LPA
          - Probability of Above-Normal monsoon (empirical + ML)
          - Plain-English farmer advice
Run with: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — PAGE CONFIGURATION
# Must be the FIRST streamlit command in the script.
# =============================================================================

st.set_page_config(
    page_title  = "Rainfall Analysis Framework",
    page_icon   = "🌧️",
    layout      = "wide",          # Use full browser width
    initial_sidebar_state = "expanded"
)


# =============================================================================
# SECTION 2 — CUSTOM CSS STYLING
# Streamlit allows injecting raw CSS via st.markdown with unsafe_allow_html.
# This gives us control over fonts, colors, and card styling.
# =============================================================================

st.markdown("""
<style>
    /* Import a clean, readable Google Font */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

    /* Apply font globally */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Main title styling */
    .main-title {
        font-size: 5rem;
        font-weight: 800;
        color: #2c7be5;                
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        font-size: 1.25rem;
        color: #999;
        margin-top: 4px;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        margin-bottom: 16px;
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
        margin-bottom: 8px;
    }

    .metric-value-green  { font-size: 3rem; font-weight: 600; color: #2d7a4f; line-height: 1; }
    .metric-value-blue   { font-size: 3rem; font-weight: 600; color: #1a5fa8; line-height: 1; }
    .metric-value-orange { font-size: 3rem; font-weight: 600; color: #c07a00; line-height: 1; }
    .metric-value-red    { font-size: 3rem; font-weight: 600; color: #b03030; line-height: 1; }
    .metric-value-gray   { font-size: 3rem; font-weight: 600; color: #555;    line-height: 1; }

    .metric-sublabel {
        font-size: 0.85rem;
        color: #999;
        margin-top: 6px;
    }

    /* Advice card */
    .advice-card-green  { background:#f0faf4; border-left:4px solid #2d7a4f; border-radius:8px; padding:20px; }
    .advice-card-blue   { background:#f0f6ff; border-left:4px solid #1a5fa8; border-radius:8px; padding:20px; }
    .advice-card-orange { background:#fff8e6; border-left:4px solid #c07a00; border-radius:8px; padding:20px; }
    .advice-card-red    { background:#fff0f0; border-left:4px solid #b03030; border-radius:8px; padding:20px; }

    .advice-title { font-size:1rem; font-weight:600; margin-bottom:8px; }
    .advice-body  { font-size:0.92rem; line-height:1.7; color:#333; }

    /* Section headers */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #aaa;
        border-bottom: 1px solid #eee;
        padding-bottom: 8px;
        margin-bottom: 16px;
        margin-top: 8px;
    }

    /* Info banner */
    .info-banner {
        background: #f8f9ff;
        border: 1px solid #dde3ff;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #445;
        margin-bottom: 16px;
    }

    /* Warning banner */
    .warn-banner {
        background: #fffbf0;
        border: 1px solid #ffe082;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #664d00;
        margin-bottom: 16px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3 — DATA & MODEL LOADERS
# @st.cache_data tells Streamlit: "load this once, then reuse from memory".
# Without caching, every slider move or dropdown change would reload the CSV.
# =============================================================================

@st.cache_data
def load_processed_data():
    """
    Loads all four processed CSVs directly from Google Drive.
    To update: replace the file ID strings below with your own.
    Get the file ID from the sharing link:
      https://drive.google.com/drive/folders/19YpQ1g7SOaZup3iLS75Gme315cnpfg7Q?usp=drive_link
    """

    # ── Paste your Google Drive file IDs here ──────────────────────────
    FILE_IDS = {
        "monthly"     : "1AY2n7HBfu0BsrLlDL80iflWqlqLSYiMH",
        "seasonal"    : "1rnbhP44S_gah-v7L6BRJKBZInwLSLjDG",
        "probability" : "1wHgLiXOuvqLmpzaPSoj73rWpTPmHYgp2",
    }
    # ───────────────────────────────────────────────────────────────────

    def drive_url(file_id: str) -> str:
        # Converts a Drive file ID into a direct download URL pandas can read
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    data = {}
    for key, file_id in FILE_IDS.items():
        try:
            data[key] = pd.read_csv(drive_url(file_id))
        except Exception as e:
            st.error(
                f"Could not load '{key}' from Google Drive.\n\n"
                f"Check that the file ID is correct and the file is shared "
                f"publicly (Anyone with the link → Viewer).\n\nError: {e}"
            )
            st.stop()

    return data


@st.cache_resource          # cache_resource for non-serialisable objects like ML models
def load_model():
    """
    Loads the trained Random Forest model and its metadata from disk.
    Returns (model, metadata) or (None, None) if files not found.
    """
    model_path    = "models/rainfall_model.pkl"
    metadata_path = "models/model_metadata.pkl"

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None

    with open(model_path,    "rb") as f: model    = pickle.load(f)
    with open(metadata_path, "rb") as f: metadata = pickle.load(f)

    return model, metadata


# =============================================================================
# SECTION 4 — HELPER FUNCTIONS
# =============================================================================

def get_colour_for_probability(probability: float) -> str:
    """Maps a probability % to a colour name used in CSS class names."""
    if probability >= 65:   return "green"
    elif probability >= 50: return "blue"
    elif probability >= 35: return "orange"
    else:                   return "red"


def get_advice(probability: float, district: str) -> dict:
    """
    Returns a dict with title + bullet-point advice based on probability.
    Written in plain English for farmers, not data scientists.
    """
    if probability >= 65:
        return {
            "title"  : "✅ Good monsoon likely — plan for full sowing",
            "points" : [
                f"Strong signal of above-normal rainfall for {district} this Kharif season.",
                "Proceed with your full sowing plan for water-intensive crops (paddy, sugarcane).",
                "Ensure drainage channels and bunds are cleared before June to handle surplus water.",
                "Stock up on seeds and fertilisers early — demand will be high.",
                "Consider crop insurance nonetheless; even good monsoons can have dry spells.",
            ]
        }
    elif probability >= 50:
        return {
            "title"  : "🔵 Moderate outlook — proceed with caution",
            "points" : [
                f"Moderate confidence of above-normal rainfall for {district}.",
                "Sow primary crops as planned but reserve 20% of your area as a buffer.",
                "Diversify: mix a high-value water-intensive crop with one drought-tolerant variety.",
                "Monitor weekly district rainfall updates — adjust irrigation schedules accordingly.",
                "Consult your local Krishi Vigyan Kendra (KVK) for variety recommendations.",
            ]
        }
    elif probability >= 35:
        return {
            "title"  : "🟡 Near-normal expected — standard precautions advised",
            "points" : [
                f"Season likely to be near the historical average for {district}.",
                "Follow your standard sowing schedule for the district.",
                "Keep drought-tolerant variety seeds (e.g. millets, sorghum) as backup.",
                "Check soil moisture before each irrigation cycle to avoid over-watering.",
                "Review your crop insurance policy before June 30th.",
            ]
        }
    else:
        return {
            "title"  : "🔴 Below-normal risk — take protective action",
            "points" : [
                f"Higher than usual risk of below-normal monsoon for {district}.",
                "Strongly consider drought-resistant crop varieties this season.",
                "Prioritise water conservation: repair farm ponds and check-dams now.",
                "Avoid over-investment in water-intensive crops like paddy this cycle.",
                "Contact your district agriculture officer and register for drought relief schemes.",
                "Visit the nearest Krishi Vigyan Kendra before finalising sowing plans.",
            ]
        }


def make_ml_prediction(model, metadata, district: str, state: str,
                        june_mm: float, seasonal_df: pd.DataFrame,
                        monthly_df: pd.DataFrame) -> float:
    """
    Runs the trained model for the selected district and June rainfall.
    v4: Builds a 20-feature vector matching get_feature_columns() in
    train_model.py. Temporal + SPI + ENSO + GEE features are estimated
    from the june_mm input and historical data in the loaded DataFrames.
    Returns the probability (0–100) of Above-Normal season, or None.
    """

    le_district = metadata["label_encoder_district"]
    le_state    = metadata["label_encoder_state"]

    if district not in metadata["known_districts"]:
        return None

    district_enc = le_district.transform([district])[0]
    state_enc    = le_state.transform([state])[0] \
                   if state in metadata["known_states"] else 0

    # ── District historical context ────────────────────────────────────────
    kharif_rows  = seasonal_df[
        (seasonal_df["district_name"] == district) &
        (seasonal_df["season"] == "Kharif")
    ]["total_rainfall_mm"]

    district_mean = kharif_rows.mean() if len(kharif_rows) > 0 else 500
    district_std  = kharif_rows.std()  if len(kharif_rows) > 0 else 100
    district_cv   = district_std / district_mean if district_mean > 0 else 0.2

    # ── June LPA and departure ─────────────────────────────────────────────
    june_rows = monthly_df[
        (monthly_df["district_name"] == district) &
        (monthly_df["month"] == 6)
    ]["lpa_mm"]
    june_lpa = june_rows.mean() if len(june_rows) > 0 else district_mean / 4

    june_departure_pct    = ((june_mm - june_lpa) / june_lpa * 100) \
                             if june_lpa > 0 else 0
    june_vs_district_mean = june_mm / district_mean if district_mean > 0 else 1.0
    june_was_above_normal = int(june_departure_pct > 5)

    # ── Temporal feature estimates (from june_mm total) ───────────────────
    daily_avg            = june_mm / 30.0
    june_rolling_7d_avg  = daily_avg * 7
    june_cumulative_rain = june_mm
    june_rain_lag_1d     = daily_avg
    june_rain_lag_7d     = daily_avg
    rainy_fraction       = min(june_mm / (june_lpa * 1.2), 1.0) \
                           if june_lpa > 0 else 0.5
    june_dry_streak      = round((1 - rainy_fraction) * 20)

    # ── SPI estimate ───────────────────────────────────────────────────────
    # Z-score of june_mm relative to the district's historical June mean/std.
    # Uses the monthly DataFrame's departure_pct as a proxy if spi not stored.
    june_spi_30d = june_departure_pct / 33.3   # rough conversion: ±100% ≈ ±3σ

    # ── ENSO code for current year ─────────────────────────────────────────
    # Look up from seasonal data if available, else default to Neutral (1)
    import datetime
    current_year = datetime.datetime.now().year
    enso_lookup  = {
        2017: 1, 2018: 1, 2019: 2, 2020: 0,
        2021: 0, 2022: 0, 2023: 2, 2024: 1, 2025: 0,
    }
    enso_code = enso_lookup.get(current_year, 1)

    # ── GEE features — use historical district median if available ─────────
    # Pull from seasonal_df if enrich_with_gee_features() has been run.
    def _dist_median(col: str) -> float:
        if col in seasonal_df.columns:
            vals = seasonal_df[
                seasonal_df["district_name"] == district
            ][col].dropna()
            return float(vals.median()) if len(vals) > 0 else float("nan")
        return float("nan")

    susm_may_mean         = _dist_median("susm_may_mean")
    susm_may_max          = _dist_median("susm_may_max")
    temp_june_mean        = _dist_median("temp_june_mean")
    temp_june_stress_days = _dist_median("temp_june_stress_days")

    # If GEE data not yet available, use sensible defaults for Telangana
    import math
    if math.isnan(susm_may_mean):         susm_may_mean         = 45.0
    if math.isnan(susm_may_max):          susm_may_max          = 65.0
    if math.isnan(temp_june_mean):        temp_june_mean        = 32.0
    if math.isnan(temp_june_stress_days): temp_june_stress_days = 8.0

    # ── Feature vector — must match get_feature_columns() in train_model.py ─
    # Order: Group A (4) | Group B (5) | Group C (2) | Group D (4) |
    #        Group E (3) | Group F (2)  = 20 total
    feature_vector = [[
        # Group A — Core June signal
        june_mm,
        june_departure_pct,
        june_vs_district_mean,
        june_was_above_normal,
        # Group B — Temporal pattern
        june_rolling_7d_avg,
        june_cumulative_rain,
        june_rain_lag_1d,
        june_rain_lag_7d,
        june_dry_streak,
        # Group C — Climate context
        june_spi_30d,
        enso_code,
        # Group D — GEE soil moisture & temperature
        susm_may_mean,
        susm_may_max,
        temp_june_mean,
        temp_june_stress_days,
        # Group E — District historical context
        district_mean,
        district_std,
        district_cv,
        # Group F — Identity encoding
        district_enc,
        state_enc,
    ]]

    prob = model.predict_proba(feature_vector)[0][1]
    return round(prob * 100, 1)


# =============================================================================
# SECTION 5 — CHART FUNCTIONS
# =============================================================================

def plot_rainfall_trend(seasonal_df: pd.DataFrame, district: str) -> plt.Figure:
    """
    Draws a bar chart of Kharif season total rainfall per year for the
    selected district, with the LPA shown as a horizontal reference line.
    Bars are coloured green (above LPA) or coral (below LPA).
    """

    # Filter to this district's Kharif rows only
    dist_data = seasonal_df[
        (seasonal_df["district_name"] == district) &
        (seasonal_df["season"] == "Kharif")
    ].sort_values("year").copy()

    if dist_data.empty:
        return None

    lpa = dist_data["lpa_mm"].iloc[0]     # LPA is the same for all rows (it's a constant per district)
    years    = dist_data["year"].astype(str).tolist()
    rainfall = dist_data["total_rainfall_mm"].tolist()

    # Colour each bar based on whether it exceeded the LPA
    bar_colours = ["#2d7a4f" if r >= lpa else "#c0392b" for r in rainfall]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # Draw bars
    bars = ax.bar(years, rainfall, color=bar_colours, width=0.55,
                  zorder=3, edgecolor="white", linewidth=0.8)

    # LPA reference line
    ax.axhline(y=lpa, color="#1a1a2e", linewidth=1.5,
               linestyle="--", zorder=4, label=f"LPA: {lpa:.0f} mm")

    # Annotate each bar with its value
    for bar, val in zip(bars, rainfall):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{val:.0f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#333",
            fontfamily="monospace"
        )

    # Labels and formatting
    ax.set_xlabel("Year", fontsize=10, color="#555", labelpad=8)
    ax.set_ylabel("Total Rainfall (mm)", fontsize=10, color="#555", labelpad=8)
    ax.set_title(
        f"Kharif Season Rainfall — {district}",
        fontsize=13, fontweight="600", color="#1a1a2e", pad=14
    )
    ax.tick_params(colors="#666", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ddd")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#ddd", zorder=0)
    ax.set_axisbelow(True)

    # Legend
    above_patch = mpatches.Patch(color="#2d7a4f", label="Above LPA")
    below_patch = mpatches.Patch(color="#c0392b", label="Below LPA")
    lpa_line    = plt.Line2D([0], [0], color="#1a1a2e", linewidth=1.5,
                              linestyle="--", label=f"LPA ({lpa:.0f} mm)")
    ax.legend(handles=[above_patch, below_patch, lpa_line],
              fontsize=8.5, framealpha=0.9, loc="upper right")

    plt.tight_layout()
    return fig


def plot_departure_heatmap(monthly_df: pd.DataFrame, district: str) -> plt.Figure:
    """
    Draws a month × year heatmap of departure % for the selected district.
    Warm colours = excess, cool colours = deficit.
    This lets the farmer spot which months are consistently wet or dry.
    """

    dist_monthly = monthly_df[
        monthly_df["district_name"] == district
    ].copy()

    if dist_monthly.empty:
        return None

    # Pivot into a month × year matrix
    pivot = dist_monthly.pivot_table(
        index="month", columns="year",
        values="departure_pct", aggfunc="mean"
    )

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.index = [month_labels[m-1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    # RdYlGn: red = deficit, yellow = normal, green = excess
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-100, vmax=100)

    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str), fontsize=9, color="#444")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color="#444")

    # Annotate each cell with the departure value
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_colour = "white" if abs(val) > 55 else "#333"
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, color=text_colour,
                        fontfamily="monospace")

    plt.colorbar(im, ax=ax, label="Departure from LPA (%)",
                 fraction=0.03, pad=0.04)

    ax.set_title(
        f"Monthly Departure from LPA — {district}",
        fontsize=13, fontweight="600", color="#1a1a2e", pad=14
    )
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 6 — SIDEBAR
# =============================================================================

def render_sidebar(data: dict) -> tuple:
    """
    Renders the sidebar with State, District, and June Rainfall inputs.
    Returns (selected_state, selected_district, june_rainfall_input).
    """

    st.sidebar.markdown("## 🌧️ Rainfall Framework")
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### 📍 Select Location")

    seasonal_df = data["seasonal"]

    # State selector — populated from actual data
    states = sorted(seasonal_df["state_name"].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("State", states)

    # District selector — filtered by selected state
    districts = sorted(
        seasonal_df[seasonal_df["state_name"] == selected_state]
        ["district_name"].dropna().unique().tolist()
    )
    selected_district = st.sidebar.selectbox("District", districts)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌦️ June Rainfall Input")
    st.sidebar.markdown(
        "<small>Enter the total June rainfall recorded so far "
        "to generate an ML prediction for the rest of the season.</small>",
        unsafe_allow_html=True
    )

    # Get the June LPA for this district to use as the default slider value
    monthly_df = data["monthly"]
    june_lpa_rows = monthly_df[
        (monthly_df["district_name"] == selected_district) &
        (monthly_df["month"] == 6)
    ]["lpa_mm"]
    june_lpa_default = float(june_lpa_rows.mean()) if len(june_lpa_rows) > 0 else 100.0

    june_rainfall_input = st.sidebar.number_input(
        label       = "June Rainfall (mm)",
        min_value   = 0.0,
        max_value   = 1000.0,
        value       = round(june_lpa_default, 1),
        step        = 5.0,
        help        = f"Historical June LPA for {selected_district}: {june_lpa_default:.1f} mm"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#aaa'>Data: IMD Grid Model | "
        "Years: 2018–2024 | Districts: Telangana</small>",
        unsafe_allow_html=True
    )

    return selected_state, selected_district, june_rainfall_input


# =============================================================================
# SECTION 7 — MAIN DASHBOARD LAYOUT
# =============================================================================

def main():

    # ── Load data and model ────────────────────────────────────────────────
    data             = load_processed_data()
    model, metadata  = load_model()
    seasonal_df      = data["seasonal"]
    monthly_df       = data["monthly"]
    probability_df   = data["probability"]

    # ── Sidebar ────────────────────────────────────────────────────────────
    selected_state, selected_district, june_rainfall_input = render_sidebar(data)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="main-title">🌧️ Rainfall Analysis Dashboard</div>'
        f'<div class="main-subtitle">District-wise monsoon intelligence for Kharif crop planning</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Compute probabilities for selected district ────────────────────────

    # 1. Empirical probability (from pipeline's 04_ file — more reliable)
    emp_row = probability_df[
        (probability_df["district_name"] == selected_district) &
        (probability_df["season"] == "Kharif")
    ]
    empirical_prob = float(emp_row["prob_above_normal_pct"].iloc[0]) \
                     if not emp_row.empty else None

    # 2. ML model probability (based on June input)
    ml_prob = None
    if model is not None and metadata is not None:
        ml_prob = make_ml_prediction(
            model, metadata,
            selected_district, selected_state,
            june_rainfall_input,
            seasonal_df, monthly_df
        )

    # Use empirical as the primary display probability
    # (more trustworthy with our small dataset, as explained after Step 3)
    primary_prob   = empirical_prob if empirical_prob is not None else 50.0
    primary_colour = get_colour_for_probability(primary_prob)

    # ── ROW 1: Three metric cards ──────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        colour = get_colour_for_probability(primary_prob)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Empirical Probability</div>
            <div class="metric-value-{colour}">{primary_prob:.0f}%</div>
            <div class="metric-sublabel">Above-Normal Kharif Season<br>
            (based on {emp_row['years_above_normal'].iloc[0] if not emp_row.empty else '—'} of
            {emp_row['total_years'].iloc[0] if not emp_row.empty else '—'} historical years)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if ml_prob is not None:
            ml_colour = get_colour_for_probability(ml_prob)
            ml_display = f"{ml_prob:.0f}%"
            ml_note    = f"ML model | June input: {june_rainfall_input:.0f} mm"
            ml_acc     = f"Model accuracy: {metadata['cv_mean_accuracy']*100:.0f}% (LOOCV)"
        else:
            ml_colour  = "gray"
            ml_display = "N/A"
            ml_note    = "Model not loaded"
            ml_acc     = "Run train_model.py first"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Model Prediction</div>
            <div class="metric-value-{ml_colour}">{ml_display}</div>
            <div class="metric-sublabel">{ml_note}<br>{ml_acc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Kharif season total for last available full year
        last_year_row = seasonal_df[
            (seasonal_df["district_name"] == selected_district) &
            (seasonal_df["season"] == "Kharif")
        ].sort_values("year", ascending=False).head(1)

        if not last_year_row.empty:
            last_year      = int(last_year_row["year"].iloc[0])
            last_total     = last_year_row["total_rainfall_mm"].iloc[0]
            last_departure = last_year_row["departure_pct"].iloc[0]
            last_lpa       = last_year_row["lpa_mm"].iloc[0]
            dep_colour     = "green" if last_departure > 5 else \
                             ("red" if last_departure < -5 else "orange")
            dep_sign       = "+" if last_departure > 0 else ""
        else:
            last_year = last_total = last_departure = last_lpa = "—"
            dep_colour = "gray"
            dep_sign   = ""

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Last Full Season ({last_year})</div>
            <div class="metric-value-{dep_colour}">{dep_sign}{last_departure:.0f}%</div>
            <div class="metric-sublabel">Departure from LPA<br>
            {last_total:.0f} mm actual vs {last_lpa:.0f} mm LPA</div>
        </div>
        """, unsafe_allow_html=True)

    # ── ROW 2: Farmer's Advice ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Farmer\'s Advice</div>',
                unsafe_allow_html=True)

    advice = get_advice(primary_prob, selected_district)
    advice_html = "".join([f"<li style='margin-bottom:6px'>{pt}</li>"
                           for pt in advice["points"]])

    st.markdown(f"""
    <div class="advice-card-{primary_colour}">
        <div class="advice-title">{advice['title']}</div>
        <div class="advice-body">
            <ul style="margin:0; padding-left:20px">{advice_html}</ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Small disclaimer about ML accuracy
    if ml_prob is not None and metadata is not None:
        acc = metadata['cv_mean_accuracy'] * 100
        if acc < 65:
            st.markdown(f"""
            <div class="warn-banner">
            ⚠️ <strong>Note:</strong> The ML model has {acc:.0f}% cross-validation accuracy
            on this dataset. With only 6–7 years of data per district, the <strong>empirical
            probability</strong> (left card) is more reliable for decision-making.
            The ML prediction improves as more years of data are added.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 3: Charts ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Rainfall Charts</div>',
                unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        fig_trend = plot_rainfall_trend(seasonal_df, selected_district)
        if fig_trend:
            st.pyplot(fig_trend, use_container_width=True)
        else:
            st.info("No Kharif data available for this district.")

    with chart_col2:
        fig_heat = plot_departure_heatmap(monthly_df, selected_district)
        if fig_heat:
            st.pyplot(fig_heat, use_container_width=True)
        else:
            st.info("No monthly data available for this district.")
    
    # ── ROW 4: Raw Data Table (collapsible) ───────────────────────────────
    with st.expander("📊 View raw seasonal data for this district"):

        dist_seasonal = seasonal_df[
            seasonal_df["district_name"] == selected_district
        ][["year","season","total_rainfall_mm","lpa_mm",
           "departure_pct","anomaly_category"]].sort_values(
            ["year","season"]
        ).reset_index(drop=True)

        # Colour-code the anomaly_category column
        def colour_anomaly(val):
            colours = {
                "Large Excess"  : "background-color: #c8f7c5",
                "Above Normal"  : "background-color: #d4edda",
                "Normal"        : "background-color: #fff9c4",
                "Below Normal"  : "background-color: #fde8d8",
                "Large Deficit" : "background-color: #f9c0c0",
            }
            return colours.get(val, "")

        styled = dist_seasonal.style\
            .applymap(colour_anomaly, subset=["anomaly_category"])\
            .format({
                "total_rainfall_mm": "{:.1f}",
                "lpa_mm"           : "{:.1f}",
                "departure_pct"    : "{:+.1f}%",
            })

        st.dataframe(styled, use_container_width=True, height=300)

    # ── ROW 5: District comparison (all districts, current season) ────────
    with st.expander("🗺️ Compare all districts — Kharif above-normal probability"):

        kharif_probs = probability_df[
            probability_df["season"] == "Kharif"
        ].sort_values("prob_above_normal_pct", ascending=False).reset_index(drop=True)

        # Simple horizontal bar chart
        fig_comp, ax_comp = plt.subplots(figsize=(6, max(3, len(kharif_probs) * 0.22)))
        fig_comp.patch.set_facecolor("#fafafa")
        ax_comp.set_facecolor("#fafafa")

        colours_comp = [
            "#2d7a4f" if p >= 65 else
            "#1a5fa8" if p >= 50 else
            "#c07a00" if p >= 35 else "#b03030"
            for p in kharif_probs["prob_above_normal_pct"]
        ]

        # Highlight selected district
        # Matplotlib barh doesn't accept a list for alpha — draw in two passes
        for i, (district, val, colour) in enumerate(zip(
            kharif_probs["district_name"],
            kharif_probs["prob_above_normal_pct"],
            colours_comp
        )):
            alpha = 1.0 if district == selected_district else 0.55
            ax_comp.barh(district, val, color=colour, alpha=alpha,
                         height=0.65, edgecolor="white", linewidth=0.5)

        ax_comp.axvline(x=50, color="#aaa", linewidth=1,
                        linestyle="--", label="50% threshold")
        ax_comp.set_xlabel("Probability of Above-Normal Kharif (%)",
                           fontsize=9, color="#555")
        ax_comp.set_title("District Comparison — Above-Normal Probability",
                          fontsize=11, fontweight="600", color="#1a1a2e", pad=10)
        ax_comp.tick_params(labelsize=8, colors="#555")
        ax_comp.spines[["top","right"]].set_visible(False)
        ax_comp.spines[["left","bottom"]].set_color("#ddd")
        ax_comp.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_comp.set_xlim(0, 105)
        ax_comp.invert_yaxis()   # Highest probability at the top

        # Label bars
        for i, val in enumerate(kharif_probs["prob_above_normal_pct"]):
            ax_comp.text(val + 1, i,
                         f"{val:.0f}%", va="center", fontsize=7.5,
                         color="#333", fontfamily="monospace")

        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()