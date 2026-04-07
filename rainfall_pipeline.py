"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 2: Data Processing Pipeline — v4 (Climate Context Enrichment)
=============================================================================
New additions over v3:
  - calculate_spi()           : 30-day Standardized Precipitation Index
  - add_enso_context()        : ENSO state categorical feature (year lookup)
  - enrich_with_gee_features(): merges GEE soil moisture + temperature
  - All new features flow into seasonal aggregates for train_model.py

Run with: python rainfall_pipeline.py
=============================================================================
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

RAW_FILE_IDS = {
    "2018": "1GJtKaG1Ht82cDrYUSyLi63lUdx_fONrT",
    "2019": "1OS_JAicP0iE-ZiMWye8m_ynfJ5Ypf2eO",
    "2020": "1nB6qe_6SqVPDx5yyCVGJcX-ydeqtwlrE",
    "2021": "1QwtMNFi-TxS3sn2SM9BteuDGSQS3L5mW",
    "2022": "1179FbAiLT1KZJAvQiBcE8T2NQPHAC7zE",
    "2023": "1OgHoFuSwd_JUdadvuxPV1Pj0QFBE4sjf",
    "2024": "1q_yHt0UeqOzo1Kvzz8MTjaP-KEKBU3hr",
}

KHARIF_MONTHS = {6, 7, 8, 9}
RABI_MONTHS   = {10, 11, 12, 1, 2}

COLUMN_MAP = {
    "Date"        : "date",
    "State"       : "state_name",
    "District"    : "district_name",
    "Avg_rainfall": "rainfall_mm",
}

DISTRICT_NAME_CORRECTIONS = {
    "Jagtial"            : "Jagitial",
    "Jangoan"            : "Jangaon",
    "Kumuram Bheem"      : "Kumuram Bheem Asifabad",
    "Medchal-Malkajgiri" : "Medchal Malkajgiri",
    "Rangareddy"         : "Ranga Reddy",
    "Ranjanna Sircilla"  : "Rajanna Sircilla",
    "Warangal Rural"     : "Warangal (Rural)",
    "Warangal Urban"     : "Warangal (Urban)",
}

# ENSO State lookup by year.
# Source: NOAA Climate Prediction Center historical records.
# Encoding: 0 = La Niña, 1 = Neutral, 2 = El Niño
# WHY ENSO MATTERS: El Niño suppresses Indian monsoon rainfall;
# La Niña enhances it. This is the single strongest large-scale
# climate driver of Indian monsoon interannual variability.
ENSO_LOOKUP = {
    2017: {"enso_state": "Neutral",               "enso_code": 1},
    2018: {"enso_state": "Neutral",               "enso_code": 1},
    2019: {"enso_state": "Weak El Niño",          "enso_code": 2},
    2020: {"enso_state": "La Niña",               "enso_code": 0},
    2021: {"enso_state": "La Niña",               "enso_code": 0},
    2022: {"enso_state": "La Niña",               "enso_code": 0},
    2023: {"enso_state": "Strong El Niño",        "enso_code": 2},
    2024: {"enso_state": "Neutral/La Niña trans", "enso_code": 1},
    2025: {"enso_state": "La Niña",               "enso_code": 0},
}


def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# =============================================================================
# SECTION 2 — LOAD RAW DATA
# =============================================================================

def load_all_csvs(file_ids_dict: dict) -> pd.DataFrame:
    """Downloads and merges all yearly IMD CSVs from Google Drive."""

    yearly_frames = []
    print(f"Downloading {len(file_ids_dict)} files from Drive...\n")

    for year, file_id in file_ids_dict.items():
        try:
            df_year = pd.read_csv(drive_url(file_id))
            df_year["source_file"] = f"{year}.csv"
            yearly_frames.append(df_year)
            print(f"  ✅ {year}: {len(df_year):,} rows")
        except Exception as e:
            print(f"  ❌ {year}: {e}")

    if not yearly_frames:
        raise ValueError("No data loaded. Check RAW_FILE_IDS and internet.")

    combined = pd.concat(yearly_frames, ignore_index=True)
    print(f"\nTotal rows: {len(combined):,}\n")
    return combined


# =============================================================================
# SECTION 3 — CLEAN & STANDARDISE
# =============================================================================

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns, parses dates, strips whitespace, corrects districts,
    and fills missing rainfall via linear interpolation."""

    df = df.rename(columns=COLUMN_MAP)

    assert "rainfall_mm" in df.columns, (
        f"Rename failed. Expected 'Avg_rainfall'. Got: {df.columns.tolist()}"
    )

    df["date"]        = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["season"]      = df["month"].apply(assign_season)

    df["state_name"]    = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].replace(DISTRICT_NAME_CORRECTIONS)

    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    missing = df["rainfall_mm"].isna().sum()
    print(f"Missing rainfall: {missing} ({missing/len(df)*100:.2f}%)")

    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.interpolate(method="linear"))
    )
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.ffill().bfill())
    )
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    remaining = df["rainfall_mm"].isna().sum()
    print(f"Missing after cleaning: {remaining}\n")
    return df


def assign_season(month: int) -> str:
    if month in KHARIF_MONTHS:   return "Kharif"
    elif month in RABI_MONTHS:   return "Rabi"
    else:                        return "Zaid"


# =============================================================================
# SECTION 4 — STANDARDIZED PRECIPITATION INDEX (SPI)
# =============================================================================

"""
WHAT IS SPI AND WHY IS IT BETTER THAN RAW RAINFALL?
──────────────────────────────────────────────────────────────────────────────
The Standardized Precipitation Index (SPI) is a Z-score of rainfall:

    SPI = (X - μ) / σ

where X is the observed rainfall, μ is the historical mean for that
district-month, and σ is the historical standard deviation.

WHY THIS NORMALIZES DROUGHT ACROSS DISTRICTS:
  Adilabad receives ~180mm in June historically.
  Hyderabad receives ~80mm in June historically.

  If both districts receive 100mm in June:
    Raw rainfall says: Adilabad = 100mm, Hyderabad = 100mm (identical)
    SPI says: Adilabad = -2.1 (severe drought), Hyderabad = +0.8 (above avg)

  The raw number is the same. The meteorological reality is opposite.
  SPI makes them comparable — it tells the model where each district
  stands relative to its own normal, not an absolute scale.

INTERPRETATION:
  SPI > +1.5  : Extremely wet
  SPI > +1.0  : Very wet
  SPI 0 to +1 : Near to above normal
  SPI -1 to 0 : Near to below normal
  SPI < -1.0  : Moderate drought
  SPI < -1.5  : Severe drought
  SPI < -2.0  : Extreme drought
"""

def calculate_spi(daily_df: pd.DataFrame,
                  window_days: int = 30) -> pd.DataFrame:
    """
    Computes the 30-day rolling SPI for each district.
    SPI is added as a new column 'spi_30d' to the daily DataFrame.

    Steps:
      1. Compute rolling 30-day sum of rainfall per district
      2. For each district-month, compute historical mean and std
         across all years
      3. Z-score = (rolling_sum - mean) / std

    Args:
        daily_df     : Cleaned daily DataFrame with district_name,
                       date, year, month, rainfall_mm
        window_days  : Rolling window size (default 30 days)

    Returns:
        daily_df with new column 'spi_30d'
    """

    print("── Computing 30-day SPI ────────────────────────────────────────")

    df = daily_df.copy()
    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    # Step 1: Rolling 30-day sum per district
    # min_periods=1 prevents NaN for the first few days of each district's series
    df["rolling_30d_sum"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.rolling(window=window_days, min_periods=1).sum())
    )

    # Step 2: Historical mean and std of 30-day rolling sum
    # Grouped by district + month so seasonal patterns are preserved.
    # A June SPI is normalised against other June values only.
    dist_month_stats = (
        df.groupby(["district_name", "month"])["rolling_30d_sum"]
        .agg(
            spi_mean = "mean",
            spi_std  = "std"
        )
        .reset_index()
    )

    df = df.merge(dist_month_stats, on=["district_name", "month"], how="left")

    # Step 3: Z-score = SPI
    # np.where guards against division by zero (std = 0 for very uniform months)
    df["spi_30d"] = np.where(
        df["spi_std"] > 0,
        (df["rolling_30d_sum"] - df["spi_mean"]) / df["spi_std"],
        0.0
    )
    df["spi_30d"] = df["spi_30d"].round(3)

    # Drop helper columns — not needed downstream
    df = df.drop(columns=["rolling_30d_sum", "spi_mean", "spi_std"])

    print(f"  SPI range: {df['spi_30d'].min():.2f} to {df['spi_30d'].max():.2f}")
    print(f"  SPI mean:  {df['spi_30d'].mean():.3f} (should be ~0)\n")

    return df


# =============================================================================
# SECTION 5 — ENSO CONTEXT
# =============================================================================

def add_enso_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ENSO state columns to any DataFrame that has a 'year' column.
    Works on both daily and seasonal/monthly DataFrames.

    Columns added:
      enso_state : string label e.g. "La Niña", "Strong El Niño"
      enso_code  : integer encoding (0=La Niña, 1=Neutral, 2=El Niño)
                   Used as a numeric feature by the Random Forest.

    WHY ENSO MATTERS FOR KHARIF PREDICTION:
    The Indian Summer Monsoon is strongly anti-correlated with El Niño.
    During El Niño years, the Walker Circulation weakens, reducing moisture
    transport to South Asia. During La Niña, the opposite enhances rainfall.
    Adding ENSO as a feature gives the model the large-scale climate context
    that explains why some years are systematically wetter or drier than
    others — independent of district-level rainfall patterns.
    """

    enso_df = pd.DataFrame.from_dict(ENSO_LOOKUP, orient="index").reset_index()
    enso_df = enso_df.rename(columns={"index": "year"})
    enso_df["year"] = enso_df["year"].astype(int)

    df = df.merge(enso_df, on="year", how="left")

    # Fill years not in lookup with Neutral
    df["enso_state"] = df["enso_state"].fillna("Neutral")
    df["enso_code"]  = df["enso_code"].fillna(1).astype(int)

    print(f"ENSO context added. Distribution:")
    print(df.groupby("enso_state")["year"].nunique()
          .rename("years").to_string())
    print()

    return df


# =============================================================================
# SECTION 6 — GEE FEATURE ENRICHMENT
# =============================================================================

def enrich_with_gee_features(seasonal_df: pd.DataFrame,
                               district_geometries: dict,
                               use_gee: bool = False) -> pd.DataFrame:
    """
    Merges GEE-derived soil moisture and temperature features into the
    seasonal DataFrame. One row per (district, year) is added.

    Args:
        seasonal_df          : Seasonal aggregates DataFrame
        district_geometries  : Dict of {district_name: geometry_geojson_dict}
                               Pre-computed from GeoJSON shapefile.
        use_gee              : Set True only when GEE is authenticated and
                               you want to re-fetch. False uses cached values
                               from data/external/gee_features.csv if present.

    Returns:
        seasonal_df with added columns:
          susm_may_mean         — pre-monsoon sub-surface soil moisture (mm)
          susm_may_max          — peak pre-monsoon soil moisture (mm)
          temp_june_mean        — mean June temperature (°C)
          temp_june_stress_days — days > 35°C in June
    """

    GEE_CACHE_PATH = "data/external/gee_features.csv"

    # ── Use cached GEE data if available and GEE not requested ────────────
    if not use_gee and os.path.exists(GEE_CACHE_PATH):
        print(f"Loading cached GEE features from {GEE_CACHE_PATH}")
        gee_df = pd.read_csv(GEE_CACHE_PATH)
        seasonal_df = seasonal_df.merge(
            gee_df, on=["district_name", "year"], how="left"
        )
        print(f"GEE features merged. "
              f"Coverage: {gee_df['district_name'].nunique()} districts\n")
        return seasonal_df

    # ── Fetch from GEE ─────────────────────────────────────────────────────
    if use_gee:
        try:
            from gee_gateway import initialize_gee, fetch_district_climate_features
        except ImportError:
            print("⚠️  gee_gateway.py not found. Skipping GEE enrichment.")
            return _add_empty_gee_columns(seasonal_df)

        if not initialize_gee():
            print("⚠️  GEE not available. Skipping enrichment.")
            return _add_empty_gee_columns(seasonal_df)

        print("Fetching GEE features for all districts and years...")
        gee_rows = []

        for district in seasonal_df["district_name"].unique():
            if district not in district_geometries:
                print(f"  ⚠️  No geometry for {district} — skipping.")
                continue

            geom = district_geometries[district]
            years = sorted(
                seasonal_df[seasonal_df["district_name"] == district]
                ["year"].unique()
            )

            for year in years:
                print(f"  Fetching {district} {year}...", end=" ")
                try:
                    row = fetch_district_climate_features(district, geom, year)
                    gee_rows.append(row)
                    print("✅")
                except Exception as e:
                    print(f"❌ {e}")
                    gee_rows.append({
                        "district_name"        : district,
                        "year"                 : year,
                        "susm_may_mean"        : np.nan,
                        "susm_may_max"         : np.nan,
                        "temp_june_mean"       : np.nan,
                        "temp_june_stress_days": np.nan,
                    })

        if gee_rows:
            gee_df = pd.DataFrame(gee_rows)
            os.makedirs("data/external", exist_ok=True)
            gee_df.to_csv(GEE_CACHE_PATH, index=False)
            print(f"\nGEE features saved → {GEE_CACHE_PATH}")

            seasonal_df = seasonal_df.merge(
                gee_df, on=["district_name", "year"], how="left"
            )

    else:
        print("⚠️  GEE not enabled and no cache found. "
              "Run with use_gee=True after authenticating GEE.")
        seasonal_df = _add_empty_gee_columns(seasonal_df)

    return seasonal_df


def _add_empty_gee_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds GEE columns as NaN when GEE is unavailable."""
    for col in ["susm_may_mean", "susm_may_max",
                "temp_june_mean", "temp_june_stress_days"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# =============================================================================
# SECTION 7 — AGGREGATE TO MONTHLY TOTALS
# =============================================================================

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses daily rows into one row per (district, year, month).
    Also aggregates SPI to monthly mean."""

    agg_dict = {
        "total_rainfall_mm" : ("rainfall_mm", "sum"),
        "rainy_days"        : ("rainfall_mm", lambda x: (x > 2.5).sum()),
        "data_days"         : ("rainfall_mm", "count"),
    }

    # Include SPI if computed
    if "spi_30d" in df.columns:
        agg_dict["mean_spi_30d"] = ("spi_30d", "mean")

    monthly = (
        df.groupby(
            ["state_name", "district_name", "year", "month"], as_index=False
        ).agg(**agg_dict)
    )

    monthly = monthly.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    print(f"Monthly aggregation: {len(monthly):,} rows\n")
    return monthly


# =============================================================================
# SECTION 8 — AGGREGATE TO SEASONAL TOTALS
# =============================================================================

def aggregate_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses daily rows into one row per (district, year, season).
    Rabi seasons are attributed to the start year."""

    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["year"] - 1
        if (row["season"] == "Rabi" and row["month"] <= 2)
        else row["year"],
        axis=1
    )

    agg_dict = {
        "total_rainfall_mm" : ("rainfall_mm", "sum"),
        "rainy_days"        : ("rainfall_mm", lambda x: (x > 2.5).sum()),
        "data_days"         : ("rainfall_mm", "count"),
    }
    if "spi_30d" in df.columns:
        agg_dict["mean_spi_30d"] = ("spi_30d", "mean")

    seasonal = (
        df.groupby(
            ["state_name", "district_name", "season_year", "season"],
            as_index=False
        ).agg(**agg_dict)
    )

    seasonal = seasonal.rename(columns={"season_year": "year"})
    seasonal = seasonal.sort_values(
        ["district_name", "year", "season"]
    ).reset_index(drop=True)

    print(f"Seasonal aggregation: {len(seasonal):,} rows\n")
    return seasonal


# =============================================================================
# SECTION 9 — LPA AND DEPARTURE FROM MEAN
# =============================================================================

def calculate_lpa_and_departure(monthly_df: pd.DataFrame,
                                 seasonal_df: pd.DataFrame):
    """Computes LPA and Departure% for monthly and seasonal aggregates."""

    # ── Monthly ────────────────────────────────────────────────────────────
    monthly_lpa = (
        monthly_df
        .groupby(["district_name", "month"], as_index=False)["total_rainfall_mm"]
        .mean().rename(columns={"total_rainfall_mm": "lpa_mm"})
    )
    monthly_with_lpa = monthly_df.merge(
        monthly_lpa, on=["district_name", "month"], how="left"
    )
    monthly_with_lpa["departure_pct"] = np.where(
        monthly_with_lpa["lpa_mm"] > 0,
        ((monthly_with_lpa["total_rainfall_mm"] - monthly_with_lpa["lpa_mm"])
         / monthly_with_lpa["lpa_mm"]) * 100, 0
    )
    monthly_with_lpa["departure_pct"] = monthly_with_lpa["departure_pct"].round(2)
    monthly_with_lpa["lpa_mm"]        = monthly_with_lpa["lpa_mm"].round(2)
    monthly_with_lpa["anomaly_category"] = monthly_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    # ── Seasonal ───────────────────────────────────────────────────────────
    seasonal_lpa = (
        seasonal_df
        .groupby(["district_name", "season"], as_index=False)["total_rainfall_mm"]
        .mean().rename(columns={"total_rainfall_mm": "lpa_mm"})
    )
    seasonal_with_lpa = seasonal_df.merge(
        seasonal_lpa, on=["district_name", "season"], how="left"
    )
    seasonal_with_lpa["departure_pct"] = np.where(
        seasonal_with_lpa["lpa_mm"] > 0,
        ((seasonal_with_lpa["total_rainfall_mm"] - seasonal_with_lpa["lpa_mm"])
         / seasonal_with_lpa["lpa_mm"]) * 100, 0
    )
    seasonal_with_lpa["departure_pct"]    = seasonal_with_lpa["departure_pct"].round(2)
    seasonal_with_lpa["lpa_mm"]           = seasonal_with_lpa["lpa_mm"].round(2)
    seasonal_with_lpa["anomaly_category"] = seasonal_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    print("LPA and Departure calculations complete.\n")
    return monthly_with_lpa, seasonal_with_lpa


def classify_anomaly(departure_pct: float) -> str:
    if departure_pct > 20:    return "Large Excess"
    elif departure_pct > 5:   return "Above Normal"
    elif departure_pct >= -5: return "Normal"
    elif departure_pct >= -20: return "Below Normal"
    else:                     return "Large Deficit"


# =============================================================================
# SECTION 10 — ABOVE-NORMAL PROBABILITY
# =============================================================================

def calculate_above_normal_probability(seasonal_with_lpa: pd.DataFrame) -> pd.DataFrame:
    """Empirical P(Above-Normal) per district-season."""

    seasonal_with_lpa["is_above_normal"] = (
        seasonal_with_lpa["departure_pct"] > 5
    ).astype(int)

    prob_df = (
        seasonal_with_lpa
        .groupby(["district_name", "season"], as_index=False)
        .agg(
            years_above_normal = ("is_above_normal", "sum"),
            total_years        = ("is_above_normal", "count")
        )
    )
    prob_df["prob_above_normal_pct"] = (
        (prob_df["years_above_normal"] / prob_df["total_years"]) * 100
    ).round(1)

    print("Above-Normal probability complete.\n")
    return prob_df


# =============================================================================
# SECTION 11 — SAVE OUTPUTS
# =============================================================================

def save_outputs(daily_clean: pd.DataFrame,
                 monthly_final: pd.DataFrame,
                 seasonal_final: pd.DataFrame,
                 probability_df: pd.DataFrame,
                 output_folder: str = "data/processed/") -> None:
    """Saves all processed DataFrames locally. Upload 02/03/04 to Drive."""

    os.makedirs(output_folder, exist_ok=True)

    paths = {
        "01_daily_clean.csv"            : daily_clean,
        "02_monthly_with_departure.csv" : monthly_final,
        "03_seasonal_with_departure.csv": seasonal_final,
        "04_above_normal_probability.csv": probability_df,
    }

    for filename, df in paths.items():
        path = os.path.join(output_folder, filename)
        df.to_csv(path, index=False)
        print(f"  Saved → {path}  ({len(df):,} rows)")

    print("\nDone. Upload 02/03/04 to Google Drive to refresh dashboard.")


# =============================================================================
# SECTION 12 — MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_pipeline(use_gee: bool = False,
                 district_geometries: dict = None):
    """
    Full pipeline: Load → Clean → SPI → ENSO → Aggregate → LPA → Save

    Args:
        use_gee              : Set True to fetch GEE features.
                               Requires gee_gateway.py and GEE authentication.
        district_geometries  : Dict {district_name: geojson_dict}.
                               Required only when use_gee=True.
                               Load from your converted shapefile.
    """

    print("=" * 65)
    print("  RAINFALL PIPELINE v4 — START")
    print("=" * 65 + "\n")

    # 1. Load raw data
    raw_df = load_all_csvs(RAW_FILE_IDS)

    # 2. Clean and standardise
    clean_df = clean_and_standardise(raw_df)

    print(f"Date range : {clean_df['date'].min().date()} → "
          f"{clean_df['date'].max().date()}")
    print(f"Districts  : {clean_df['district_name'].nunique()}\n")

    # 3. Compute 30-day SPI on daily data
    clean_df = calculate_spi(clean_df, window_days=30)

    # 4. Add ENSO context to daily data
    clean_df = add_enso_context(clean_df)

    # 5. Aggregate to monthly and seasonal
    monthly_df  = aggregate_monthly(clean_df)
    seasonal_df = aggregate_seasonal(clean_df)

    # 6. LPA and departure
    monthly_final, seasonal_final = calculate_lpa_and_departure(
        monthly_df, seasonal_df
    )

    # 7. Add ENSO to seasonal (needed by train_model.py)
    seasonal_final = add_enso_context(seasonal_final)

    # 8. Enrich with GEE features (soil moisture + temperature)
    seasonal_final = enrich_with_gee_features(
        seasonal_final, district_geometries or {}, use_gee=use_gee
    )

    # 9. Above-normal probability
    probability_df = calculate_above_normal_probability(seasonal_final)

    # 10. Save
    print("Saving processed files...")
    save_outputs(clean_df, monthly_final, seasonal_final, probability_df)

    # 11. Preview
    print("\n── Seasonal Output (first 3 rows) ─────────────────────────────")
    preview_cols = ["district_name", "year", "season",
                    "total_rainfall_mm", "lpa_mm", "departure_pct",
                    "anomaly_category", "enso_state"]
    if "mean_spi_30d" in seasonal_final.columns:
        preview_cols.append("mean_spi_30d")
    print(seasonal_final[preview_cols].head(3).to_string())

    print("\n" + "=" * 65)
    print("  PIPELINE v4 COMPLETE")
    print("  Next: run train_model.py, upload 02/03/04 CSVs to Drive")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

with open("data/external/district_geometries.json") as f:
    district_geometries = json.load(f)

run_pipeline(use_gee=True, district_geometries=district_geometries)