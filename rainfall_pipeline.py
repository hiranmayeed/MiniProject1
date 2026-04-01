"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 2: Data Processing Pipeline
=============================================================================
Purpose : Load IMD/OGD daily CSVs, clean them, aggregate to monthly &
          seasonal totals per district, and compute Departure from Mean (%).
Author  : (your name)
Data    : IMD daily district-wise rainfall CSVs — one file per year
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import glob


# =============================================================================
# SECTION 1 — UPDATED CONFIGURATION
# =============================================================================

# Instead of a local folder, use a dictionary of your raw yearly file IDs
RAW_FILE_IDS = {
    "2018": "1GJtKaG1Ht82cDrYUSyLi63lUdx_fONrT",
    "2019": "1OS_JAicP0iE-ZiMWye8m_ynfJ5Ypf2eO",
    "2020": "1nB6qe_6SqVPDx5yyCVGJcX-ydeqtwlrE",
    "2021": "1QwtMNFi-TxS3sn2SM9BteuDGSQS3L5mW",
    "2022": "1179FbAiLT1KZJAvQiBcE8T2NQPHAC7zE",
    "2023": "1OgHoFuSwd_JUdadvuxPV1Pj0QFBE4sjf",
    "2024": "1q_yHt0UeqOzo1Kvzz8MTjaP-KEKBU3hr",
    # Add all years through 2024
}

def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# =============================================================================
# SECTION 2 — UPDATED LOAD FUNCTION
# =============================================================================

def load_all_csvs(file_ids_dict: dict) -> pd.DataFrame:
    """
    Loads yearly CSVs directly from Google Drive links.
    """
    yearly_frames = []

    print(f"Starting Drive download for years: {list(file_ids_dict.keys())}")

    for year, f_id in file_ids_dict.items():
        url = drive_url(f_id)
        try:
            # Pandas can read directly from a URL
            df_year = pd.read_csv(url)
            
            # We still add the 'source_file' column so the rest of your 
            # pipeline (cleaning/standardization) works exactly the same.
            df_year["source_file"] = f"{year}.csv"
            
            yearly_frames.append(df_year)
            print(f"  ✅ Successfully loaded {year}")
            
        except Exception as e:
            print(f"  ❌ Error loading {year}: {e}")
            print("  Check if the file is shared as 'Anyone with the link' in Drive.")

    if not yearly_frames:
        raise ValueError("No data was loaded. Pipeline cannot continue.")

    # Stack them all together
    combined_df = pd.concat(yearly_frames, ignore_index=True)
    return combined_df


# =============================================================================
# SECTION 3 — CLEAN & STANDARDISE THE RAW DATA
# =============================================================================

def clean_and_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs four cleaning steps:
      A. Rename columns to our standard internal names (in case CSVs differ).
      B. Parse the date column into a proper Python datetime object.
      C. Strip whitespace from text columns (common in IMD data).
      D. Handle missing rainfall values — explained in detail below.
    """

    # ── A. Rename columns ──────────────────────────────────────────────────
    # Maps whatever your CSV uses → our standard internal names.
    # If a column name already matches, this is a no-op (harmless).
    # NEW
    df = df.rename(columns={
    COL_DATE:     "date",
    COL_STATE:    "state_name",
    COL_DISTRICT: "district_name",
    COL_RAINFALL: "rainfall_mm",
    })
    
# Safety check — confirm the rename worked
    print(f"Columns after rename: {df.columns.tolist()}")
    assert "rainfall_mm" in df.columns, \
    f"Rename failed. COL_RAINFALL='{COL_RAINFALL}' not found in CSV. " \
    f"Available columns: {df.columns.tolist()}"
    # ── B. Parse dates ─────────────────────────────────────────────────────
    # pd.to_datetime handles most date formats automatically (YYYY-MM-DD,
    # DD-MM-YYYY, etc.).  dayfirst=True handles Indian DD-MM-YYYY convention.
    # NEW
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Extract year, month, and ISO week as separate columns.
    # Storing these avoids re-computing them in every aggregation query.
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)

    # Assign the agricultural season label to each row
    df["season"] = df["month"].apply(assign_season)

    # ── C. Strip whitespace from text ──────────────────────────────────────
    # IMD CSVs often have trailing spaces like "Warangal " which would be
    # treated as a different district from "Warangal". .str.strip() fixes this.
    df["state_name"]    = df["state_name"].str.strip().str.title()
    df["district_name"] = df["district_name"].str.strip().str.title()

    DISTRICT_NAME_CORRECTIONS = {
        "Jagtial"              : "Jagitial",
        "Jangoan"              : "Jangaon",
        "Kumuram Bheem"        : "Kumuram Bheem Asifabad",
        "Medchal-Malkajgiri"   : "Medchal Malkajgiri",
        "Rangareddy"           : "Ranga Reddy",
        "Ranjanna Sircilla"    : "Rajanna Sircilla",
        "Warangal Rural"       : "Warangal (Rural)",
        "Warangal Urban"       : "Warangal (Urban)",
    }
    df["district_name"] = df["district_name"].replace(DISTRICT_NAME_CORRECTIONS)

    # ── D. Handle missing rainfall values ──────────────────────────────────
    missing_count = df["rainfall_mm"].isna().sum()
    print(f"Missing rainfall values found: {missing_count} "
          f"({missing_count / len(df) * 100:.2f}% of rows)")

    # WHY TIME-BASED INTERPOLATION (not zero-fill, not mean-fill)?
    # ─────────────────────────────────────────────────────────────
    # Option A — Fill with 0: WRONG. A missing reading ≠ no rainfall.
    #            Zeros would artificially depress your LPA and anomaly scores.
    #
    # Option B — Fill with column mean: BAD. Rainfall is seasonal — June mean
    #            inserted into a December gap is meteorologically nonsensical.
    #
    # Option C — Time interpolation: BEST for this data. It assumes the
    #            missing day's rainfall sits somewhere between the day before
    #            and the day after — which is physically reasonable for a
    #            continuous weather variable. Works well when gaps are small
    #            (1–3 days), which is typical for IMD data.
    #
    # We sort by district + date first so interpolation stays within each
    # district's own time series and doesn't bleed across districts.

    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda series: series.interpolate(method="linear"))
    )

    # If a district has missing values at the very START or END of its series
    # (interpolation can't fill edges), backfill/forward-fill as a last resort.
    df["rainfall_mm"] = (
        df.groupby("district_name")["rainfall_mm"]
          .transform(lambda series: series.fillna(method="bfill").fillna(method="ffill"))
    )

    # Safety check — ensure rainfall is never negative after interpolation
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0)

    remaining_missing = df["rainfall_mm"].isna().sum()
    print(f"Missing values remaining after cleaning: {remaining_missing}\n")

    return df


def assign_season(month: int) -> str:
    """
    Maps a month number to its agricultural season name.
    Called row-by-row via .apply() in clean_and_standardise().
    """
    if month in KHARIF_MONTHS:
        return "Kharif"       # Sowing Jun–Sep; main Indian crop season
    elif month in RABI_MONTHS:
        return "Rabi"         # Sowing Oct–Feb; winter crop season
    else:
        return "Zaid"         # Mar–May; minor summer crop season


# =============================================================================
# SECTION 4 — AGGREGATE TO MONTHLY TOTALS PER DISTRICT
# =============================================================================

def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, month).

    Why SUM and not mean?
    → Farmers and meteorologists care about TOTAL monthly rainfall,
      not the average daily drizzle. "July got 180mm" is meaningful;
      "July averaged 5.8mm/day" is not.
    """

    monthly = (
        df.groupby(["state_name", "district_name", "year", "month"], as_index=False)
          .agg(
              total_rainfall_mm = ("rainfall_mm", "sum"),   # monthly total
              rainy_days        = ("rainfall_mm",            # days with rain > 2.5mm
                                   lambda x: (x > 2.5).sum()),
              data_days         = ("rainfall_mm", "count")  # how many days had readings
          )
    )

    # Sort for readability
    monthly = monthly.sort_values(
        ["district_name", "year", "month"]
    ).reset_index(drop=True)

    print(f"Monthly aggregation complete: {len(monthly):,} rows "
          f"(one per district-year-month)\n")
    return monthly


# =============================================================================
# SECTION 5 — AGGREGATE TO SEASONAL TOTALS PER DISTRICT
# =============================================================================

def aggregate_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses daily rows into one row per (district, year, season).

    NOTE on Rabi year attribution:
    Rabi spans October of year Y through February of year Y+1.
    We attribute the ENTIRE Rabi season to the year it STARTED (October).
    Example: Oct 2020 – Feb 2021 is labelled as Rabi 2020.
    This keeps season labels consistent with how IMD publishes them.
    """

    # Create a 'season_year' column that correctly labels the Rabi season
    df = df.copy()
    df["season_year"] = df.apply(
        lambda row: row["year"] - 1 if (row["season"] == "Rabi" and row["month"] <= 2)
                    else row["year"],
        axis=1
    )

    seasonal = (
        df.groupby(["state_name", "district_name", "season_year", "season"],
                   as_index=False)
          .agg(
              total_rainfall_mm = ("rainfall_mm", "sum"),
              rainy_days        = ("rainfall_mm", lambda x: (x > 2.5).sum()),
              data_days         = ("rainfall_mm", "count")
          )
    )

    seasonal = seasonal.rename(columns={"season_year": "year"})
    seasonal = seasonal.sort_values(
        ["district_name", "year", "season"]
    ).reset_index(drop=True)

    print(f"Seasonal aggregation complete: {len(seasonal):,} rows "
          f"(one per district-year-season)\n")
    return seasonal


# =============================================================================
# SECTION 6 — CALCULATE LONG PERIOD AVERAGE (LPA) AND DEPARTURE FROM MEAN
# =============================================================================

def calculate_lpa_and_departure(monthly_df: pd.DataFrame,
                                 seasonal_df: pd.DataFrame):
    """
    Computes:
      1. LPA — the multi-year average rainfall for each district+month
                and each district+season.
      2. Departure from Mean (%) — how far each year's actual rainfall
         deviates from that LPA, expressed as a percentage.

    Formula:
        Departure % = ((Actual - LPA) / LPA) * 100

    A positive value = above-normal rainfall (good for most Kharif crops).
    A negative value = below-normal / drought signal.
    """

    # ── Monthly LPA & Departure ────────────────────────────────────────────

    # Step 1: Compute the LPA — average across all years for each
    #         district+month combination.
    monthly_lpa = (
        monthly_df
        .groupby(["district_name", "month"], as_index=False)
        ["total_rainfall_mm"]
        .mean()
        .rename(columns={"total_rainfall_mm": "lpa_mm"})
    )

    # Step 2: Merge the LPA back into the monthly DataFrame so each row now
    #         knows its own LPA benchmark.
    monthly_with_lpa = monthly_df.merge(
        monthly_lpa,
        on=["district_name", "month"],
        how="left"
    )

    # Step 3: Calculate Departure %.
    # np.where guards against division by zero (if LPA is 0 for dry months).
    monthly_with_lpa["departure_pct"] = np.where(
        monthly_with_lpa["lpa_mm"] > 0,
        ((monthly_with_lpa["total_rainfall_mm"] - monthly_with_lpa["lpa_mm"])
         / monthly_with_lpa["lpa_mm"]) * 100,
        0   # If LPA is 0 (historically dry month), departure is set to 0
    )

    # Round to 2 decimal places for readability
    monthly_with_lpa["departure_pct"] = monthly_with_lpa["departure_pct"].round(2)
    monthly_with_lpa["lpa_mm"]        = monthly_with_lpa["lpa_mm"].round(2)

    # Step 4: Apply IMD's official anomaly category labels
    monthly_with_lpa["anomaly_category"] = monthly_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    # ── Seasonal LPA & Departure ───────────────────────────────────────────

    seasonal_lpa = (
        seasonal_df
        .groupby(["district_name", "season"], as_index=False)
        ["total_rainfall_mm"]
        .mean()
        .rename(columns={"total_rainfall_mm": "lpa_mm"})
    )

    seasonal_with_lpa = seasonal_df.merge(
        seasonal_lpa,
        on=["district_name", "season"],
        how="left"
    )

    seasonal_with_lpa["departure_pct"] = np.where(
        seasonal_with_lpa["lpa_mm"] > 0,
        ((seasonal_with_lpa["total_rainfall_mm"] - seasonal_with_lpa["lpa_mm"])
         / seasonal_with_lpa["lpa_mm"]) * 100,
        0
    )

    seasonal_with_lpa["departure_pct"]    = seasonal_with_lpa["departure_pct"].round(2)
    seasonal_with_lpa["lpa_mm"]           = seasonal_with_lpa["lpa_mm"].round(2)
    seasonal_with_lpa["anomaly_category"] = seasonal_with_lpa["departure_pct"].apply(
        classify_anomaly
    )

    print("LPA and Departure calculations complete.\n")
    return monthly_with_lpa, seasonal_with_lpa


def classify_anomaly(departure_pct: float) -> str:
    """
    Applies IMD's official 5-tier anomaly classification.
    Called row-by-row via .apply() in calculate_lpa_and_departure().

    Thresholds source: India Meteorological Department seasonal outlook docs.
    """
    if departure_pct > 20:
        return "Large Excess"
    elif departure_pct > 5:
        return "Above Normal"
    elif departure_pct >= -5:
        return "Normal"
    elif departure_pct >= -20:
        return "Below Normal"
    else:
        return "Large Deficit"


# =============================================================================
# SECTION 7 — PROBABILITY OF ABOVE-NORMAL (YOUR PREDICTION METRIC)
# =============================================================================

def calculate_above_normal_probability(seasonal_with_lpa: pd.DataFrame) -> pd.DataFrame:
    """
    For each district+season, computes the empirical probability of receiving
    Above Normal or Large Excess rainfall — i.e., how many of the 8 years
    cleared the +5% departure threshold.

    Formula: P(Above Normal) = count(years where departure > +5%) / total_years

    This is your headline farmer-facing prediction number — explainable,
    data-grounded, and requires no ML model.
    """

    # Flag each row: 1 if above-normal, 0 if not
    seasonal_with_lpa["is_above_normal"] = (
        seasonal_with_lpa["departure_pct"] > 5
    ).astype(int)

    # Aggregate: count above-normal years and total years per district+season
    prob_df = (
        seasonal_with_lpa
        .groupby(["district_name", "season"], as_index=False)
        .agg(
            years_above_normal = ("is_above_normal", "sum"),
            total_years        = ("is_above_normal", "count")
        )
    )

    # Divide to get probability, round to 1 decimal for display
    prob_df["prob_above_normal_pct"] = (
        (prob_df["years_above_normal"] / prob_df["total_years"]) * 100
    ).round(1)

    print("Above-Normal probability calculation complete.\n")
    return prob_df


# =============================================================================
# SECTION 8 — SAVE ALL OUTPUT FILES
# =============================================================================

def save_outputs(daily_clean: pd.DataFrame,
                 monthly_final: pd.DataFrame,
                 seasonal_final: pd.DataFrame,
                 probability_df: pd.DataFrame,
                 output_folder: str = "data/processed/") -> None:
    """
    Saves all processed DataFrames to CSV files in the output folder.
    These CSVs will feed directly into your dashboard in the next phase.
    """

    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    paths = {
        "01_daily_clean.csv"           : daily_clean,
        "02_monthly_with_departure.csv": monthly_final,
        "03_seasonal_with_departure.csv": seasonal_final,
        "04_above_normal_probability.csv": probability_df,
    }

    for filename, df in paths.items():
        full_path = os.path.join(output_folder, filename)
        df.to_csv(full_path, index=False)
        print(f"  Saved → {full_path}  ({len(df):,} rows)")

    print("\nAll outputs saved successfully.")


# =============================================================================
# SECTION 9 — MAIN PIPELINE ORCHESTRATOR
# Calls every section above in the correct order.
# =============================================================================

def run_pipeline():
    """
    Master function — runs the full pipeline end to end.
    Call this from a notebook or terminal: python rainfall_pipeline.py
    """

    print("=" * 65)
    print("  RAINFALL PIPELINE — START")
    print("=" * 65 + "\n")

    # Step 1: Load and merge all yearly CSVs
    raw_df = load_all_csvs(DATA_FOLDER)

    # Step 2: Clean, standardise, and fill missing values
    clean_df = clean_and_standardise(raw_df)

    # Quick sanity check — print date range and district count
    print(f"Date range in data : {clean_df['date'].min()} → {clean_df['date'].max()}")
    print(f"Unique districts   : {clean_df['district_name'].nunique()}")
    print(f"Unique states      : {clean_df['state_name'].nunique()}\n")

    # Step 3: Aggregate to monthly and seasonal totals
    monthly_df  = aggregate_monthly(clean_df)
    seasonal_df = aggregate_seasonal(clean_df)

    # Step 4: Calculate LPA and Departure from Mean
    monthly_final, seasonal_final = calculate_lpa_and_departure(
        monthly_df, seasonal_df
    )

    # Step 5: Calculate Above-Normal probability (your prediction metric)
    probability_df = calculate_above_normal_probability(seasonal_final)

    # Step 6: Save all outputs
    print("\nSaving processed files …")
    save_outputs(clean_df, monthly_final, seasonal_final, probability_df)

    # Step 7: Preview the key output tables
    print("\n── Monthly Output (first 5 rows) ──────────────────────────")
    print(monthly_final[["district_name", "year", "month",
                          "total_rainfall_mm", "lpa_mm",
                          "departure_pct", "anomaly_category"]].head())

    print("\n── Seasonal Output (first 5 rows) ─────────────────────────")
    print(seasonal_final[["district_name", "year", "season",
                           "total_rainfall_mm", "lpa_mm",
                           "departure_pct", "anomaly_category"]].head())

    print("\n── Above-Normal Probability (first 5 rows) ─────────────────")
    print(probability_df.head())

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — check data/processed/ for outputs")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # This block runs only when you execute this file directly.
    # It does NOT run when you import this file as a module.
    run_pipeline()