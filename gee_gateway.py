"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Module: gee_gateway.py — Google Earth Engine Data Fetcher
=============================================================================
Purpose : Authenticate with GEE and fetch sub-surface soil moisture
          (SMAP) and thermal stress (ERA5-Land) for Telangana districts.
          Results are cached via @st.cache_data to prevent re-fetching.

Prerequisites:
    pip install earthengine-api

First-time setup (run once in terminal):
    python -c "import ee; ee.Authenticate()"
    → Opens browser, saves credentials to disk permanently.
    After that, ee.Initialize() works silently on every run.

Usage:
    from gee_gateway import initialize_gee, fetch_district_climate_features
=============================================================================
"""

import ee
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — GEE INITIALISATION
# =============================================================================

"""
AUTHENTICATE vs INITIALIZE — The Difference
─────────────────────────────────────────────
Authenticate : Proves your identity to Google — runs ONCE, saves a token
               to disk at ~/.config/earthengine/credentials
Initialize   : Connects to GEE servers using the saved token — runs EVERY
               time the script starts. Fast (< 1 second).

The two-attempt pattern below handles all real-world scenarios:
  1. Token exists and is valid     → Initialize succeeds silently
  2. Token expired or missing      → Falls back to full Authenticate flow
  3. No internet / account issue   → Returns False with clear error message
"""

# Set your Google Cloud Project ID here if you have one.
# Leave as None to use the default project tied to your GEE account.
GEE_PROJECT_ID = "rainfallanalytics"   # e.g. "my-rainfall-project"


def initialize_gee(project_id: str = GEE_PROJECT_ID) -> bool:
    """
    Robustly initialises the GEE connection. Tries Initialize first,
    falls back to Authenticate + Initialize if credentials are stale.

    Returns True if successful, False if GEE is unavailable.
    """

    # ── Attempt 1: Silent initialize with saved credentials ────────────────
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        # Connectivity test — tiny API call to confirm the link is live
        ee.Number(1).getInfo()
        print("✅ GEE initialised successfully.")
        return True

    except ee.EEException as e:
        print(f"⚠️  GEE auth error: {e}. Attempting re-authentication...")
    except Exception as e:
        print(f"⚠️  GEE initialize failed: {e}. Attempting authentication...")

    # ── Attempt 2: Full OAuth flow ─────────────────────────────────────────
    try:
        ee.Authenticate(auth_mode="notebook")
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        ee.Number(1).getInfo()
        print("✅ GEE re-authenticated and initialised.")
        return True

    except Exception as e:
        print(f"❌ GEE initialisation failed completely: {e}")
        print(
            "Troubleshooting:\n"
            "  1. Sign up at earthengine.google.com if you haven't.\n"
            "  2. Run: python -c \"import ee; ee.Authenticate()\"\n"
            "  3. Check your internet connection."
        )
        return False


# =============================================================================
# SECTION 2 — HELPER: IMAGE COLLECTION → DATAFRAME
# =============================================================================

def _collection_to_df(collection: ee.ImageCollection,
                       geometry: ee.Geometry,
                       bands: list,
                       scale: int) -> pd.DataFrame:
    """
    Converts a GEE ImageCollection to a Pandas DataFrame by computing the
    spatial mean of each band over the district geometry per image.

    BEGINNER CONCEPT — Server-side vs client-side:
    GEE runs computations on Google's servers. Nothing is downloaded until
    .getInfo() is called. Before that, you're just building a recipe.
    reduceRegion() collapses all pixels within the district polygon into
    a single mean value per band per date — that's the download.
    """

    def extract_mean(image: ee.Image) -> ee.Feature:
        means = image.reduceRegion(
            reducer    = ee.Reducer.mean(),
            geometry   = geometry,
            scale      = scale,
            maxPixels  = 1e9,
            bestEffort = True        # auto-downsample if geometry is too large
        )
        return ee.Feature(None, means.set(
            "date", image.date().format("YYYY-MM-dd")
        ))

    features = collection.map(extract_mean)

    try:
        feature_list = features.getInfo()["features"]
    except Exception as e:
        raise RuntimeError(
            f"GEE fetch failed: {e}\n"
            "Check internet connection and GEE account status."
        )

    if not feature_list:
        return pd.DataFrame(columns=["date"] + bands)

    rows = []
    for feat in feature_list:
        props = feat.get("properties", {})
        row   = {"date": props.get("date")}
        for band in bands:
            row[band] = props.get(band)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=bands, how="all")
    return df


# =============================================================================
# SECTION 3 — SOIL MOISTURE FETCHER (NASA SMAP)
# =============================================================================

"""
WHAT IS SUB-SURFACE SOIL MOISTURE (susm)?
──────────────────────────────────────────
SMAP (Soil Moisture Active Passive) is a NASA satellite that measures
microwave emissions from Earth's surface to infer soil water content.

  ssm  = surface soil moisture (top ~5cm) — responds in hours
  susm = sub-surface soil moisture (~100cm) — responds over weeks

WHY susm IS MORE USEFUL THAN ssm FOR SEASONAL PREDICTION:
  ssm is volatile — it saturates after 10mm of rain and dries out in
  2–3 days of sun. It captures weather, not climate.

  susm integrates weeks to months of rainfall history. It represents
  the "memory" of the land surface — how much water is stored in the
  root zone. This is what crops actually use, and what determines
  whether early rains translate into a sustained monsoon season.

  A district entering June with high susm (from pre-monsoon rains or
  a late Rabi season) requires less monsoon rain to achieve a good
  crop outcome. A district with low susm needs significantly above-
  normal monsoon rainfall just to reach normal crop performance.

  This is what makes susm a "State Variable" — it summarises the
  cumulative effect of all past rainfall without requiring the model
  to see that full history explicitly.
"""

@st.cache_data(
    ttl=86400,  # cache for 24 hours — SMAP updates every 2-3 days
    show_spinner="Fetching soil moisture from NASA SMAP..."
)
def get_soil_moisture(geometry_dict: dict,
                       start_date: str,
                       end_date: str) -> pd.DataFrame:
    """
    Fetches NASA SMAP 10km soil moisture data for a district.

    Args:
        geometry_dict : District geometry as GeoJSON dict
                        (ee.Geometry objects aren't cacheable — use .getInfo())
        start_date    : "YYYY-MM-DD" (inclusive)
        end_date      : "YYYY-MM-DD" (exclusive)

    Returns:
        DataFrame with columns:
          date        — datetime
          susm        — sub-surface soil moisture (mm), key feature
          ssm         — surface soil moisture (mm), reference
          susm_status — qualitative label for dashboard display
    """

    geometry = ee.Geometry(geometry_dict)

    collection = (
        ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture")
        .filterDate(start_date, end_date)
        .select(["ssm", "susm"])
        .map(lambda img: img.clip(geometry))
    )

    n_images = collection.size().getInfo()
    if n_images == 0:
        print(f"⚠️  No SMAP data for {start_date} → {end_date}. "
              "Data starts April 2015.")
        return pd.DataFrame(columns=["date", "ssm", "susm"])

    print(f"SMAP: {n_images} images for {start_date} → {end_date}")

    df = _collection_to_df(
        collection = collection,
        geometry   = geometry,
        bands      = ["ssm", "susm"],
        scale      = 10000   # 10km native resolution
    )

    # Add qualitative status for dashboard display
    if "susm" in df.columns:
        df["susm_status"] = df["susm"].apply(_classify_susm)

    return df


def _classify_susm(val: float) -> str:
    """Qualitative sub-surface soil moisture labels for farmer display."""
    if val is None or pd.isna(val): return "No data"
    elif val < 20:  return "Critically Dry"
    elif val < 40:  return "Dry"
    elif val < 70:  return "Adequate"
    elif val < 100: return "Moist"
    else:           return "Saturated"


# =============================================================================
# SECTION 4 — TEMPERATURE FETCHER (ECMWF ERA5-LAND)
# =============================================================================

"""
WHY THERMAL STRESS MATTERS FOR KHARIF PREDICTION
──────────────────────────────────────────────────
High temperatures amplify evapotranspiration — the rate at which water
leaves the soil and plant canopy. In a district with adequate rainfall,
a heat wave can still cause drought stress by increasing water demand
faster than rain replenishes it.

The ERA5-Land 2m temperature (temperature_2m) is stored in Kelvin.
We convert to Celsius. A district experiencing daily means >35°C in
June is under thermal stress regardless of its rainfall levels.

For the model, temperature acts as a MODIFIER of rainfall impact:
  Same june_rainfall_mm + low temperature  → better season outcome
  Same june_rainfall_mm + high temperature → worse season outcome

This interaction can't be captured from rainfall data alone.
"""

@st.cache_data(
    ttl=86400,
    show_spinner="Fetching temperature from ECMWF ERA5-Land..."
)
def get_temperature(geometry_dict: dict,
                     start_date: str,
                     end_date: str) -> pd.DataFrame:
    """
    Fetches ERA5-Land hourly temperature aggregated to daily means.

    Args:
        geometry_dict : District geometry as GeoJSON dict
        start_date    : "YYYY-MM-DD"
        end_date      : "YYYY-MM-DD"

    Returns:
        DataFrame with columns:
          date             — datetime
          temp_celsius     — daily mean 2m air temperature (°C)
          temp_stress_flag — 1 if temp > 35°C (heat stress day)
    """

    geometry = ee.Geometry(geometry_dict)

    # ERA5-Land is hourly — aggregate to daily means
    n_days = (
        datetime.strptime(end_date, "%Y-%m-%d") -
        datetime.strptime(start_date, "%Y-%m-%d")
    ).days

    hourly = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start_date, end_date)
        .select(["temperature_2m"])
        .map(lambda img: img.clip(geometry))
    )

    def daily_mean(offset):
        """Aggregate 24 hourly images into one daily mean image."""
        day_start = ee.Date(start_date).advance(offset, "day")
        day_end   = day_start.advance(1, "day")
        return (
            hourly.filterDate(day_start, day_end).mean()
            .set("system:time_start", day_start.millis())
            .set("date", day_start.format("YYYY-MM-dd"))
        )

    daily_collection = ee.ImageCollection(
        ee.List.sequence(0, n_days - 1).map(daily_mean)
    )

    n_images = daily_collection.size().getInfo()
    if n_images == 0:
        print(f"⚠️  No ERA5 data for {start_date} → {end_date}.")
        return pd.DataFrame(columns=["date", "temp_celsius"])

    print(f"ERA5-Land: {n_images} daily images for {start_date} → {end_date}")

    df = _collection_to_df(
        collection = daily_collection,
        geometry   = geometry,
        bands      = ["temperature_2m"],
        scale      = 9000   # 9km native resolution
    )

    # Convert Kelvin → Celsius
    df["temp_celsius"]     = (df["temperature_2m"] - 273.15).round(2)
    df["temp_stress_flag"] = (df["temp_celsius"] > 35).astype(int)
    df = df.drop(columns=["temperature_2m"])

    return df


# =============================================================================
# SECTION 5 — COMBINED DISTRICT CLIMATE FEATURES
# =============================================================================

def fetch_district_climate_features(district_name: str,
                                     geometry_dict: dict,
                                     year: int) -> dict:
    """
    Fetches and aggregates GEE climate features for a district's June period.
    Returns a single dict of scalar values — one row for the feature matrix.

    This is the function called by rainfall_pipeline.py during enrichment.
    It returns pre-June (May) soil moisture and June temperature stats,
    which are the most predictive windows for Kharif season outcome.

    Args:
        district_name : e.g. "Adilabad"
        geometry_dict : ee.Geometry.Rectangle(...).getInfo()
        year          : e.g. 2022

    Returns dict with keys:
        susm_may_mean       — mean sub-surface soil moisture in May (mm)
                              (pre-monsoon soil water state — the key feature)
        susm_may_max        — max susm in May (peak soil moisture)
        temp_june_mean      — mean temperature in June (°C)
        temp_june_stress_days — count of days > 35°C in June
    """

    result = {
        "district_name"         : district_name,
        "year"                  : year,
        "susm_may_mean"         : np.nan,
        "susm_may_max"          : np.nan,
        "temp_june_mean"        : np.nan,
        "temp_june_stress_days" : np.nan,
    }

    # ── May soil moisture (pre-monsoon state) ──────────────────────────────
    try:
        may_start = f"{year}-05-01"
        may_end   = f"{year}-05-31"
        sm_df = get_soil_moisture(geometry_dict, may_start, may_end)

        if not sm_df.empty and "susm" in sm_df.columns:
            result["susm_may_mean"] = round(sm_df["susm"].mean(), 3)
            result["susm_may_max"]  = round(sm_df["susm"].max(),  3)
    except Exception as e:
        print(f"  ⚠️  Soil moisture fetch failed for {district_name} {year}: {e}")

    # ── June temperature ───────────────────────────────────────────────────
    try:
        jun_start = f"{year}-06-01"
        jun_end   = f"{year}-06-30"
        temp_df = get_temperature(geometry_dict, jun_start, jun_end)

        if not temp_df.empty and "temp_celsius" in temp_df.columns:
            result["temp_june_mean"]        = round(temp_df["temp_celsius"].mean(), 2)
            result["temp_june_stress_days"] = int(
                temp_df["temp_stress_flag"].sum()
            )
    except Exception as e:
        print(f"  ⚠️  Temperature fetch failed for {district_name} {year}: {e}")

    return result


# =============================================================================
# SECTION 6 — STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    """
    Run directly to verify GEE connectivity:
        python gee_gateway.py
    """
    print("=" * 65)
    print("  GEE GATEWAY — CONNECTIVITY TEST")
    print("=" * 65 + "\n")

    if not initialize_gee():
        print("Cannot proceed — GEE not initialised.")
        exit(1)

    # Approximate bounding box for Adilabad district, Telangana
    test_geom = ee.Geometry.Rectangle([77.9, 18.8, 78.6, 19.5]).getInfo()

    print("\n── Test: Soil Moisture (May 2022) ─────────────────────────────")
    sm = get_soil_moisture(test_geom, "2022-05-01", "2022-05-31")
    print(sm.head())

    print("\n── Test: Temperature (June 2022) ──────────────────────────────")
    temp = get_temperature(test_geom, "2022-06-01", "2022-06-30")
    print(temp.head())

    print("\n── Test: Combined District Features ───────────────────────────")
    features = fetch_district_climate_features("Adilabad", test_geom, 2022)
    for k, v in features.items():
        print(f"  {k:<30} : {v}")

    print("\n" + "=" * 65)
    print("  GEE GATEWAY TEST COMPLETE")
    print("=" * 65)