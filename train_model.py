"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 3: Prediction Engine — v4 (Climate Context Features)
=============================================================================
New features over v3:
  - june_spi_30d          : SPI Z-score for June (normalised drought level)
  - enso_code             : ENSO state (0=La Niña, 1=Neutral, 2=El Niño)
  - susm_may_mean         : pre-monsoon sub-surface soil moisture (mm)
  - susm_may_max          : peak pre-monsoon soil moisture (mm)
  - temp_june_mean        : mean June temperature (°C)
  - temp_june_stress_days : heat stress days in June (>35°C)

GridSearchCV re-runs with the expanded feature set. New optimal
hyperparameters may differ from v3 now that the model has climate context.

Run with: python train_model.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

SEASONAL_DATA_PATH  = "data/processed/03_seasonal_with_departure.csv"
MONTHLY_DATA_PATH   = "data/processed/02_monthly_with_departure.csv"
DAILY_DATA_PATH     = "data/processed/01_daily_clean.csv"
MODEL_OUTPUT_FOLDER = "models/"

PREDICTOR_MONTH        = 6
TARGET_SEASON          = "Kharif"
ABOVE_NORMAL_THRESHOLD = 5.0


# =============================================================================
# SECTION 2 — LOAD DATA
# =============================================================================

def load_data() -> tuple:
    """Loads seasonal, monthly, and daily CSVs from data/processed/."""

    for path in [SEASONAL_DATA_PATH, MONTHLY_DATA_PATH, DAILY_DATA_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: '{path}'\n"
                "Run rainfall_pipeline.py first."
            )

    seasonal_df = pd.read_csv(SEASONAL_DATA_PATH)
    monthly_df  = pd.read_csv(MONTHLY_DATA_PATH)
    daily_df    = pd.read_csv(DAILY_DATA_PATH, parse_dates=["date"])

    print(f"Seasonal : {len(seasonal_df):,} rows | "
          f"{seasonal_df['district_name'].nunique()} districts | "
          f"years: {sorted(seasonal_df['year'].unique().tolist())}")
    print(f"Monthly  : {len(monthly_df):,} rows")
    print(f"Daily    : {len(daily_df):,} rows\n")

    return seasonal_df, monthly_df, daily_df


# =============================================================================
# SECTION 3 — MASTER FEATURE LIST
# =============================================================================

def get_feature_columns() -> list:
    """
    Complete list of features fed to the model — v4 edition.
    ORDER MATTERS. make_ml_prediction() in app.py must match this exactly.

    Feature groups:
      Group A — Core June rainfall signal (4 features)
      Group B — Temporal / pattern features (5 features)
      Group C — Climate context: SPI + ENSO (2 features)
      Group D — GEE features: soil moisture + temperature (4 features)
      Group E — District historical context (3 features)
      Group F — Identity encoding (2 features)

    Total: 20 features
    GEE features (susm_may_mean, susm_may_max, temp_june_mean,
    temp_june_stress_days) will be NaN if GEE hasn't been run yet.
    The model handles NaN gracefully via imputation in build_feature_matrix().
    """
    return [
        # ── Group A: Core June signal ─────────────────────────────────────
        "june_rainfall_mm",
        "june_departure_pct",
        "june_vs_district_mean",
        "june_was_above_normal",

        # ── Group B: Temporal pattern features ───────────────────────────
        "june_rolling_7d_avg",
        "june_cumulative_rain",
        "june_rain_lag_1d",
        "june_rain_lag_7d",
        "june_dry_streak",

        # ── Group C: Climate context ──────────────────────────────────────
        "june_spi_30d",             # SPI Z-score — normalised drought level
        "enso_code",                # 0=La Niña, 1=Neutral, 2=El Niño

        # ── Group D: GEE features (NaN if GEE not run) ───────────────────
        "susm_may_mean",            # Pre-monsoon sub-surface soil moisture
        "susm_may_max",             # Peak pre-monsoon soil moisture
        "temp_june_mean",           # Mean June temperature (°C)
        "temp_june_stress_days",    # Days > 35°C in June

        # ── Group E: District historical context ──────────────────────────
        "district_mean_mm",
        "district_std_mm",
        "district_cv",

        # ── Group F: Identity encoding ────────────────────────────────────
        "district_encoded",
        "state_encoded",
    ]


# =============================================================================
# SECTION 4 — TEMPORAL FEATURES FROM DAILY DATA
# =============================================================================

def compute_june_temporal_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling, cumulative, lag, and dry-streak features
    for June only. Returns one row per (district_name, year).
    """

    print("── Computing June temporal features ───────────────────────────")

    df = daily_df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.sort_values(["district_name", "date"]).reset_index(drop=True)

    # 7-day rolling average
    df["rolling_7d"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    # Cumulative within each month
    df["cumulative_rain"] = (
        df.groupby(["district_name", "year", "month"])["rainfall_mm"]
        .transform("cumsum")
    )

    # Lag features
    df["rain_lag_1d"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.shift(1))
    ).fillna(0)

    df["rain_lag_7d"] = (
        df.groupby("district_name")["rainfall_mm"]
        .transform(lambda s: s.shift(7))
    ).fillna(0)

    # Dry streak (max consecutive days < 2.5mm)
    def max_dry_streak(series: pd.Series) -> int:
        max_s = curr_s = 0
        for v in series:
            if v < 2.5:
                curr_s += 1
                max_s   = max(max_s, curr_s)
            else:
                curr_s  = 0
        return max_s

    june_df = df[df["month"] == PREDICTOR_MONTH].copy()

    june_features = (
        june_df.groupby(["district_name", "year"])
        .agg(
            june_rolling_7d_avg  = ("rolling_7d",      "mean"),
            june_cumulative_rain = ("cumulative_rain",  "max"),
            june_rain_lag_1d     = ("rain_lag_1d",      "mean"),
            june_rain_lag_7d     = ("rain_lag_7d",      "mean"),
            june_dry_streak      = ("rainfall_mm",
                                    lambda x: max_dry_streak(x)),
        )
        .reset_index()
    )

    print(f"  Temporal features: "
          f"{june_features['district_name'].nunique()} districts × "
          f"{june_features['year'].nunique()} years\n")

    return june_features


# =============================================================================
# SECTION 5 — SPI FEATURES FROM DAILY DATA
# =============================================================================

def compute_june_spi(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the mean 30-day SPI for June per (district, year).
    Requires 'spi_30d' column already computed by the pipeline.
    If not present, returns empty DataFrame with correct columns.
    """

    if "spi_30d" not in daily_df.columns:
        print("  ⚠️  spi_30d not in daily data. "
              "Run rainfall_pipeline.py v4 to compute SPI.")
        return pd.DataFrame(columns=["district_name", "year", "june_spi_30d"])

    june_df = daily_df[daily_df["month"] == PREDICTOR_MONTH].copy() \
              if "month" in daily_df.columns else \
              daily_df[pd.to_datetime(daily_df["date"]).dt.month == PREDICTOR_MONTH].copy()

    spi_agg = (
        june_df.groupby(["district_name", "year"])["spi_30d"]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"spi_30d": "june_spi_30d"})
    )

    return spi_agg


# =============================================================================
# SECTION 6 — BUILD FEATURE MATRIX
# =============================================================================

def build_feature_matrix(seasonal_df: pd.DataFrame,
                          monthly_df:  pd.DataFrame,
                          daily_df:    pd.DataFrame) -> tuple:
    """
    Merges all feature sources into one ML-ready matrix.
    NaN GEE features are median-imputed per district so the model
    can train even before GEE data is available.
    """

    # ── Target variable ────────────────────────────────────────────────────
    kharif_df = seasonal_df[seasonal_df["season"] == TARGET_SEASON].copy()
    print(f"Kharif rows: {len(kharif_df)}")

    kharif_df["target_above_normal"] = (
        kharif_df["departure_pct"] > ABOVE_NORMAL_THRESHOLD
    ).astype(int)

    above = kharif_df["target_above_normal"].sum()
    print(f"Above-Normal: {above}/{len(kharif_df)} "
          f"({above/len(kharif_df)*100:.1f}%)\n")

    # ── Core June features from monthly ───────────────────────────────────
    june_monthly = monthly_df[monthly_df["month"] == PREDICTOR_MONTH][
        ["district_name", "year", "total_rainfall_mm", "departure_pct"]
    ].copy().rename(columns={
        "total_rainfall_mm": "june_rainfall_mm",
        "departure_pct"    : "june_departure_pct",
    })

    # ── Temporal features from daily ───────────────────────────────────────
    june_temporal = compute_june_temporal_features(daily_df)

    # ── SPI from daily ─────────────────────────────────────────────────────
    june_spi = compute_june_spi(daily_df)

    # ── ENSO from seasonal ────────────────────────────────────────────────
    enso_cols = ["district_name", "year", "enso_code"]
    if "enso_code" in kharif_df.columns:
        enso_df = kharif_df[enso_cols].copy()
    else:
        enso_df = pd.DataFrame(columns=enso_cols)
        print("  ⚠️  enso_code not in seasonal data. "
              "Run rainfall_pipeline.py v4.")

    # ── GEE features from seasonal ────────────────────────────────────────
    gee_cols = ["district_name", "year",
                "susm_may_mean", "susm_may_max",
                "temp_june_mean", "temp_june_stress_days"]
    gee_present = [c for c in gee_cols if c in kharif_df.columns]
    gee_df = kharif_df[gee_present].copy() if len(gee_present) > 2 else \
             pd.DataFrame(columns=gee_cols)

    # ── District historical stats ──────────────────────────────────────────
    district_stats = (
        kharif_df.groupby("district_name")["total_rainfall_mm"]
        .agg(
            district_mean_mm = "mean",
            district_std_mm  = "std",
            district_cv      = lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        .reset_index()
    )

    # ── Merge all sources ──────────────────────────────────────────────────
    features_df = kharif_df[
        ["district_name", "state_name", "year", "target_above_normal"]
    ].copy()

    features_df = features_df.merge(june_monthly,   on=["district_name","year"], how="left")
    features_df = features_df.merge(june_temporal,  on=["district_name","year"], how="left")
    features_df = features_df.merge(june_spi,       on=["district_name","year"], how="left")
    features_df = features_df.merge(district_stats, on="district_name",          how="left")

    if not enso_df.empty:
        features_df = features_df.merge(enso_df, on=["district_name","year"], how="left")
    else:
        features_df["enso_code"] = 1   # default to Neutral

    if not gee_df.empty and len(gee_present) > 2:
        features_df = features_df.merge(gee_df, on=["district_name","year"], how="left")
    else:
        for col in ["susm_may_mean","susm_may_max",
                    "temp_june_mean","temp_june_stress_days"]:
            features_df[col] = np.nan

    # ── Derived features ───────────────────────────────────────────────────
    features_df["june_vs_district_mean"] = (
        features_df["june_rainfall_mm"] / features_df["district_mean_mm"]
    ).round(4)

    features_df["june_was_above_normal"] = (
        features_df["june_departure_pct"] > ABOVE_NORMAL_THRESHOLD
    ).astype(int)

    # ── Label encoding ─────────────────────────────────────────────────────
    le_district = LabelEncoder()
    le_state    = LabelEncoder()
    features_df["district_encoded"] = le_district.fit_transform(
        features_df["district_name"]
    )
    features_df["state_encoded"] = le_state.fit_transform(
        features_df["state_name"]
    )

    # ── Drop rows missing core features ────────────────────────────────────
    before = len(features_df)
    features_df = features_df.dropna(
        subset=["june_rainfall_mm", "june_departure_pct",
                "june_rolling_7d_avg"]
    )
    dropped = before - len(features_df)
    if dropped > 0:
        print(f"Dropped {dropped} rows missing core June features.")

    # ── Median-impute GEE features (NaN when GEE not run) ─────────────────
    gee_feature_cols = ["susm_may_mean", "susm_may_max",
                        "temp_june_mean", "temp_june_stress_days"]
    for col in gee_feature_cols:
        if col in features_df.columns and features_df[col].isna().any():
            median_val = features_df[col].median()
            n_filled   = features_df[col].isna().sum()
            features_df[col] = features_df[col].fillna(median_val)
            if n_filled > 0:
                print(f"  Imputed {n_filled} NaN in '{col}' "
                      f"with median {median_val:.2f}")

    # Fill remaining NaN lag/streak/spi with 0
    for col in ["june_rain_lag_1d", "june_rain_lag_7d",
                "june_dry_streak",  "june_spi_30d", "enso_code"]:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0)

    n_gee_available = features_df["susm_may_mean"].notna().sum()
    print(f"\nFeature matrix: {features_df.shape}")
    print(f"GEE features available: {n_gee_available}/{len(features_df)} rows")
    print(f"Districts: {features_df['district_name'].nunique()}\n")

    return features_df, le_district, le_state


# =============================================================================
# SECTION 7 — GRIDSEARCHCV
# =============================================================================

def run_gridsearch(X: np.ndarray, y: np.ndarray) -> tuple:
    """GridSearchCV: 18 parameter combinations × 5-fold CV = 90 fits."""

    print("=" * 65)
    print("  GRIDSEARCHCV — HYPERPARAMETER TUNING")
    print("=" * 65)

    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_features": ["sqrt", "log2"],
        "max_depth"   : [None, 5, 10],
    }

    n_combos = 18
    print(f"\nParameter grid ({n_combos} combinations × 5-fold = 90 fits):")
    for p, v in param_grid.items():
        print(f"  {p}: {v}")

    base_rf = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=base_rf, param_grid=param_grid,
        cv=5, scoring="accuracy", n_jobs=-1, verbose=1, refit=True
    )

    print(f"\nRunning... (2–5 minutes)\n")
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_estimators", "param_max_features", "param_max_depth",
         "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    print("\n── All combinations ranked ───────────────────────────────────")
    print(results_df.to_string(index=False))
    print("\n── Best Parameters ───────────────────────────────────────────")
    for p, v in grid_search.best_params_.items():
        print(f"  {p:<20} : {v}")
    print(f"\n── Best 5-fold CV : {grid_search.best_score_:.2%}")

    return grid_search.best_estimator_, grid_search, results_df


# =============================================================================
# SECTION 8 — LOOCV EVALUATION
# =============================================================================

def evaluate_with_loocv(model, X, y) -> np.ndarray:
    """LOOCV honest accuracy estimate for small datasets."""

    print("\n── Leave-One-Out CV ──────────────────────────────────────────")
    loo_scores = cross_val_score(
        model, X, y, cv=LeaveOneOut(), scoring="accuracy"
    )
    print(f"  Rounds   : {len(loo_scores)}")
    print(f"  Mean     : {loo_scores.mean():.2%}")
    print(f"  Std      : {loo_scores.std():.2%}")
    print(f"  Min/Max  : {loo_scores.min():.2%} / {loo_scores.max():.2%}")

    mean = loo_scores.mean()
    verdict = (
        "GOOD"          if mean >= 0.75 else
        "MODERATE-GOOD" if mean >= 0.65 else
        "MODERATE"      if mean >= 0.60 else "WEAK"
    )
    print(f"  Verdict  : {verdict}\n")
    return loo_scores


# =============================================================================
# SECTION 9 — FEATURE IMPORTANCE PLOT
# =============================================================================

def plot_feature_importance(model, feature_cols: list,
                             output_folder: str) -> None:
    """Saves feature importance bar chart to models/feature_importance.png."""

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [feature_cols[i] for i in indices]
    sorted_imp  = importances[indices]

    # Colour-code by feature group for easy reading
    group_colours = {
        "june_rainfall_mm"      : "#2d7a4f",
        "june_departure_pct"    : "#2d7a4f",
        "june_vs_district_mean" : "#2d7a4f",
        "june_was_above_normal" : "#2d7a4f",
        "june_rolling_7d_avg"   : "#1a5fa8",
        "june_cumulative_rain"  : "#1a5fa8",
        "june_rain_lag_1d"      : "#1a5fa8",
        "june_rain_lag_7d"      : "#1a5fa8",
        "june_dry_streak"       : "#1a5fa8",
        "june_spi_30d"          : "#7b2d8b",
        "enso_code"             : "#7b2d8b",
        "susm_may_mean"         : "#c07a00",
        "susm_may_max"          : "#c07a00",
        "temp_june_mean"        : "#c07a00",
        "temp_june_stress_days" : "#c07a00",
    }
    default_colour = "#555555"

    colors = [group_colours.get(f, default_colour) for f in sorted_feat]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    bars = ax.barh(range(len(sorted_feat)), sorted_imp,
                   color=colors, height=0.65,
                   edgecolor="white", linewidth=0.4)

    ax.set_yticks(range(len(sorted_feat)))
    ax.set_yticklabels(sorted_feat, fontsize=8.5)
    ax.set_xlabel("Feature Importance (Mean Gini Reduction)",
                  fontsize=9, color="#555")
    ax.set_title("Feature Importances — v4 Climate Context Model",
                 fontsize=11, fontweight="600", color="#1a1a2e", pad=10)
    ax.invert_yaxis()

    for bar, val in zip(bars, sorted_imp):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7.5, color="#333")

    patches = [
        mpatches.Patch(color="#2d7a4f", label="Core June signal"),
        mpatches.Patch(color="#1a5fa8", label="Temporal pattern"),
        mpatches.Patch(color="#7b2d8b", label="Climate context (SPI/ENSO)"),
        mpatches.Patch(color="#c07a00", label="GEE: soil moisture & temp"),
        mpatches.Patch(color="#555555", label="District context & encoding"),
    ]
    ax.legend(handles=patches, fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#ddd")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(0, max(sorted_imp) * 1.22)

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#fafafa")
    plt.close()
    print(f"Feature importance chart → {save_path}")

    print("\n── Feature Importances (ranked) ──────────────────────────────")
    for feat, imp in zip(sorted_feat, sorted_imp):
        bar_str = "█" * int(imp * 50)
        print(f"  {feat:<30} {bar_str:<18}  {imp:.4f}")
    print()


# =============================================================================
# SECTION 10 — SAVE ARTIFACTS
# =============================================================================

def save_model_artifacts(model, le_district, le_state,
                          loo_scores, best_params,
                          grid_results, output_folder) -> None:
    """Saves model + metadata + GridSearch log to models/ folder."""

    os.makedirs(output_folder, exist_ok=True)

    model_path = os.path.join(output_folder, "rainfall_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved    → {model_path}")

    metadata = {
        "feature_columns"        : get_feature_columns(),
        "label_encoder_district" : le_district,
        "label_encoder_state"    : le_state,
        "above_normal_threshold" : ABOVE_NORMAL_THRESHOLD,
        "predictor_month"        : PREDICTOR_MONTH,
        "target_season"          : TARGET_SEASON,
        "cv_mean_accuracy"       : float(loo_scores.mean()),
        "cv_std_accuracy"        : float(loo_scores.std()),
        "gridsearch_best_params" : best_params,
        "gridsearch_5fold_score" : float(loo_scores.mean()),
        "known_districts"        : list(le_district.classes_),
        "known_states"           : list(le_state.classes_),
    }

    metadata_path = os.path.join(output_folder, "model_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved → {metadata_path}")

    gs_path = os.path.join(output_folder, "gridsearch_results.csv")
    grid_results.to_csv(gs_path, index=False)
    print(f"GridSearch log → {gs_path}\n")


# =============================================================================
# SECTION 11 — MAIN ORCHESTRATOR
# =============================================================================

def run_trainer():

    print("=" * 65)
    print("  RAINFALL PREDICTION ENGINE v4 — TRAINING START")
    print("=" * 65 + "\n")

    # 1. Load data
    seasonal_df, monthly_df, daily_df = load_data()

    # 2. Build feature matrix
    features_df, le_district, le_state = build_feature_matrix(
        seasonal_df, monthly_df, daily_df
    )

    FEATURE_COLS = get_feature_columns()
    X = features_df[FEATURE_COLS].values
    y = features_df["target_above_normal"].values

    print(f"Training: {X.shape[0]} samples × {X.shape[1]} features\n")

    # 3. GridSearchCV
    best_model, grid_search, grid_results = run_gridsearch(X, y)

    # 4. LOOCV
    loo_scores = evaluate_with_loocv(best_model, X, y)

    # 5. Feature importance
    plot_feature_importance(best_model, FEATURE_COLS, MODEL_OUTPUT_FOLDER)

    # 6. Save
    print("Saving artifacts...")
    save_model_artifacts(
        model=best_model, le_district=le_district, le_state=le_state,
        loo_scores=loo_scores, best_params=grid_search.best_params_,
        grid_results=grid_results, output_folder=MODEL_OUTPUT_FOLDER
    )

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Features    : {len(FEATURE_COLS)} total")
    print(f"  Best params : {grid_search.best_params_}")
    print(f"  5-fold CV   : {grid_search.best_score_:.2%}")
    print(f"  LOOCV       : {loo_scores.mean():.2%}")
    print("  Check models/feature_importance.png")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_trainer()