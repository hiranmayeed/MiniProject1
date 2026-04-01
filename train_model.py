"""
=============================================================================
RAINFALL ANALYSIS & PREDICTION FRAMEWORK
Step 3: Prediction Engine — ML Trainer
=============================================================================
Purpose : Train a Random Forest classifier that predicts whether the
          remaining monsoon season will be 'Above-Normal' based on
          June's rainfall data for a given district.
Input   : data/processed/03_seasonal_with_departure.csv  (from Step 2)
Output  : models/rainfall_model.pkl   — the saved trained model
          models/model_metadata.pkl   — label encoders + feature info
Author  : (your name)
=============================================================================

BEGINNER CONCEPT — What is a Random Forest?
────────────────────────────────────────────
Think of it as 100 people (decision trees) each independently looking at
your rainfall data and voting "Above-Normal" or "Not Above-Normal".
The majority vote becomes the final prediction.
It's robust, works well with small datasets (like our 8 years), and
doesn't need the data to follow any particular statistical distribution.
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle                          # Used to save/load the trained model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")      # Suppress minor sklearn warnings


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Path to the seasonal output file produced by rainfall_pipeline.py (Step 2)
SEASONAL_DATA_PATH = "data/processed/03_seasonal_with_departure.csv"

# Folder where we'll save the trained model
MODEL_OUTPUT_FOLDER = "models/"

# The month number we use as our early-season predictor signal
# June = 6 (first month of Kharif; available before the full season ends)
PREDICTOR_MONTH = 6

# The season we're predicting for
TARGET_SEASON = "Kharif"

# The departure % threshold that defines "Above-Normal" (from IMD standard)
ABOVE_NORMAL_THRESHOLD = 5.0


# =============================================================================
# SECTION 2 — LOAD AND VALIDATE THE CLEANED DATA
# =============================================================================

def load_seasonal_data(path: str) -> pd.DataFrame:
    """
    Loads the seasonal CSV produced by the pipeline in Step 2.
    Validates that required columns are present before proceeding.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'.\n"
            "Make sure you've run rainfall_pipeline.py first (Step 2)."
        )

    df = pd.read_csv(path)

    # Columns we absolutely need for training
    required_columns = [
        "district_name", "state_name", "year",
        "season", "total_rainfall_mm", "departure_pct"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in seasonal data: {missing}")

    print(f"Loaded seasonal data: {len(df):,} rows")
    print(f"Districts: {df['district_name'].nunique()} | "
          f"Years: {sorted(df['year'].unique())}\n")

    return df


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING
# =============================================================================

"""
BEGINNER CONCEPT — What is Feature Engineering?
────────────────────────────────────────────────
A "feature" is any input variable the model uses to make its prediction.
Raw data (like total_rainfall_mm) is rarely in the right shape for ML.
Feature engineering = transforming raw data into signals the model can learn from.

Our prediction question:
  "Given only June's rainfall for a district, can we predict whether
   the full Kharif season (June–September) will be Above-Normal?"

Why only June?
  In a real deployment, when June ends, the farmer needs an EARLY warning
  for July–September. We can only use data that exists at prediction time.
"""

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw seasonal data into a feature matrix ready for ML.

    Each row in the output represents ONE district in ONE year, with:
      - Features (X): what we KNOW at prediction time (June's rainfall)
      - Target  (y): what we want to PREDICT (was the full season Above-Normal?)
    """

    # ── Step 3a: Filter to Kharif season only ─────────────────────────────
    # We only predict for Kharif (monsoon) season — that's what matters for
    # farmers' primary sowing decisions.
    kharif_df = df[df["season"] == TARGET_SEASON].copy()
    print(f"Kharif rows: {len(kharif_df):,}")

    # ── Step 3b: Create the TARGET variable (what we predict) ─────────────
    # Binary classification: 1 = Above-Normal season, 0 = Not Above-Normal
    # This becomes the "answer key" the model learns from.
    kharif_df["target_above_normal"] = (
        kharif_df["departure_pct"] > ABOVE_NORMAL_THRESHOLD
    ).astype(int)

    above_count = kharif_df["target_above_normal"].sum()
    print(f"Above-Normal seasons in data: {above_count} / {len(kharif_df)} "
          f"({above_count/len(kharif_df)*100:.1f}%)\n")

    # ── Step 3c: Build June-only predictor features ────────────────────────
    # We need a separate monthly DataFrame to extract June figures.
    # IMPORTANT: This assumes you also have 02_monthly_with_departure.csv
    monthly_path = "data/processed/02_monthly_with_departure.csv"

    if not os.path.exists(monthly_path):
        raise FileNotFoundError(
            f"Could not find '{monthly_path}'.\n"
            "Ensure rainfall_pipeline.py has been run successfully."
        )

    monthly_df = pd.read_csv(monthly_path)

    # Filter to June rows only (month == 6)
    june_df = monthly_df[monthly_df["month"] == PREDICTOR_MONTH][
        ["district_name", "year", "total_rainfall_mm", "departure_pct"]
    ].copy()

    # Rename columns so we know they're June-specific features
    june_df = june_df.rename(columns={
        "total_rainfall_mm": "june_rainfall_mm",
        "departure_pct":     "june_departure_pct"
    })

    # ── Step 3d: Compute historical district-level features ────────────────
    # These give the model context about each district's typical behaviour.
    # "Is 150mm high for this district, or is that normal for them?"

    district_stats = (
        kharif_df.groupby("district_name")["total_rainfall_mm"]
        .agg(
            district_mean_mm   = "mean",   # avg seasonal rainfall over 8 years
            district_std_mm    = "std",    # how variable rainfall is
            district_cv        = lambda x: x.std() / x.mean()  # coefficient of variation
        )
        .reset_index()
    )

    # ── Step 3e: Merge everything into one feature matrix ──────────────────
    # Start with Kharif seasonal rows (one per district-year)
    features_df = kharif_df[
        ["district_name", "state_name", "year", "target_above_normal"]
    ].copy()

    # Merge in June rainfall features (left join keeps all kharif rows)
    features_df = features_df.merge(
        june_df, on=["district_name", "year"], how="left"
    )

    # Merge in district historical context
    features_df = features_df.merge(
        district_stats, on="district_name", how="left"
    )

    # ── Step 3f: Add derived features ──────────────────────────────────────
    # "How does this June compare to this district's historical average?"
    # This normalises June rainfall relative to each district's baseline.
    features_df["june_vs_district_mean"] = (
        features_df["june_rainfall_mm"] / features_df["district_mean_mm"]
    ).round(4)

    # Flag: was this June itself already above-normal?
    features_df["june_was_above_normal"] = (
        features_df["june_departure_pct"] > ABOVE_NORMAL_THRESHOLD
    ).astype(int)

    # ── Step 3g: Encode district and state as numbers ─────────────────────
    # Random Forest needs numbers, not text. LabelEncoder converts
    # "Warangal" → 42, "Hyderabad" → 17, etc.
    # We save the encoder so the dashboard can reverse it later.
    le_district = LabelEncoder()
    le_state    = LabelEncoder()

    features_df["district_encoded"] = le_district.fit_transform(
        features_df["district_name"]
    )
    features_df["state_encoded"] = le_state.fit_transform(
        features_df["state_name"]
    )

    # Drop rows where June data is missing (can't predict without it)
    before_drop = len(features_df)
    features_df = features_df.dropna(subset=[
        "june_rainfall_mm", "june_departure_pct"
    ])
    dropped = before_drop - len(features_df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing June data.\n")

    print(f"Feature matrix shape: {features_df.shape}")
    print(f"Features built for {features_df['district_name'].nunique()} districts\n")

    return features_df, le_district, le_state


# =============================================================================
# SECTION 4 — TRAIN / TEST SPLIT
# =============================================================================

"""
BEGINNER CONCEPT — Why Split Data at All?
──────────────────────────────────────────
If you train the model on ALL your data and then test it on the same data,
it's like giving students the exam questions in advance — of course they'll
score 100%. That tells you nothing about real-world performance.

Splitting means: train on SOME years → test on YEARS THE MODEL NEVER SAW.

The Challenge with 8 Years of Data:
  8 years is a very small dataset for ML. Standard 80/20 splits would give
  us only ~1–2 test years, which is statistically unreliable.

Our Solution — Leave-One-Out Cross Validation (LOOCV):
  We train 8 separate models, each time leaving out 1 year as the test set.
  This uses every year as a test year exactly once, giving us the most
  honest performance estimate possible with limited data.
  
  Round 1: Train on 2017-2023 → Test on 2016
  Round 2: Train on 2016,2018-2023 → Test on 2017
  ... and so on for all 8 years.
"""

def get_feature_columns() -> list:
    """
    Returns the list of column names used as model inputs (features X).
    Centralised here so training and prediction always use identical features.
    """
    return [
        "june_rainfall_mm",       # Raw June total
        "june_departure_pct",     # How anomalous was June?
        "june_vs_district_mean",  # June relative to district's 8yr average
        "june_was_above_normal",  # Binary: was June itself above normal?
        "district_mean_mm",       # District's historical average season
        "district_std_mm",        # District's historical variability
        "district_cv",            # Coefficient of variation (consistency)
        "district_encoded",       # District identity (as number)
        "state_encoded",          # State identity (as number)
    ]


def split_and_evaluate(features_df: pd.DataFrame) -> tuple:
    """
    Performs Leave-One-Out Cross Validation to evaluate model performance,
    then trains a FINAL model on ALL available data for production use.

    Returns:
        final_model  — trained on 100% of data, ready for deployment
        cv_scores    — array of accuracy scores from each LOOCV round
    """

    FEATURE_COLS = get_feature_columns()
    TARGET_COL   = "target_above_normal"

    X = features_df[FEATURE_COLS].values   # Feature matrix (inputs)
    y = features_df[TARGET_COL].values     # Target vector (answers)

    # ── Random Forest configuration ────────────────────────────────────────
    # n_estimators=100 : use 100 decision trees (more = more stable, slower)
    # max_depth=4      : limit tree depth to prevent overfitting on small data
    # random_state=42  : fixed seed so results are reproducible every run
    # class_weight='balanced' : compensates if one class (above/not) is rarer
    model = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 4,
        random_state  = 42,
        class_weight  = "balanced"
    )

    # ── Leave-One-Out Cross Validation ─────────────────────────────────────
    print("Running Leave-One-Out Cross Validation...")
    print("(Each year is held out once as the test set)\n")

    loo = LeaveOneOut()

    cv_scores = cross_val_score(
        model, X, y,
        cv      = loo,
        scoring = "accuracy"
    )

    print(f"LOOCV Results across {len(cv_scores)} rounds:")
    print(f"  Accuracy per round : {[round(s,2) for s in cv_scores]}")
    print(f"  Mean accuracy      : {cv_scores.mean():.2%}")
    print(f"  Std deviation      : {cv_scores.std():.2%}")

    # ── Interpretation guide ───────────────────────────────────────────────
    mean_acc = cv_scores.mean()
    if mean_acc >= 0.75:
        verdict = "GOOD — model has useful predictive signal"
    elif mean_acc >= 0.60:
        verdict = "MODERATE — better than random, but interpret with caution"
    else:
        verdict = "WEAK — model struggles; more data or features needed"
    print(f"  Verdict            : {verdict}\n")

    # ── Confusion Matrix via LOOCV predictions ─────────────────────────────
    # cross_val_predict collects each fold's held-out prediction,
    # giving us a full y_pred aligned with y for the confusion matrix.
    print("Generating confusion matrix from LOOCV predictions...")
    y_pred_loo = cross_val_predict(model, X, y, cv=loo)

    cm = confusion_matrix(y, y_pred_loo)
    print("\nConfusion Matrix (LOOCV):")
    print(f"  Labels: 0 = Not Above-Normal | 1 = Above-Normal")
    print(cm)
    print()
    print(classification_report(
        y, y_pred_loo,
        target_names=["Not Above-Normal", "Above-Normal"]
    ))

    # ── Plot and save confusion matrix ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Above-Normal", "Above-Normal"],
        yticklabels=["Not Above-Normal", "Above-Normal"],
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Rainfall Prediction (LOOCV)", fontsize=13)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    cm_path = "models/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved → {cm_path}\n")

    # ── Train FINAL model on ALL data ──────────────────────────────────────
    # LOOCV was only for evaluation. The actual deployed model trains on
    # everything — more data = better generalisation for new predictions.
    print("Training final model on full dataset...")
    model.fit(X, y)
    print("Final model trained.\n")

    # ── Feature importance — what does the model rely on most? ─────────────
    print("Feature importances (higher = more influential):")
    importance_df = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:<30} {bar}  {row['importance']:.3f}")
    print()

    return model, cv_scores, importance_df, y_pred_loo


# =============================================================================
# SECTION 5 — SAVE THE MODEL AND METADATA
# =============================================================================

"""
BEGINNER CONCEPT — What is a .pkl file?
────────────────────────────────────────
pkl = "pickle" — Python's way of freezing any object to disk.
Training a model = teaching it patterns from data (takes compute + time).
Pickling = saving that trained state so you NEVER have to retrain.

In the dashboard (Step 4), we'll just load the .pkl and predict instantly
without touching the training data again.
"""

def save_model_artifacts(model,
                          le_district: LabelEncoder,
                          le_state:    LabelEncoder,
                          importance_df: pd.DataFrame,
                          cv_scores:   np.ndarray,
                          output_folder: str) -> None:
    """
    Saves three files to the models/ folder:
      1. rainfall_model.pkl    — the trained Random Forest
      2. model_metadata.pkl    — encoders + feature list (needed at prediction time)
      3. model_report.csv      — feature importances for reference
    """

    os.makedirs(output_folder, exist_ok=True)

    # ── 1. Save the trained model ──────────────────────────────────────────
    model_path = os.path.join(output_folder, "rainfall_model.pkl")
    with open(model_path, "wb") as f:      # "wb" = write bytes
        pickle.dump(model, f)
    print(f"Model saved       → {model_path}")

    # ── 2. Save metadata (everything the dashboard needs alongside model) ──
    # The dashboard needs the SAME encoders used during training to convert
    # district/state names to the numbers the model expects.
    metadata = {
        "feature_columns"     : get_feature_columns(),
        "label_encoder_district": le_district,
        "label_encoder_state"   : le_state,
        "above_normal_threshold": ABOVE_NORMAL_THRESHOLD,
        "predictor_month"       : PREDICTOR_MONTH,
        "target_season"         : TARGET_SEASON,
        "cv_mean_accuracy"      : cv_scores.mean(),
        "cv_std_accuracy"       : cv_scores.std(),
        "known_districts"       : list(le_district.classes_),
        "known_states"          : list(le_state.classes_),
    }

    metadata_path = os.path.join(output_folder, "model_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved    → {metadata_path}")

    # ── 3. Save feature importance report as CSV ───────────────────────────
    report_path = os.path.join(output_folder, "model_report.csv")
    importance_df.to_csv(report_path, index=False)
    print(f"Report saved      → {report_path}\n")


# =============================================================================
# SECTION 6 — PREDICTION HELPER (used by the dashboard in Step 4)
# =============================================================================

def predict_for_district(district_name:    str,
                          state_name:       str,
                          june_rainfall_mm: float,
                          model_folder:     str = "models/") -> dict:
    """
    Loads the saved model and makes a prediction for a single district.

    This is the function the Streamlit dashboard will call in Step 4.
    It takes human-readable inputs and returns a probability + category.

    Args:
        district_name    : e.g. "Warangal"
        state_name       : e.g. "Telangana"
        june_rainfall_mm : total rainfall recorded in June for this district

    Returns:
        dict with keys: probability, category, confidence, advice
    """

    # ── Load saved model and metadata ─────────────────────────────────────
    model_path    = os.path.join(model_folder, "rainfall_model.pkl")
    metadata_path = os.path.join(model_folder, "model_metadata.pkl")

    with open(model_path,    "rb") as f: model    = pickle.load(f)
    with open(metadata_path, "rb") as f: metadata = pickle.load(f)

    le_district = metadata["label_encoder_district"]
    le_state    = metadata["label_encoder_state"]

    # ── Handle unseen districts gracefully ────────────────────────────────
    # If the dashboard user selects a district not in training data, we
    # can't encode it. Return a clear error rather than crashing.
    if district_name not in metadata["known_districts"]:
        return {
            "error": f"District '{district_name}' was not in training data. "
                     f"Available: {metadata['known_districts'][:5]}..."
        }

    district_enc = le_district.transform([district_name])[0]
    state_enc    = le_state.transform([state_name])[0] \
                   if state_name in metadata["known_states"] else 0

    # ── Build the feature vector for this prediction ───────────────────────
    # We need district historical stats — load them from the seasonal CSV
    seasonal_df = pd.read_csv("data/processed/03_seasonal_with_departure.csv")
    kharif_df   = seasonal_df[seasonal_df["season"] == TARGET_SEASON]
    dist_stats  = kharif_df[kharif_df["district_name"] == district_name][
        "total_rainfall_mm"
    ]

    district_mean = dist_stats.mean() if len(dist_stats) > 0 else 500
    district_std  = dist_stats.std()  if len(dist_stats) > 0 else 100
    district_cv   = district_std / district_mean if district_mean > 0 else 0.2

    june_lpa_path = "data/processed/02_monthly_with_departure.csv"
    monthly_df    = pd.read_csv(june_lpa_path)
    june_lpa_row  = monthly_df[
        (monthly_df["district_name"] == district_name) &
        (monthly_df["month"] == PREDICTOR_MONTH)
    ]["lpa_mm"]
    june_lpa = june_lpa_row.mean() if len(june_lpa_row) > 0 else district_mean / 4

    june_departure_pct   = ((june_rainfall_mm - june_lpa) / june_lpa * 100) \
                           if june_lpa > 0 else 0
    june_vs_district_mean = june_rainfall_mm / district_mean \
                            if district_mean > 0 else 1.0
    june_was_above_normal = int(june_departure_pct > ABOVE_NORMAL_THRESHOLD)

    # Assemble into the same order as training features
    feature_vector = [[
        june_rainfall_mm,
        june_departure_pct,
        june_vs_district_mean,
        june_was_above_normal,
        district_mean,
        district_std,
        district_cv,
        district_enc,
        state_enc,
    ]]

    # ── Make prediction ────────────────────────────────────────────────────
    # predict_proba returns [[prob_class_0, prob_class_1]]
    # class 1 = Above-Normal, so we take index [0][1]
    probability = model.predict_proba(feature_vector)[0][1]

    # ── Map probability to IMD category and farmer advice ─────────────────
    if probability >= 0.70:
        category = "Above Normal"
        colour   = "green"
        advice   = (
            "High confidence of above-normal monsoon. "
            "Proceed with full Kharif sowing plan. "
            "Ensure drainage channels are clear to handle surplus water."
        )
    elif probability >= 0.50:
        category = "Normal to Above Normal"
        colour   = "blue"
        advice   = (
            "Moderate confidence of good monsoon. "
            "Sow primary crops as planned but keep 20% area as buffer. "
            "Monitor weekly rainfall and adjust irrigation accordingly."
        )
    elif probability >= 0.35:
        category = "Normal"
        colour   = "orange"
        advice   = (
            "Season likely to be near average. "
            "Follow standard sowing schedule. "
            "Keep drought-tolerant variety seeds as backup."
        )
    else:
        category = "Below Normal"
        colour   = "red"
        advice   = (
            "Risk of below-normal monsoon. "
            "Consider drought-resistant crop varieties. "
            "Prioritise water conservation. "
            "Consult local Krishi Vigyan Kendra before full sowing."
        )

    return {
        "district"          : district_name,
        "state"             : state_name,
        "june_rainfall_mm"  : june_rainfall_mm,
        "june_departure_pct": round(june_departure_pct, 2),
        "probability"       : round(probability * 100, 1),
        "category"          : category,
        "colour"            : colour,
        "advice"            : advice,
        "model_accuracy"    : round(metadata["cv_mean_accuracy"] * 100, 1),
    }


# =============================================================================
# SECTION 7 — MAIN ORCHESTRATOR
# =============================================================================

def run_trainer():
    """
    Master function — runs the full training pipeline end to end.
    """

    print("=" * 65)
    print("  RAINFALL PREDICTION ENGINE — TRAINING START")
    print("=" * 65 + "\n")

    # Step 1: Load cleaned seasonal data from pipeline output
    seasonal_df = load_seasonal_data(SEASONAL_DATA_PATH)

    # Step 2: Engineer features and build the ML-ready matrix
    features_df, le_district, le_state = build_feature_matrix(seasonal_df)

    # Step 3: Evaluate with LOOCV, train final model on full data
    model, cv_scores, importance_df, y_pred_loo = split_and_evaluate(features_df)

    # Step 4: Save model, encoders, and report to disk
    print("Saving model artifacts...")
    save_model_artifacts(
        model, le_district, le_state,
        importance_df, cv_scores,
        MODEL_OUTPUT_FOLDER
    )

    # Step 5: Quick smoke test — predict for a sample district
    print("─" * 65)
    print("SMOKE TEST — sample prediction")
    print("─" * 65)

    sample_district = features_df["district_name"].iloc[0]
    sample_state    = features_df["state_name"].iloc[0]
    sample_june_mm  = features_df["june_rainfall_mm"].iloc[0]

    result = predict_for_district(
        district_name    = sample_district,
        state_name       = sample_state,
        june_rainfall_mm = sample_june_mm
    )

    print(f"District          : {result['district']}, {result['state']}")
    print(f"June rainfall     : {result['june_rainfall_mm']} mm "
          f"(departure: {result['june_departure_pct']}%)")
    print(f"Prediction        : {result['category']}")
    print(f"Probability       : {result['probability']}% chance of Above-Normal")
    print(f"Farmer advice     : {result['advice']}")
    print(f"Model accuracy    : {result['model_accuracy']}% (LOOCV)")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE — check models/ folder for outputs")
    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_trainer()