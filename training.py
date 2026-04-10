# =============================================
# CRIME RISK PREDICTION MODEL TRAINING
# With temporal decay + recency boosting
# Exports model as JSON for browser inference
# =============================================
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATASET_PATH = "crime_dataset.csv"
MODEL_PKL_PATH = "crime_model.pkl"
MODEL_JSON_PATH = "crime_model.json"

# =============================================
# TEMPORAL CONFIG  ← tweak these as needed
# =============================================
RECENT_WINDOW_DAYS   = 60    # "last 2 months"
RECENCY_BOOST_MAX    = 1.40  # crimes in window → up to +40% risk
DECAY_FLOOR          = 0.40  # very old crimes floor at 40% of original risk
DECAY_HALF_LIFE_DAYS = 90    # risk halves every 90 days beyond the window
FREQ_RADIUS_DEG      = 0.02  # ~2 km box for frequency count

print("Loading dataset...")
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")
print("Total rows:", len(data))
print("Columns:", ", ".join(data.columns))

# =============================================
# STEP 1 - NORMALISE THE DATASET SCHEMA
# =============================================
if "Police_Presence" in data.columns and "Police_Awareness" not in data.columns:
    data["Police_Awareness"] = data["Police_Presence"]

if "Time" in data.columns:
    data["Time"] = pd.to_datetime(data["Time"], format="%H:%M", errors="coerce")
    data["Hour"] = data["Time"].dt.hour
else:
    data["Hour"] = 12

# Parse Date → derive Month AND Days_Since_Crime
if "Date" in data.columns:
    parsed_dates = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    data["Month"] = parsed_dates.dt.month.fillna(0).astype(int)

    # Use the most recent date in dataset as "today" for reproducibility
    # In production you can replace this with: pd.Timestamp.today()
    reference_date = parsed_dates.max()
    data["Days_Since_Crime"] = (reference_date - parsed_dates).dt.days.fillna(999).astype(int)
else:
    data["Month"] = 0
    data["Days_Since_Crime"] = 999  # unknown → treat as old

level_map = {"Low": 1, "Medium": 2, "High": 3}
lighting_map = {"Poor": 1, "Moderate": 2, "Good": 3}

for col in ["Crowd_Presence", "Police_Presence", "Police_Awareness"]:
    if col in data.columns:
        data[col] = data[col].map(level_map)

if "Lighting" in data.columns:
    data["Lighting"] = data["Lighting"].map(lighting_map)

if "Risk_Score" not in data.columns:
    raise ValueError("Risk_Score column is required for training.")
data["Risk_Score"] = pd.to_numeric(data["Risk_Score"], errors="coerce")

# =============================================
# STEP 2 - TEMPORAL RISK ADJUSTMENT
# =============================================
print("\nApplying temporal risk adjustments...")

def temporal_multiplier(days: pd.Series) -> pd.Series:
    """
    Recent (≤ RECENT_WINDOW_DAYS):  boost up to RECENCY_BOOST_MAX
    Old (> RECENT_WINDOW_DAYS):     decay toward DECAY_FLOOR using half-life
    """
    multiplier = pd.Series(np.ones(len(days)), index=days.index)

    recent_mask = days <= RECENT_WINDOW_DAYS
    old_mask    = ~recent_mask

    # Boost: linearly scale from 1.0 (at window edge) → RECENCY_BOOST_MAX (at day 0)
    if recent_mask.any():
        t = 1.0 - (days[recent_mask] / RECENT_WINDOW_DAYS)   # 1.0 = today, 0.0 = edge
        multiplier[recent_mask] = 1.0 + t * (RECENCY_BOOST_MAX - 1.0)

    # Decay: exponential beyond the window, floored at DECAY_FLOOR
    if old_mask.any():
        extra_days = days[old_mask] - RECENT_WINDOW_DAYS
        decay = np.exp(-np.log(2) * extra_days / DECAY_HALF_LIFE_DAYS)
        multiplier[old_mask] = np.maximum(DECAY_FLOOR, decay)

    return multiplier

mult = temporal_multiplier(data["Days_Since_Crime"])
data["Risk_Score"] = np.clip(data["Risk_Score"] * mult, 0, 100)

# =============================================
# STEP 3 - CRIME FREQUENCY IN LAST 2 MONTHS
# =============================================
print("Computing per-area crime frequency (last 2 months)...")

def compute_local_frequency(df: pd.DataFrame) -> pd.Series:
    """
    For each row, count crimes within FREQ_RADIUS_DEG lat/lng box
    AND within RECENT_WINDOW_DAYS. Normalised to 0-10 range.
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.Series(0, index=df.index)

    lats  = df["Latitude"].values
    lngs  = df["Longitude"].values
    days  = df["Days_Since_Crime"].values
    freq  = np.zeros(len(df), dtype=float)

    recent = days <= RECENT_WINDOW_DAYS

    for i in range(len(df)):
        in_box = (
            (np.abs(lats - lats[i]) <= FREQ_RADIUS_DEG) &
            (np.abs(lngs - lngs[i]) <= FREQ_RADIUS_DEG) &
            recent
        )
        freq[i] = in_box.sum()

    # Normalise: 0-10 scale (10 = 99th percentile count)
    p99 = np.percentile(freq, 99) if freq.max() > 0 else 1.0
    return pd.Series(np.clip(freq / max(p99, 1) * 10, 0, 10).round(2), index=df.index)

data["crime_frequency_2m"] = compute_local_frequency(data)
print(f"  Frequency range: {data['crime_frequency_2m'].min():.1f} – {data['crime_frequency_2m'].max():.1f}")

# =============================================
# STEP 4 - ENCODE CATEGORICAL COLUMNS
# =============================================
label_maps = {}
for col in ["Crime_Type", "Risk_Level"]:
    if col in data.columns:
        cat = data[col].astype("category")
        label_maps[col] = {str(v): int(k) for k, v in enumerate(cat.cat.categories)}
        data[col] = cat.cat.codes

# =============================================
# STEP 5 - SELECT USABLE FEATURES
# =============================================
feature_candidates = [
    "Crime_Type",
    "Crowd_Presence",
    "Lighting",
    "Police_Presence",
    "Latitude",
    "Longitude",
    "Hour",
    "Month",
    "Days_Since_Crime",       # ← NEW: how old is this crime record
    "crime_frequency_2m",     # ← NEW: how busy is this area recently
]
features = [col for col in feature_candidates if col in data.columns]
target = "Risk_Score"

data = data.dropna(subset=features + [target]).copy()
print("Rows after cleaning:", len(data))
if data.empty:
    raise ValueError("No rows remain after cleaning. Check the dataset values.")
if not features:
    raise ValueError("No usable feature columns were found in the dataset.")

print("Training features:", ", ".join(features))

# =============================================
# STEP 6 - TRAIN
# =============================================
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining model...")
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12,
    min_samples_leaf=2,
)
model.fit(X_train, y_train)

preds = np.clip(model.predict(X_test), 0, 100)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2:  {r2_score(y_test, preds):.3f}")

joblib.dump(model, MODEL_PKL_PATH)
print(f"Saved {MODEL_PKL_PATH}")

# =============================================
# STEP 7 - EXPORT JSON FOR BROWSER
# =============================================
print("\nExporting model to JSON for browser...")


def export_tree(tree):
    tree_data = tree.tree_

    def recurse(node):
        if tree_data.children_left[node] == -1:
            return {"v": round(float(tree_data.value[node][0][0]), 2)}
        return {
            "f": int(tree_data.feature[node]),
            "t": round(float(tree_data.threshold[node]), 4),
            "l": recurse(tree_data.children_left[node]),
            "r": recurse(tree_data.children_right[node]),
        }

    return recurse(0)


trees = [export_tree(est) for est in model.estimators_]
ai_preds = np.clip(model.predict(data[features]), 0, 100)

point_preds = []
for lat, lng, ai, raw, freq, days in zip(
    data["Latitude"].values,
    data["Longitude"].values,
    ai_preds,
    data["Risk_Score"].values,
    data["crime_frequency_2m"].values,
    data["Days_Since_Crime"].values,
):
    point_preds.append({
        "lat":   round(float(lat), 5),
        "lng":   round(float(lng), 5),
        "ai":    round(float(ai), 1),
        "raw":   round(float(raw), 1),
        "freq":  round(float(freq), 1),   # recent frequency score
        "days":  int(days),               # age of this crime record
    })

feature_defaults = {
    "Crime_Type":          0,
    "Crowd_Presence":      2,
    "Lighting":            2,
    "Police_Presence":     2,
    "Latitude":            round(float(data["Latitude"].median()), 6) if "Latitude" in data.columns else 0,
    "Longitude":           round(float(data["Longitude"].median()), 6) if "Longitude" in data.columns else 0,
    "Hour":                12,
    "Month":               3,
    "Days_Since_Crime":    30,   # default: assume recent-ish
    "crime_frequency_2m":  1.0,  # default: low activity
}

model_json = {
    "features":            features,
    "label_maps":          label_maps,
    "level_map":           level_map,
    "lighting_map":        lighting_map,
    "n_trees":             len(trees),
    "trees":               trees,
    "point_predictions":   point_preds,
    "feature_defaults":    {k: v for k, v in feature_defaults.items() if k in features},
    "target":              target,
    # Expose config so frontend can replicate the logic
    "temporal_config": {
        "recent_window_days":   RECENT_WINDOW_DAYS,
        "recency_boost_max":    RECENCY_BOOST_MAX,
        "decay_floor":          DECAY_FLOOR,
        "decay_half_life_days": DECAY_HALF_LIFE_DAYS,
        "freq_radius_deg":      FREQ_RADIUS_DEG,
    },
}

with open(MODEL_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(model_json, f, separators=(",", ":"))

size_kb = len(json.dumps(model_json, separators=(",", ":"))) / 1024
print(f"Saved {MODEL_JSON_PATH} ({size_kb:.0f} KB)")
print("\nTRAINING COMPLETED SUCCESSFULLY")
print(f"Files created: {MODEL_PKL_PATH}, {MODEL_JSON_PATH}")