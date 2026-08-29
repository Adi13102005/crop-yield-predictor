"""
Trains and compares models for crop yield prediction.

Instead of one global model across all crops, this trains a SEPARATE
model per crop. Reason: crop yields differ by an order of magnitude
(Sugarcane ~65 t/ha vs Rice/Maize/Soybean ~2-4 t/ha), so a single
shared model lets `crop` dominate every prediction's explanation
(SHAP/feature importance), drowning out the actually useful signals
like rainfall, NPK, and pH. Per-crop models fix that: each model only
ever sees one crop's yield scale, so its explanations reflect real
agronomic drivers for that crop.

Run: python models/train.py
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/crop_data.csv"
MODEL_OUT = "models/yield_model.joblib"

# NOTE: "crop" is no longer a model input feature — it now determines
# WHICH model to use, rather than being one-hot-encoded alongside the rest.
CATEGORICAL = ["region", "soil_type"]
NUMERIC = [
    "rainfall_mm", "temperature_c", "humidity_pct",
    "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha",
    "ph", "area_hectares",
]
TARGET = "yield_tonnes_per_hectare"


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def train_for_crop(df_crop):
    X = df_crop[NUMERIC + CATEGORICAL]
    y = df_crop[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    best = None
    for name, model in candidates.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        if best is None or r2 > best["metrics"]["R2"]:
            best = {
                "pipeline": pipe,
                "model_name": name,
                "metrics": {"MAE": round(mae, 3), "R2": round(r2, 3)},
            }

    return best


def main():
    df = pd.read_csv(DATA_PATH)
    crops = sorted(df["crop"].unique().tolist())

    per_crop_models = {}
    for crop in crops:
        df_crop = df[df["crop"] == crop]
        result = train_for_crop(df_crop)
        per_crop_models[crop] = result
        print(
            f"{crop:12s}  best={result['model_name']:18s}  "
            f"MAE={result['metrics']['MAE']}  R2={result['metrics']['R2']}"
        )

    joblib.dump(
        {
            "models": per_crop_models,
            "numeric_features": NUMERIC,
            "categorical_features": CATEGORICAL,
            "crops": crops,
            "regions": sorted(df["region"].unique().tolist()),
            "soil_types": sorted(df["soil_type"].unique().tolist()),
        },
        MODEL_OUT,
    )
    print(f"\nSaved {len(per_crop_models)} per-crop models -> {MODEL_OUT}")


if __name__ == "__main__":
    main()