"""
Generates a synthetic but realistic crop yield dataset.
Swap this out later with a real dataset (e.g. from data.gov.in, Kaggle
'Crop Recommendation Dataset', or ICRISAT) — keep the same column names
and the rest of the pipeline (train.py, api/) will work unchanged.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

crops = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Soybean"]
regions = ["North", "South", "East", "West", "Central"]
soil_types = ["Alluvial", "Black", "Red", "Laterite", "Sandy"]

df = pd.DataFrame({
    "crop": np.random.choice(crops, N),
    "region": np.random.choice(regions, N),
    "soil_type": np.random.choice(soil_types, N),
    "rainfall_mm": np.random.normal(1000, 300, N).clip(100, 2500),
    "temperature_c": np.random.normal(27, 5, N).clip(10, 45),
    "humidity_pct": np.random.normal(65, 15, N).clip(20, 100),
    "nitrogen_kg_ha": np.random.normal(80, 25, N).clip(0, 200),
    "phosphorus_kg_ha": np.random.normal(40, 15, N).clip(0, 150),
    "potassium_kg_ha": np.random.normal(40, 15, N).clip(0, 150),
    "ph": np.random.normal(6.5, 0.8, N).clip(4.5, 9.0),
    "area_hectares": np.random.exponential(2, N).clip(0.1, 20),
})

# Inject a semi-realistic relationship so the model has real signal to learn,
# not pure noise. Yield responds positively to N/P/K and rainfall up to a
# point, then plateaus/declines (diminishing returns + waterlogging risk).
def compute_yield(row):
    base = {
        "Rice": 3.5, "Wheat": 3.0, "Maize": 2.8,
        "Sugarcane": 65.0, "Cotton": 1.8, "Soybean": 2.2,
    }[row["crop"]]

    rain_factor = 1 - abs(row["rainfall_mm"] - 1100) / 2500
    npk_factor = (row["nitrogen_kg_ha"] * 0.4 + row["phosphorus_kg_ha"] * 0.3
                  + row["potassium_kg_ha"] * 0.3) / 100
    ph_factor = 1 - abs(row["ph"] - 6.5) / 5
    temp_factor = 1 - abs(row["temperature_c"] - 26) / 30

    noise = np.random.normal(1, 0.15)

    yield_t_ha = base * (0.5 + 0.3 * rain_factor + 0.4 * npk_factor
                          + 0.2 * ph_factor + 0.2 * temp_factor) * noise
    return max(yield_t_ha, 0.1)

df["yield_tonnes_per_hectare"] = df.apply(compute_yield, axis=1).round(2)

df.to_csv("data/crop_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/crop_data.csv")
print(df.head())
print("\nYield stats:\n", df["yield_tonnes_per_hectare"].describe())
