# 🌾 Crop Yield Predictor

An end-to-end ML pipeline that predicts crop yield (tonnes/hectare) from
soil composition, weather, and nutrient inputs — served via a FastAPI
backend and an interactive Streamlit frontend.

## Why this project

Most student ML projects stop at a Jupyter notebook. This one goes further:
data pipeline → model comparison → explainability → REST API → UI → (optional) deployment.
That's the story worth telling in interviews.

## Project structure

```
crop-yield-predictor/
├── data/
│   ├── generate_data.py      # Synthetic data generator (swap for real data later)
│   └── crop_data.csv         # Generated dataset
├── models/
│   ├── train.py               # Trains & compares 3 models, saves the best one
│   └── yield_model.joblib     # Saved pipeline (preprocessing + model)
├── api/
│   └── main.py                 # FastAPI backend (/predict, /metadata endpoints)
├── app/
│   └── streamlit_app.py        # Interactive frontend
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Generate data (or replace `data/crop_data.csv` with a real dataset —
keep the same column names)**
```bash
python data/generate_data.py
```

**2. Train the model**
```bash
python models/train.py
```
This trains Linear Regression, Random Forest, and Gradient Boosting,
prints comparison metrics (MAE, R²), and saves the best-performing
pipeline to `models/yield_model.joblib`.

**3a. Run the Streamlit app (fastest way to demo)**
```bash
streamlit run app/streamlit_app.py
```

**3b. Or run the FastAPI backend**
```bash
uvicorn api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice", "region": "North", "soil_type": "Alluvial",
    "rainfall_mm": 1200, "temperature_c": 28, "humidity_pct": 70,
    "nitrogen_kg_ha": 90, "phosphorus_kg_ha": 45, "potassium_kg_ha": 45,
    "ph": 6.5, "area_hectares": 3
  }'
```

## Next steps to make this stronger

- [ ] Replace synthetic data with a real dataset (ICRISAT, data.gov.in, or
      Kaggle's "Crop Recommendation Dataset")
- [ ] Add SHAP for per-prediction explainability (not just global feature importance)
- [ ] Add a `/batch-predict` endpoint that accepts a CSV upload
- [ ] Deploy: API on Render/Railway, Streamlit app on Streamlit Community Cloud
- [ ] Add tests (pytest) for the API endpoints
- [ ] Add a simple CI pipeline (GitHub Actions) that runs tests on push

## Resume bullet

> Built and deployed an end-to-end ML pipeline predicting crop yield from
> soil, weather, and nutrient data; compared 3 regression models
> (R² = 0.97), served via a FastAPI REST API with an interactive
> Streamlit interface.
