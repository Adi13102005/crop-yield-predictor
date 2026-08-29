"""
Streamlit frontend for the crop yield predictor.

Uses a separate model per crop (see models/train.py) so predictions
and SHAP explanations reflect that crop's own agronomic drivers,
instead of being dominated by scale differences between crops.

Run:
    streamlit run app/streamlit_app.py
"""

import joblib
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("models/yield_model.joblib")

@st.cache_resource
def get_explainer(_model, crop):
    # Keyed by crop so each crop's model gets its own cached explainer.
    return shap.TreeExplainer(_model)

bundle = load_bundle()
per_crop_models = bundle["models"]

st.title("🌾 Crop Yield Predictor")
st.caption("A separate model is trained per crop — see performance below once you pick one.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("Crop", bundle["crops"])
        region = st.selectbox("Region", bundle["regions"])
        soil_type = st.selectbox("Soil type", bundle["soil_types"])
        area_hectares = st.number_input("Area (hectares)", min_value=0.1, value=2.0)

    with col2:
        rainfall_mm = st.slider("Rainfall (mm)", 0, 3000, 1000)
        temperature_c = st.slider("Temperature (°C)", -10, 55, 27)
        humidity_pct = st.slider("Humidity (%)", 0, 100, 65)
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5)

    st.markdown("**Nutrients (kg/hectare)**")
    n1, n2, n3 = st.columns(3)
    nitrogen_kg_ha = n1.number_input("Nitrogen (N)", 0, 300, 80)
    phosphorus_kg_ha = n2.number_input("Phosphorus (P)", 0, 300, 40)
    potassium_kg_ha = n3.number_input("Potassium (K)", 0, 300, 40)

    submitted = st.form_submit_button("Predict Yield", use_container_width=True)

if submitted:
    entry = per_crop_models[crop]
    pipeline = entry["pipeline"]

    st.caption(
        f"Model for {crop}: {entry['model_name']}  |  "
        f"R² = {entry['metrics']['R2']}  |  MAE = {entry['metrics']['MAE']} t/ha"
    )

    input_df = pd.DataFrame([{
        "region": region, "soil_type": soil_type,
        "rainfall_mm": rainfall_mm, "temperature_c": temperature_c,
        "humidity_pct": humidity_pct, "nitrogen_kg_ha": nitrogen_kg_ha,
        "phosphorus_kg_ha": phosphorus_kg_ha, "potassium_kg_ha": potassium_kg_ha,
        "ph": ph, "area_hectares": area_hectares,
    }])

    pred_per_ha = float(pipeline.predict(input_df)[0])
    pred_total = pred_per_ha * area_hectares

    st.success("Prediction complete")
    m1, m2 = st.columns(2)
    m1.metric("Yield per hectare", f"{pred_per_ha:.2f} t/ha")
    m2.metric("Total estimated yield", f"{pred_total:.2f} t")

    # Per-prediction SHAP explanation, scoped to this crop's own model —
    # no more "not Sugarcane" dominating every chart.
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    if hasattr(model, "feature_importances_"):
        st.markdown(f"**What's driving *this* {crop} prediction (SHAP values)**")

        cat_features = (
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(bundle["categorical_features"])
        )
        all_features = bundle["numeric_features"] + list(cat_features)

        X_transformed = preprocessor.transform(input_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        explainer = get_explainer(model, crop)
        shap_values = explainer.shap_values(X_transformed)[0]

        shap_df = pd.DataFrame({"Feature": all_features, "SHAP value": shap_values})
        shap_df["abs"] = shap_df["SHAP value"].abs()
        shap_df = (
            shap_df.sort_values("abs", ascending=False)
            .head(8)
            .drop(columns="abs")
            .set_index("Feature")
        )

        st.bar_chart(shap_df)
        st.caption(
            "Positive values push this prediction's yield up; negative values "
            "push it down. Specific to the inputs you just entered, using the "
            f"model trained only on {crop} data."
        )
    else:
        st.info(f"The best model for {crop} was Linear Regression, which doesn't "
                "support SHAP tree explanations — coefficients drive it linearly instead.")

st.divider()
st.caption(
    "Note: trained on synthetic data for demo purposes."
   
)
