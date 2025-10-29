# app.py — NexGen – Predictive Logistics Dashboard (Mihika)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="NexGen – Predictive Logistics Dashboard", layout="wide")
st.title("🚚 NexGen – Predictive Logistics Dashboard")
st.caption("Developed by Mihika Arora | AI-Powered Delay Prediction (XGBoost)")

# -------------------------
# LOAD MODEL + FEATURES
# -------------------------
@st.cache_resource
def load_model_and_schema():
    model_path = "models/xgb_delay.pkl"
    cols_path  = "models/feature_cols.pkl"
    if not (os.path.exists(model_path) and os.path.exists(cols_path)):
        raise FileNotFoundError(
            "Model files not found. Please ensure 'models/xgb_delay.pkl' and 'models/feature_cols.pkl' exist."
        )
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    # ensure list
    if not isinstance(feature_cols, (list, tuple, pd.Index)):
        feature_cols = list(feature_cols)
    return model, list(feature_cols)

try:
    model, feature_cols = load_model_and_schema()
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

# -------------------------
# INPUTS + PREDICTION
# -------------------------
st.subheader("📊 Enter Order Details")

col1, col2, col3 = st.columns(3)
with col1:
    fuel_cost = st.number_input("Fuel Cost (INR)", 0.0, 5000.0, 200.0, key="fuel_cost")
    labor_cost = st.number_input("Labor Cost (INR)", 0.0, 5000.0, 150.0, key="labor_cost")
with col2:
    vehicle_maintenance = st.number_input("Vehicle Maintenance (INR)", 0.0, 5000.0, 400.0, key="vehicle_maintenance")
    distance_km = st.number_input("Distance (KM)", 0.0, 2000.0, 500.0, key="distance_km")
with col3:
    traffic_delay_minutes = st.number_input("Traffic Delay (Minutes)", 0.0, 300.0, 30.0, key="traffic_delay_minutes")
    weather_choice = st.selectbox("Weather Impact", ["None", "Light_Rain", "Heavy_Rain"], key="weather_impact")

thresh = st.slider("Alert threshold (probability)", 0.1, 0.9, 0.5, 0.05, key="threshold")

def build_aligned_row():
    """
    Build a single-row DataFrame with EXACTLY the same columns as the model was trained on.
    We zero-fill everything, then set values for columns that exist in the schema.
    """
    Xrow = pd.DataFrame(columns=feature_cols)
    Xrow.loc[0] = 0

    # numeric features that might exist in the training schema
    valmap = {
        "fuel_cost": fuel_cost,
        "labor_cost": labor_cost,
        "vehicle_maintenance": vehicle_maintenance,
        "distance_km": distance_km,
        "traffic_delay_minutes": traffic_delay_minutes,
        # other cost features will remain zero unless you later add inputs:
        # "insurance", "packaging_cost", "technology_platform_fee", "other_overhead"
        # order/delivery fields that might be in the schema (kept at 0 if absent in UI):
        # "promised_delivery_days", "actual_delivery_days", "customer_rating", "delivery_cost_inr"
    }
    for col, val in valmap.items():
        if col in Xrow.columns:
            Xrow.loc[0, col] = val

    # weather -> dummy columns (if present in the trained schema)
    if weather_choice == "Light_Rain" and "weather_impact_Light_Rain" in Xrow.columns:
        Xrow.loc[0, "weather_impact_Light_Rain"] = 1
    if weather_choice == "Heavy_Rain" and "weather_impact_Heavy_Rain" in Xrow.columns:
        Xrow.loc[0, "weather_impact_Heavy_Rain"] = 1
    # "None" leaves both at 0

    # final clean
    Xrow = Xrow.apply(pd.to_numeric, errors="coerce").fillna(0)
    # order columns to be absolutely safe (already created with the same order)
    Xrow = Xrow[feature_cols]
    return Xrow

if st.button("🔮 Predict Delay", key="predict"):
    Xrow = build_aligned_row()

    # Predict
    yhat = int(model.predict(Xrow)[0])
    try:
        proba = float(model.predict_proba(Xrow)[0, 1])
    except Exception:
        proba = None

    # Pretty result
    if yhat == 1:
        st.error(f"🚨 Predicted: DELAYED" + (f" • confidence: {proba:.0%}" if proba is not None else ""))
    else:
        st.success(f"✅ Predicted: ON-TIME" + (f" • confidence: {(1-proba):.0%}" if proba is not None else ""))

    # thresholded label
    if proba is not None:
        label = "DELAY" if proba >= thresh else "ON-TIME"
        st.caption(f"Decision at threshold {thresh:.2f}: **{label}**  (prob={proba:.2%})")

    # Debug expander
    with st.expander("Show aligned feature vector (non-zeros)"):
        nz = Xrow.T[Xrow.T[0] != 0]
        st.write(nz if not nz.empty else "All zeros (baseline); only UI-provided features were set.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("💡 _Powered by XGBoost • Feature-aligned inference • © NexGen 2025_")

