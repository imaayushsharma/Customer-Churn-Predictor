"""
Streamlit frontend for the Churn prediction model
This single-file app:
- loads model artifacts: model.h5, label_encoder_gender.pkl, ohe_encoder_geography.pkl, scaler.pkl
- presents polished input widgets for the 10 features used in the notebooks
- shows prediction (Exited / Not Exited) and probability
- displays the input summary table and a small explanation

Place this file in the same folder as the model artifacts and run:
    streamlit run streamlit_churn_frontend.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# --- Helper: load artifacts ---
@st.cache_resource
def load_artifacts():
    base = Path(".")
    artifacts = {}
    # Model
    try:
        artifacts['model'] = load_model(base / "model.h5")
    except Exception as e:
        artifacts['model'] = None
        artifacts['model_err'] = str(e)
    # Pickles
    for name in ['label_encoder_gender.pkl', 'ohe_encoder_geography.pkl', 'scaler.pkl']:
        try:
            with open(base / name, 'rb') as f:
                artifacts[name.split('.')[0]] = pickle.load(f)
        except Exception as e:
            artifacts[name.split('.')[0]] = None
            artifacts[name.split('.')[0] + '_err'] = str(e)
    return artifacts

art = load_artifacts()

# --- Top-level UI ---
st.title("Customer Churn Predictor")
st.write(
    "Use this interface to predict whether a bank customer will exit (churn).\n" 
    "Provide customer details on the left and hit **Predict**."
)

with st.expander("Model & artifacts status", expanded=False):
    if art.get('model') is None:
        st.error(f"Model not loaded. Error: {art.get('model_err','unknown')}")
    else:
        st.success("Model loaded successfully.")
    for key in ['label_encoder_gender', 'ohe_encoder_geography', 'scaler']:
        if art.get(key) is None:
            st.warning(f"{key} not loaded. Error: {art.get(key + '_err','unknown')}")
        else:
            st.info(f"{key} loaded")

# --- Sidebar inputs ---
st.sidebar.header("Customer features")
# Reasonable defaults from the notebook
age = st.sidebar.slider('Age', 18, 100, 37)
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=1000, value=715)
geography_opts = None
if art.get('ohe_encoder_geography') is not None:
    try:
        # get categories from the fitted encoder
        geo_names = list(art['ohe_encoder_geography'].categories_[0])
        geography_opts = geo_names
    except Exception:
        geography_opts = ["France", "Spain", "Germany"]
else:
    geography_opts = ["France", "Spain", "Germany"]

geography = st.sidebar.selectbox('Geography', geography_opts, index=0)

gender_opts = None
if art.get('label_encoder_gender') is not None:
    # Try to infer classes order for better defaults
    try:
        gender_classes = list(art['label_encoder_gender'].classes_)
        gender_opts = gender_classes
    except Exception:
        gender_opts = ["Male","Female"]
else:
    gender_opts = ["Male","Female"]

gender = st.sidebar.selectbox('Gender', gender_opts, index=0)

balance = st.sidebar.number_input('Balance', min_value=0.0, value=84532.45, format="%.2f")
num_of_products = st.sidebar.selectbox('Number of Products', [0,1,2,3,4], index=2)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0,1], index=1)
is_active_member = st.sidebar.selectbox('Is Active Member', [0,1], index=0)
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, value=112450.30, format="%.2f")

# Layout: show inputs summary on the main page
st.subheader("Input summary")
input_data = {
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": float(balance),
    "NumOfProducts": int(num_of_products),
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active_member),
    "EstimatedSalary": float(estimated_salary),
}

input_df_display = pd.DataFrame([input_data])
st.table(input_df_display)

# Predict button
if st.button('Predict'):
    # Validate artifacts
    if art.get('model') is None or art.get('scaler') is None:
        st.error("Model or preprocessing artifacts are missing. See the Model & artifacts status expander.")
    else:
        # Build dataframe for model pipeline similar to prediction notebook
        input_df = pd.DataFrame([input_data])
        # One-hot encode geography
        if art.get('ohe_encoder_geography') is not None:
            try:
                encoded_geo = art['ohe_encoder_geography'].transform(input_df[["Geography"]])
                geo_df = pd.DataFrame(encoded_geo, columns=art['ohe_encoder_geography'].get_feature_names_out(["Geography"]))
                input_df = pd.concat([input_df, geo_df], axis=1)
                input_df = input_df.drop(["Geography"], axis=1)
            except Exception as e:
                st.warning(f"Geography encoding failed: {e}")
        else:
            # fallback: drop Geography
            input_df = input_df.drop(["Geography"], axis=1)

        # Label encode Gender
        if art.get('label_encoder_gender') is not None:
            try:
                input_df["Gender"] = art['label_encoder_gender'].transform(input_df["Gender"])
            except Exception as e:
                st.warning(f"Gender encoding failed: {e}")
                # fallback: map Male->1 Female->0
                input_df["Gender"] = input_df["Gender"].map({"Male":1, "Female":0}).fillna(0)
        else:
            input_df["Gender"] = input_df["Gender"].map({"Male":1, "Female":0}).fillna(0)

        # Scale
        try:
            scaled = art['scaler'].transform(input_df)
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

        # Prediction
        try:
            pred = art['model'].predict(scaled)
            # model outputs probabilities in this pipeline
            prob = float(pred[0][0]) if (hasattr(pred[0], '__len__') and len(pred[0])>0) else float(pred[0])
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        churn = 'Exited' if prob >= 0.5 else 'Not Exited'
        prob_pct = round(prob*100, 2)

        st.success(f"Prediction: **{churn}**")
        st.metric(label="Exit probability", value=f"{prob_pct}%")

        # Provide a small probability bar
        st.progress(min(max(prob, 0.0), 1.0))

        # Show transformed input shape and column names for debugging
        st.write("Transformed input shape:", scaled.shape)
        if hasattr(input_df, 'columns'):
            st.write("Transformed feature names:", list(input_df.columns))

        # Short explanation
        st.caption("Threshold used: 0.5 (probability >= 0.5 -> Exited). The preprocessing mirrors the notebook: Geography OHE, Gender label-encoding, then scaling.")

# Footer
st.markdown("---")
st.markdown("Built from provided notebooks. Ensure model.h5 and the three pickles are present in the same folder as this script.")
