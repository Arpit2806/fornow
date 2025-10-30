# streamlit_starbucks_app.py
# Simple Streamlit dashboard for a Starbucks MLR model
# - Loads a .pkl (local path or raw GitHub URL) or allows upload
# - If model exposes `feature_names_in_` or `n_features_in_` it will build inputs automatically
# - Otherwise user can enter feature names manually or provide a sample CSV to infer

import streamlit as st
import io
import os
import joblib
import pickle
import dill
import requests
import numpy as np
import pandas as pd

st.set_page_config(page_title="Starbucks MLR Predictor", layout="centered")
st.title("☕ Starbucks Sales Predictor")
st.markdown("Load your Starbucks MLR model (.pkl/.joblib) by path, URL, or upload. The app builds an input form from the model when possible.")

# Sidebar: model input
st.sidebar.header("Model input")
model_path = st.sidebar.text_input("Starbucks_MLR.py", value="")
uploaded = st.sidebar.file_uploader("Or upload model file (.pkl/.joblib)", type=["pkl","joblib","sav","bin"]) 
load_btn = st.sidebar.button("Load model")

# small helper to attempt multiple deserializers
def try_load_bytes(bytes_obj: bytes):
    errors = []
    # joblib
    try:
        bio = io.BytesIO(bytes_obj)
        return joblib.load(bio)
    except Exception as e:
        errors.append(f"joblib: {e}")
    # pickle
    try:
        return pickle.loads(bytes_obj)
    except Exception as e:
        errors.append(f"pickle: {e}")
    # dill
    try:
        return dill.loads(bytes_obj)
    except Exception as e:
        errors.append(f"dill: {e}")
    raise Exception("All loaders failed:\n" + "\n".join(errors))


def load_model_from_source(path: str, uploaded_file):
    # uploaded takes precedence
    if uploaded_file is not None:
        try:
            bytes_obj = uploaded_file.getbuffer()
            m = try_load_bytes(bytes_obj)
            return m, f"Loaded from uploaded file: {uploaded_file.name}"
        except Exception as e:
            return None, f"Failed to load uploaded file: {e}"

    if not path:
        return None, "No path provided"

    try:
        if path.startswith("http://") or path.startswith("https://"):
            r = requests.get(path, timeout=15)
            r.raise_for_status()
            m = try_load_bytes(r.content)
            return m, f"Loaded from URL"
        else:
            if not os.path.exists(path):
                return None, f"Local path not found: {path}"
            with open(path, "rb") as f:
                b = f.read()
            m = try_load_bytes(b)
            return m, f"Loaded from local path"
    except Exception as e:
        return None, f"Failed to load model: {e}"

# session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_info = ""

if load_btn:
    with st.spinner("Loading model..."):
        model, info = load_model_from_source(model_path.strip(), uploaded)
        st.session_state.model = model
        st.session_state.model_info = info
        if model is not None:
            st.sidebar.success(info)
        else:
            st.sidebar.error(info)

# Auto-load uploaded if not loaded (convenience)
if st.session_state.model is None and uploaded is not None and not load_btn:
    model, info = load_model_from_source("", uploaded)
    st.session_state.model = model
    st.session_state.model_info = info
    if model is not None:
        st.sidebar.success(info)

st.markdown("---")
# Show model status
if st.session_state.model is None:
    st.info("No model loaded. Provide a path/URL or upload a model in the left pane and click 'Load model'.")
else:
    st.success("Model loaded successfully")
    st.write("Model info:", st.session_state.model_info)
    try:
        st.write("Model type:", type(st.session_state.model).__name__)
    except Exception:
        pass

# Function to get feature names
def get_feature_names(model):
    if model is None:
        return None
    # sklearn estimators often expose feature_names_in_
    if hasattr(model, "feature_names_in_"):
        f = getattr(model, "feature_names_in_")
        return list(f) if isinstance(f, (list, np.ndarray)) else list(f)
    if hasattr(model, "n_features_in_"):
        n = getattr(model, "n_features_in_")
        # create placeholder names
        return [f"x{i+1}" for i in range(int(n))]
    # try pipeline last step
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            last = model.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(getattr(last, "feature_names_in_"))
    except Exception:
        pass
    return None

model = st.session_state.get("model", None)
feature_names = get_feature_names(model)

# Allow manual feature names if not detected
if feature_names is None:
    st.write("Could not detect features automatically.")
    sample_csv = st.file_uploader("Optional: upload a sample CSV to infer column names (first row used)", type=["csv"], key="sample_csv")
    if sample_csv is not None:
        try:
            df_sample = pd.read_csv(sample_csv, nrows=5)
            cols = df_sample.columns.tolist()
            chosen = st.multiselect("Choose columns to use", options=cols, default=cols)
            if chosen:
                feature_names = chosen
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if feature_names is None:
        manual = st.text_input("Or enter comma-separated feature names (e.g. Temp,Promo,Holiday)")
        if manual:
            feature_names = [f.strip() for f in manual.split(",") if f.strip()]

# If still None, ask for count and create placeholders
if feature_names is None:
    cnt = st.number_input("Number of features", min_value=1, max_value=50, value=3, step=1)
    feature_names = [f"x{i+1}" for i in range(int(cnt))]

st.write("Using features:", feature_names)

# Build input form
st.markdown("### Enter feature values")
with st.form("predict_form"):
    inputs = {}
    for fname in feature_names:
        # default to numeric input
        val = st.number_input(f"{fname}", value=0.0, format="%.6f", step=1.0, key=f"inp_{fname}")
        inputs[fname] = val
    predict = st.form_submit_button("Predict")

if predict:
    if model is None:
        st.error("No model loaded. Load a model first in the sidebar.")
    else:
        try:
            X = pd.DataFrame([inputs], columns=feature_names)
            st.write("Prepared input:")
            st.dataframe(X)
            # Try predict
            try:
                pred = model.predict(X)
            except Exception:
                pred = model.predict(X.values)
            # For MLR, prediction is numeric
            val = float(pred[0])
            st.write("### Prediction")
            st.success(f"Predicted value: {val:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("**Notes:**\n- For GitHub-hosted models, use the raw file URL.\n- If the model expects preprocessed features (scaling/encoding), make sure you provide them or load a pipeline that includes preprocessing.")
