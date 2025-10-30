# streamlit_starbucks_app.py
# Streamlit app for a Starbucks MLR model with robust loading and automatic feature detection/alignment.
import streamlit as st
import io
import os
import joblib
import pickle
import dill
import requests
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Starbucks MLR Predictor", layout="centered")
st.title("☕ Starbucks Sales Predictor — Robust")

st.markdown(
    "Load a `.pkl`/`.joblib` model (local path, raw GitHub URL, or upload). "
    "The app detects the model's expected features (`feature_names_in_` or `n_features_in_`) "
    "and builds the correct input form. It also prints diagnostics so you can debug shape mismatches."
)

# -------------------------
# Sidebar: model load inputs
# -------------------------
st.sidebar.header("Model input")
model_path = st.sidebar.text_input(
    "Model path or raw GitHub URL (optional)",
    value=""
)
uploaded = st.sidebar.file_uploader(
    "Or upload model file (.pkl/.joblib/.sav)", type=["pkl", "joblib", "sav", "bin"]
)
load_btn = st.sidebar.button("Load model")

def try_load_bytes(bytes_obj: bytes):
    """Try joblib -> pickle -> dill to deserialize bytes."""
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
    """Load model from uploaded file > URL > local path. Returns (model_or_None, info_str)."""
    if uploaded_file is not None:
        try:
            bytes_obj = uploaded_file.getbuffer()
            m = try_load_bytes(bytes_obj)
            return m, f"Loaded from uploaded file: {uploaded_file.name}"
        except Exception as e:
            return None, f"Failed to load uploaded file: {e}"

    if not path:
        return None, "No path provided."

    try:
        if path.startswith("http://") or path.startswith("https://"):
            r = requests.get(path, timeout=20)
            r.raise_for_status()
            m = try_load_bytes(r.content)
            return m, "Loaded from URL"
        else:
            if not os.path.exists(path):
                return None, f"Local path not found: {path}"
            with open(path, "rb") as f:
                b = f.read()
            m = try_load_bytes(b)
            return m, f"Loaded from local path: {path}"
    except Exception as e:
        return None, f"Failed to load model: {e}"

# session state for model persistence
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

# convenience: auto-load uploaded if not explicitly loaded yet
if st.session_state.model is None and uploaded is not None and not load_btn:
    model, info = load_model_from_source("", uploaded)
    st.session_state.model = model
    st.session_state.model_info = info
    if model is not None:
        st.sidebar.success(info)

st.markdown("---")

# Show model status and basic diagnostics
model = st.session_state.get("model", None)
if model is None:
    st.info("No model loaded. Provide a path/URL or upload a model in the left pane and click 'Load model'.")
else:
    st.success("Model loaded successfully.")
    st.write("Model info:", st.session_state.model_info)
    st.write("Model type:", type(model).__name__)
    # show pipeline steps if pipeline
    try:
        if isinstance(model, Pipeline):
            st.write("Pipeline steps:", [name for name, _ in model.steps])
    except Exception:
        pass

# -------------------------
# Feature detection utilities
# -------------------------
def detect_model_feature_info(model):
    """
    Returns (detected_feature_names_or_None, expected_n_or_None, info_text).
    Tries estimator.feature_names_in_, estimator.n_features_in_, and Pipeline last step.
    """
    if model is None:
        return None, None, "No model loaded."

    estimator = model
    try:
        if isinstance(model, Pipeline):
            # pick last step as estimator
            estimator = model.steps[-1][1]
    except Exception:
        estimator = model

    # feature_names_in_ on estimator
    if hasattr(estimator, "feature_names_in_"):
        f = getattr(estimator, "feature_names_in_")
        try:
            names = list(f) if not isinstance(f, str) else [f]
        except Exception:
            names = None
        return names, getattr(estimator, "n_features_in_", None), "Detected feature_names_in_ on final estimator."

    # n_features_in_ on estimator
    if hasattr(estimator, "n_features_in_"):
        try:
            n = int(getattr(estimator, "n_features_in_"))
            return None, n, "Detected n_features_in_ on final estimator."
        except Exception:
            pass

    # try top-level
    if hasattr(model, "feature_names_in_"):
        f = getattr(model, "feature_names_in_")
        try:
            names = list(f) if not isinstance(f, str) else [f]
        except Exception:
            names = None
        return names, getattr(model, "n_features_in_", None), "Detected feature_names_in_ on model."

    if hasattr(model, "n_features_in_"):
        try:
            n = int(getattr(model, "n_features_in_"))
            return None, n, "Detected n_features_in_ on model."
        except Exception:
            pass

    return None, None, "Could not detect feature info automatically."

# -------------------------
# Determine feature names / count
# -------------------------
detected_names, expected_n, detect_info = detect_model_feature_info(model)
st.write("Feature detection:", detect_info)
if detected_names is not None:
    st.write("Detected feature names:", detected_names)
if expected_n is not None:
    st.write("Model expected n_features_in_ =", expected_n)

# Provide ways for user to supply missing feature names
feature_names = None

# 1) If detection gave names, use them
if detected_names is not None:
    feature_names = detected_names

# 2) else allow sample CSV to infer columns
if feature_names is None:
    st.write("If the app couldn't detect feature names, you can upload a sample CSV (first row headers) or enter names manually.")
    sample_csv = st.file_uploader("Optional: upload sample CSV to infer column names", type=["csv"], key="sample_csv")
    if sample_csv is not None:
        try:
            df_sample = pd.read_csv(sample_csv, nrows=5)
            cols = df_sample.columns.tolist()
            st.write("Columns detected in sample CSV:", cols)
            chosen = st.multiselect("Choose columns to use as features", options=cols, default=cols)
            if chosen:
                feature_names = chosen
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# 3) manual comma-separated names
if feature_names is None:
    manual = st.text_input("Or enter comma-separated feature names (e.g. Temp,Promo,Holiday). Leave blank to use placeholder names.")
    if manual:
        feature_names = [f.strip() for f in manual.split(",") if f.strip()]

# 4) if still None but expected_n detected, create placeholders x1..xN
if feature_names is None and expected_n is not None:
    feature_names = [f"x{i+1}" for i in range(expected_n)]
    st.warning(f"Created {expected_n} placeholder feature names: {feature_names}")

# 5) final fallback ask for number and create placeholders
if feature_names is None:
    cnt = st.number_input("Number of features (couldn't detect)", min_value=1, max_value=200, value=3, step=1)
    feature_names = [f"x{i+1}" for i in range(int(cnt))]
    st.write("Using placeholder names:", feature_names)

# If expected_n exists and count mismatches, notify and adjust if necessary
if expected_n is not None and len(feature_names) != expected_n:
    st.warning(
        f"Model expects {expected_n} features but you provided {len(feature_names)} names. "
        "App will use the names you provided. If your model is a pipeline that expands features (one-hot), "
        "you must load the full pipeline or provide preprocessed features."
    )

st.write("Final feature list used by the app:", feature_names)

# -------------------------
# Build input form and predict
# -------------------------
st.markdown("### Enter feature values for prediction")
with st.form("predict_form"):
    input_values = {}
    for fname in feature_names:
        # default to numeric input; user can input numeric-coded categorical if needed
        input_values[fname] = st.number_input(f"{fname}", value=0.0, format="%.6f", step=1.0, key=f"inp_{fname}")
    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    if model is None:
        st.error("No model loaded. Load a model first in the sidebar.")
    else:
        try:
            X = pd.DataFrame([input_values], columns=feature_names)
            st.write("Prepared input (shape):", X.shape)
            st.dataframe(X)

            # Try predicting with DataFrame, then fallback to numpy values
            try:
                pred = model.predict(X)
            except Exception as e_df:
                st.write("predict(X) failed with:", str(e_df))
                try:
                    pred = model.predict(X.values)
                except Exception as e_vals:
                    st.error(f"Prediction failed for both DataFrame and numpy array:\nDataFrame error: {e_df}\nNumpy error: {e_vals}")
                    pred = None

            if pred is not None:
                # For MLR the output is numeric; format nicely
                try:
                    val = float(pred[0])
                    st.success(f"Predicted value: {val:.6f}")
                except Exception:
                    st.success(f"Predicted output: {pred}")
        except Exception as e:
            st.error(f"Prediction pipeline error: {e}")

# -------------------------
# Extra debugging helpers
# -------------------------
st.markdown("---")
st.markdown("#### Debugging helpers (toggle if you need them)")

if st.checkbox("Show model attributes (dir)"):
    if model is None:
        st.info("No model loaded.")
    else:
        try:
            st.write(dir(model))
        except Exception as e:
            st.error(f"Could not list model attributes: {e}")

if st.checkbox("Show model params / named steps (if any)"):
    if model is None:
        st.info("No model loaded.")
    else:
        try:
            if hasattr(model, "get_params"):
                st.write(model.get_params())
            if isinstance(model, Pipeline):
                st.write("Pipeline details:")
                for name, step in model.steps:
                    st.write(f"- {name}: {type(step).__name__}")
        except Exception as e:
            st.error(f"Could not inspect model: {e}")

st.markdown(
    "- If your model is a Pipeline with preprocessing (scaler, one-hot, columntransformer), you should load the *entire pipeline* so the app can accept raw inputs (strings/categories) and perform preprocessing. "
    "- If your model expects preprocessed expanded features (e.g., after one-hot encoding), you must supply those numeric features exactly as the model expects."
)
