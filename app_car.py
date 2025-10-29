# app.py (updated loader)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import gzip
import bz2
import lzma
import io
import os

st.set_page_config(page_title="Car Price Prediction Dashboard", layout="wide")
st.title("🚗 Car Price Prediction Dashboard")

MODEL_PATH = "/mnt/data/car_price_model.pkl"

@st.cache_resource
def try_pickle_load(data_bytes):
    """Try plain pickle (different protocols)."""
    try:
        return pickle.loads(data_bytes)
    except Exception:
        return None

@st.cache_resource
def try_joblib_load(file_path):
    """Try joblib.load (handles joblib and some sklearn dumps)."""
    try:
        return joblib.load(file_path)
    except Exception:
        return None

@st.cache_resource
def smart_load_model(path):
    """
    Attempts to load a model saved in several common formats:
    - plain pickle
    - gzip-compressed pickle
    - bz2-compressed pickle
    - lzma-compressed pickle
    - joblib dump
    Returns (loaded_object_or_None, diagnostics_dict)
    """
    diag = {"path": path, "exists": False, "size_bytes": None, "first_bytes_hex": None, "tried": []}
    if not os.path.exists(path):
        diag["tried"].append("file_not_found")
        return None, diag

    diag["exists"] = True
    size = os.path.getsize(path)
    diag["size_bytes"] = size

    # read initial bytes for diagnostics & format detection
    with open(path, "rb") as f:
        head = f.read(256)
    diag["first_bytes_hex"] = head[:64].hex()

    # detect compression by magic bytes
    # gzip: 1f 8b
    # bz2: 42 5a 68 -> 'BZh'
    # lzma/xz: fd 37 7a 58 5a 00
    if head.startswith(b"\x1f\x8b"):
        diag["tried"].append("gzip")
        try:
            with gzip.open(path, "rb") as gf:
                data = gf.read()
            obj = try_pickle_load(data)
            if obj is not None:
                return obj, diag
        except Exception as e:
            diag["gzip_err"] = str(e)
    elif head.startswith(b"BZh"):
        diag["tried"].append("bz2")
        try:
            with bz2.open(path, "rb") as bf:
                data = bf.read()
            obj = try_pickle_load(data)
            if obj is not None:
                return obj, diag
        except Exception as e:
            diag["bz2_err"] = str(e)
    elif head.startswith(b"\xfd7zXZ"):
        diag["tried"].append("lzma")
        try:
            with lzma.open(path, "rb") as lf:
                data = lf.read()
            obj = try_pickle_load(data)
            if obj is not None:
                return obj, diag
        except Exception as e:
            diag["lzma_err"] = str(e)

    # Try joblib.load (joblib files may not show the same magic bytes)
    diag["tried"].append("joblib")
    try:
        obj = try_joblib_load(path)
        if obj is not None:
            return obj, diag
    except Exception as e:
        diag["joblib_err"] = str(e)

    # Fallback: try reading raw bytes and pickle.loads
    diag["tried"].append("pickle_raw")
    try:
        with open(path, "rb") as f:
            data = f.read()
        obj = try_pickle_load(data)
        if obj is not None:
            return obj, diag
    except Exception as e:
        diag["pickle_raw_err"] = str(e)

    # nothing worked
    return None, diag

# --- load model using smart loader ---
model_obj = None
metadata = {}
model, diag = smart_load_model(MODEL_PATH)

if model is None:
    st.sidebar.error("Failed to load model. See diagnostics below.")
    st.sidebar.write(diag)
    st.stop()
else:
    # If it's a dict-like object, try to extract commonly used keys
    if isinstance(model, dict):
        model_obj = model.get("model") or model.get("estimator") or model.get("clf") or model
        metadata = model
    else:
        model_obj = model
        metadata = {}

    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.write({"model_type": type(model_obj).__name__})

# --- rest of app: infer features and build form (simplified) ---
feature_columns = None
if "columns" in metadata:
    feature_columns = list(metadata["columns"])
elif hasattr(model_obj, "feature_names_in_"):
    try:
        feature_columns = list(model_obj.feature_names_in_)
    except Exception:
        feature_columns = None

if feature_columns:
    st.info("Detected feature columns from model metadata.")
else:
    st.warning("No feature names detected. Please provide input fields manually below.")

# Simple manual form (fallback)
with st.form("input_form"):
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=15.0)
    engine = st.number_input("Engine Capacity (cc)", min_value=0.0, value=1200.0)
    power = st.number_input("Power (bhp)", min_value=0.0, value=74.0)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    submitted = st.form_submit_button("Predict Price")

if submitted:
    inputs = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "power": power,
        "fuel_type": fuel_type,
        "transmission": transmission
    }
    input_df = pd.DataFrame([inputs])
    st.write("### Input")
    st.dataframe(input_df)

    try:
        pred = model_obj.predict(input_df)
        st.success(f"Predicted value: {pred[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Diagnostics:", diag)
