"""Car Price Predictor: ex-showroom price (INR) from vehicle specs.

Self-contained: model code and app live in this one file.
  Run locally : streamlit run streamlit_app.py
  Deploy      : Streamlit Community Cloud, Main file path = streamlit_app.py
  Train offline: python train_model.py --data Car_Sales.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# Model code (mirrors the notebook: Ridge on log(price), 80/20 split, seed 42)
# ----------------------------------------------------------------------------
TARGET = "Ex_Showroom_Price_INR"
CATEGORICAL = ["Brand", "Vehicle_Type", "Fuel_Type", "Transmission", "Variant", "Model"]
NUMERIC = ["Seating_Capacity", "Engine_CC", "Fuel_Efficiency_KMPL", "EV_Range_KM", "Manufacturing_Year"]
RAW_INPUT_COLUMNS = [
    "Brand", "Model", "Variant", "Vehicle_Type", "Fuel_Type", "Transmission",
    "Seating_Capacity", "Engine_CC", "Mileage_KMPL_or_Range_KM", "Manufacturing_Year",
]
RANDOM_STATE = 42
MODEL_PATH = Path("models/price_model.joblib")
CATALOG_PATH = MODEL_PATH.with_suffix(".catalog.json")
METRICS_PATH = MODEL_PATH.with_suffix(".metrics.json")


def add_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split the mixed mileage column: km/l for fuel cars, range in km for EVs (other one = 0)."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    is_ev = out["Fuel_Type"].eq("Electric")
    mileage = out["Mileage_KMPL_or_Range_KM"].astype(float)
    out["Fuel_Efficiency_KMPL"] = mileage.where(~is_ev, 0.0)
    out["EV_Range_KM"] = mileage.where(is_ev, 0.0)
    return out


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
         ("num", "passthrough", NUMERIC)]
    )
    return Pipeline([
        ("prep", pre),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 60), cv=5)),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y_price: pd.Series) -> dict:
    pred_inr = np.exp(model.predict(X))
    true_inr = y_price.to_numpy(dtype=float)
    return {
        "R2_log": float(r2_score(np.log(true_inr), np.log(pred_inr))),
        "R2_INR": float(r2_score(true_inr, pred_inr)),
        "RMSE_INR_lakh": float(np.sqrt(mean_squared_error(true_inr, pred_inr)) / 1e5),
        "MAE_INR_lakh": float(mean_absolute_error(true_inr, pred_inr) / 1e5),
        "MAPE_pct": float(np.mean(np.abs(true_inr - pred_inr) / true_inr) * 100),
        "n_test": int(len(true_inr)),
    }


def fit_and_evaluate(df: pd.DataFrame, test_size: float = 0.20, refit_on_all: bool = False):
    """Fit on an 80/20 split and report test metrics; optionally refit on every row afterwards."""
    data = add_vehicle_features(df)
    X, y = data[CATEGORICAL + NUMERIC], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    model = build_pipeline().fit(X_tr, np.log(y_tr))
    metrics = evaluate(model, X_te, y_te)
    metrics["ridge_alpha"] = float(model.named_steps["ridge"].alpha_)
    if refit_on_all:
        model = build_pipeline().fit(X, np.log(y))
        metrics["refit_on_all_rows"] = True
    return model, metrics


def predict_price(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    X = add_vehicle_features(df)[CATEGORICAL + NUMERIC]
    return pd.Series(np.exp(model.predict(X)), index=df.index, name="Predicted_Ex_Showroom_Price_INR")


def build_catalog(df: pd.DataFrame) -> dict:
    """Which models exist under each brand (filters the Model dropdown)."""
    return {"models_by_brand": {b: sorted(g["Model"].unique().tolist()) for b, g in df.groupby("Brand")}}


def save_model(model, path=MODEL_PATH, metrics=None, catalog=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    if metrics is not None:
        path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    if catalog is not None:
        path.with_suffix(".catalog.json").write_text(json.dumps(catalog, indent=2))


def read_table(file_or_path, name: str) -> pd.DataFrame:
    return pd.read_excel(file_or_path) if name.lower().endswith((".xlsx", ".xls")) else pd.read_csv(file_or_path)


# ----------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------
@st.cache_resource
def get_saved_model():
    """Load the committed model if present and readable; otherwise return None."""
    if not MODEL_PATH.exists():
        return None, {}, {}, None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # e.g. scikit-learn version mismatch between training and deploy
        return None, {}, {}, f"Saved model could not be loaded ({type(exc).__name__}); please retrain below."
    catalog = json.loads(CATALOG_PATH.read_text()) if CATALOG_PATH.exists() else {}
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return model, catalog, metrics, None


def main() -> None:
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    st.title("🚗 Car Price Predictor")
    st.caption("Ridge regression on log(price), trained on the Group 9 car sales dataset.")

    model, catalog, metrics, warning = get_saved_model()
    if model is None:  # fall back to a model trained in this session
        model, catalog, metrics = (st.session_state.get(k) for k in ("model", "catalog", "metrics"))
    if model is None:
        if warning:
            st.warning(warning)
        st.info("No trained model found. Upload the car sales dataset (CSV or Excel) to train one. "
                "The file is used in this session only and is not stored.")
        up = st.file_uploader("Car sales dataset", type=["csv", "xlsx", "xls"])
        if up is not None:
            try:
                with st.spinner("Training..."):
                    data = read_table(up, up.name)
                    model, metrics = fit_and_evaluate(data, refit_on_all=True)
                    st.session_state.update(model=model, metrics=metrics, catalog=build_catalog(data))
            except Exception as exc:
                st.error(f"Could not train from this file: {exc}")
                st.stop()
            st.rerun()
        st.stop()

    prep = model.named_steps["prep"]
    cats = dict(zip(prep.transformers_[0][2], [list(c) for c in prep.named_transformers_["cat"].categories_]))
    models_by_brand = catalog.get("models_by_brand", {})

    col1, col2 = st.columns(2)
    brand = col1.selectbox("Brand", cats["Brand"])
    car_model = col2.selectbox("Model", models_by_brand.get(brand, cats["Model"]))
    variant = col1.selectbox("Variant", cats["Variant"])
    vehicle_type = col2.selectbox("Vehicle type", cats["Vehicle_Type"])
    fuel = col1.selectbox("Fuel type", cats["Fuel_Type"])
    transmission = col2.selectbox("Transmission", cats["Transmission"])
    seats = col1.number_input("Seating capacity", 2, 9, 5)
    is_ev = fuel == "Electric"
    engine = col2.number_input("Engine (cc)", 0, 6000, 0 if is_ev else 1500, disabled=is_ev,
                               help="Electric vehicles have no engine displacement.")
    mileage = col1.number_input("Range (km)" if is_ev else "Mileage (km/l)", 0.0, 1000.0,
                                400.0 if is_ev else 18.0)
    year = col2.number_input("Manufacturing year", 2015, 2030, 2025)

    if st.button("Predict price", type="primary"):
        row = pd.DataFrame([{
            "Brand": brand, "Model": car_model, "Variant": variant, "Vehicle_Type": vehicle_type,
            "Fuel_Type": fuel, "Transmission": transmission, "Seating_Capacity": seats,
            "Engine_CC": 0 if is_ev else engine, "Mileage_KMPL_or_Range_KM": mileage,
            "Manufacturing_Year": year,
        }])
        price = float(predict_price(model, row).iloc[0])
        st.metric("Estimated ex-showroom price", f"₹ {price / 1e5:,.2f} lakh")
        st.caption(f"≈ ₹ {price:,.0f}")
        if metrics:
            st.caption(f"Accuracy on held-out data: R² {metrics['R2_INR']:.2f}, typical error about "
                       f"{metrics['MAPE_pct']:.0f}%. Treat this as an estimate, not a quote.")


if __name__ == "__main__":
    main()
