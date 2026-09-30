import numpy as np
import pandas as pd
import pytest

from streamlit_app import add_vehicle_features, fit_and_evaluate, predict_price


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    brand = rng.choice(["Hyundai", "Toyota", "BMW"], n)
    fuel = rng.choice(["Petrol", "Diesel", "Electric"], n)
    vtype = np.where(brand == "BMW", "Luxury SUV", rng.choice(["SUV", "Hatchback"], n))
    cc = np.where(fuel == "Electric", 0, rng.choice([1197, 1497, 1956], n))
    base = 8e5 + cc * 500 + np.where(vtype == "Luxury SUV", 4e6, 0) + np.where(fuel == "Electric", 5e5, 0)
    return pd.DataFrame({
        "Brand": brand, "Model": rng.choice(["A", "B", "C"], n), "Variant": rng.choice(["Base", "Mid", "Top"], n),
        "Vehicle_Type": vtype, "Fuel_Type": fuel, "Transmission": rng.choice(["Manual", "Automatic"], n),
        "Seating_Capacity": rng.choice([5, 7], n), "Engine_CC": cc,
        "Mileage_KMPL_or_Range_KM": np.where(fuel == "Electric", 400.0, 18.0),
        "Manufacturing_Year": rng.choice([2023, 2024, 2025], n),
        "Ex_Showroom_Price_INR": (base * rng.lognormal(0, 0.05, n)).round(0),
    })


def test_mileage_split():
    out = add_vehicle_features(synthetic(50))
    ev = out["Fuel_Type"] == "Electric"
    assert (out.loc[ev, "Fuel_Efficiency_KMPL"] == 0).all() and (out.loc[~ev, "EV_Range_KM"] == 0).all()


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        add_vehicle_features(synthetic(10).drop(columns=["Brand"]))


def test_train_and_predict():
    df = synthetic()
    model, metrics = fit_and_evaluate(df)
    assert metrics["R2_INR"] > 0.9
    assert (predict_price(model, df.drop(columns=["Ex_Showroom_Price_INR"]).head(5)) > 0).all()


def test_unseen_category_ok():
    model, _ = fit_and_evaluate(synthetic())
    assert predict_price(model, synthetic(1).assign(Model="Brand New")).iloc[0] > 0
