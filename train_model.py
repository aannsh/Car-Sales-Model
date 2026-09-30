"""Train offline and save the model so the deployed app loads instantly.

    python train_model.py --data Car_Sales.csv
Creates models/price_model.joblib (+ .metrics.json, .catalog.json). Commit those three files.
"""
import argparse

from streamlit_app import MODEL_PATH, build_catalog, fit_and_evaluate, read_table, save_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="Car sales .csv or .xlsx")
args = ap.parse_args()

df = read_table(args.data, args.data)
model, metrics = fit_and_evaluate(df, refit_on_all=True)
save_model(model, MODEL_PATH, metrics, build_catalog(df))
print(f"Saved {MODEL_PATH}")
for k, v in metrics.items():
    print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")
