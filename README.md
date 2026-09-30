# Car Sales Analytics (Group 9)

Analysis of a 2,000-row Indian car-sales dataset (43 columns, Jan 2024 to Aug 2026) plus a
**Streamlit app that predicts a car's ex-showroom price**.

- `Final_Project_Group_9.ipynb`: full analysis (EDA, tests, regression, clustering, classification, factor analysis)
- `streamlit_app.py`: the app; the model code is inside this single file
- `train_model.py`: optional offline training so the app starts with a ready model
- `test_app.py`: tests

## Results (20% test set, log-price target)

| Model | R² (INR) | MAE (lakh) | MAPE |
|---|---|---|---|
| Linear Regression | 0.940 | 4.04 | 14.6% |
| Ridge (used in the app) | 0.940 | 4.06 | 14.7% |
| Lasso | 0.940 | 4.08 | 14.7% |
| Random Forest | 0.933 | 4.17 | 14.8% |
| Decision Tree | 0.924 | 4.31 | 15.0% |

Cancellation and financing could not be predicted from this data (AUC about 0.5), so they are analysis
only and not part of the app. Typical price error is about 15%: use it for estimates, not quotes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub (all files at the top level of the repo).
2. On share.streamlit.io choose **Create app**, pick the repo and branch `main`, and set
   **Main file path** to `streamlit_app.py`.

Without a saved model, the app asks you to upload the dataset (CSV or Excel) and trains in seconds.
The upload is used for that session only.

**Optional, so visitors skip the upload:** train locally and commit the model files.

```bash
python train_model.py --data Car_Sales.csv
git add models/
git commit -m "Add trained model"
git push
```

The dataset itself is git-ignored because it holds customer names, IDs and income. If the saved model
was trained with a very different scikit-learn version, the app detects the load failure and falls back
to the upload-and-train screen.

## License

MIT
