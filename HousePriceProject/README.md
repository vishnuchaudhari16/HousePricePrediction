# 🏠 HousePredict AI

AI-powered house price prediction web app built with Streamlit, backed by a
scikit-learn `RandomForestRegressor` pipeline (StandardScaler + OneHotEncoder).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit application (Home, Predict, About, Model, Contact) |
| `train_model.py` | Trains the model from `house_price_dataset.csv`, saves `house_price_model.pkl` + metrics |
| `generate_dataset.py` | Generates a synthetic dataset with the same schema as your notebook (only needed if you don't have real data yet) |
| `house_price_model.pkl` | The trained pipeline (loaded by the app) |
| `metrics.json` / `feature_importance.json` / `metadata.json` | Artifacts consumed by the app for metrics, charts, and dynamic form ranges |
| `house_price_dataset.csv` | Training data |

## Using your real dataset

This project shipped with a **synthetic** dataset because the original
`house_price_dataset.csv` from your notebook wasn't included in the upload.
To use your real data instead:

1. Replace `house_price_dataset.csv` with your real file — same column names:
   `area_sqft, bedrooms, bathrooms, balconies, age_years, floor, total_floors,
   parking, location, furnishing, property_type, price`
2. Re-run training:
   ```bash
   python3 train_model.py
   ```
   This regenerates `house_price_model.pkl`, `metrics.json`,
   `feature_importance.json`, and `metadata.json` — the app picks up your
   real model and stats automatically, no code changes needed.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes on scope

The original spec asked for `streamlit-lottie`, `streamlit-shadcn-ui`, and
`streamlit-card`. Those either require an internet connection to fetch
animation JSON at runtime or are less actively maintained, so this build
reproduces the same visual language (glassmorphism, gradients, floating
icons, hover glow, fade/slide/zoom animations) with custom CSS instead —
same look, no extra runtime dependencies. `streamlit-option-menu` is used
for the sticky, animated navbar.

Everything else from the spec is implemented: hero landing page, prediction
form with icons and validation, animated result card with PDF/CSV download,
sidebar with live model metrics, About/Model/Contact pages, dark mode toggle,
prediction history table with clear button, and interactive Plotly charts
(price distribution, feature importance, area vs price, location
distribution, predicted vs actual).
