"""
Trains the House Price Prediction model using the same pipeline structure
as train_model.ipynb (StandardScaler + OneHotEncoder -> RandomForestRegressor),
and additionally persists:
  - house_price_model.pkl   (the trained pipeline)
  - metrics.json            (MAE, RMSE, R2, dataset size, column info)
  - feature_importance.json (for the "Feature Importance" chart in the app)

Run this after generate_dataset.py (or after replacing house_price_dataset.csv
with your real data).
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load Dataset
df = pd.read_csv("house_price_dataset.csv")
print(df.head())

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Numerical and Categorical Columns
num_cols = list(X.select_dtypes(include=["int64", "float64"]).columns)
cat_cols = list(X.select_dtypes(include=["object"]).columns)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train Model
pipeline.fit(X_train, y_train)
print("Model Trained Successfully!")

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# Save Model
joblib.dump(pipeline, "house_price_model.pkl")
print("\nModel Saved Successfully!")

# ---- Extra artifacts for the Streamlit app ----

# Feature importance (map one-hot encoded names back to readable labels)
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_feature_names = num_cols + cat_feature_names
importances = pipeline.named_steps["model"].feature_importances_

# aggregate one-hot importances back to their original column
agg_importance = {c: 0.0 for c in num_cols + cat_cols}
for name, imp in zip(all_feature_names, importances):
    if name in agg_importance:
        agg_importance[name] += float(imp)
    else:
        # one-hot column like "location_Mumbai" -> "location"
        for c in cat_cols:
            if name.startswith(c + "_"):
                agg_importance[c] += float(imp)
                break

with open("feature_importance.json", "w") as f:
    json.dump(agg_importance, f, indent=2)

# Column metadata (used to build form dropdowns dynamically)
metadata = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
    "ranges": {
        c: {"min": int(df[c].min()), "max": int(df[c].max()), "median": int(df[c].median())}
        for c in num_cols
    },
    "n_rows": len(df),
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Metrics
metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "n_rows": len(df)}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved metrics.json, feature_importance.json, metadata.json")
