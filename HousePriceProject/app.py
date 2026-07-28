import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Load Dataset (for dropdowns)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "house_price_dataset.csv"))

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "house_price_model.pkl"))

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Enter the house details below and click Predict.")

# -----------------------------
# User Inputs
# -----------------------------

area_sqft = st.number_input(
    "Area (sq ft)",
    min_value=100,
    value=1200
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    value=2
)

balconies = st.number_input(
    "Balconies",
    min_value=0,
    value=1
)

age_years = st.number_input(
    "Age (Years)",
    min_value=0,
    value=5
)

floor = st.number_input(
    "Floor",
    min_value=0,
    value=1
)

total_floors = st.number_input(
    "Total Floors",
    min_value=1,
    value=5
)

# Parking (numeric)
parking = st.selectbox(
    "Parking",
    sorted(df["parking"].unique())
)

# Location
location = st.selectbox(
    "Location",
    sorted(df["location"].unique())
)

# Furnishing
furnishing = st.selectbox(
    "Furnishing",
    sorted(df["furnishing"].unique())
)

# Property Type
property_type = st.selectbox(
    "Property Type",
    sorted(df["property_type"].unique())
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "area_sqft": [area_sqft],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "balconies": [balconies],
        "age_years": [age_years],
        "floor": [floor],
        "total_floors": [total_floors],
        "parking": [parking],
        "location": [location],
        "furnishing": [furnishing],
        "property_type": [property_type]
    })

    prediction = model.predict(input_data)

    st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")