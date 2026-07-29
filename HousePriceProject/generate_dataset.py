"""
Generates a synthetic house_price_dataset.csv matching the schema used in
train_model.ipynb:
area_sqft, bedrooms, bathrooms, balconies, age_years, floor, total_floors,
parking, location, furnishing, property_type, price

This exists because the original dataset wasn't uploaded. Replace this file's
output with your real dataset (same column names) at any time -- app.py and
train_model.py don't care how house_price_dataset.csv was produced.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

locations = ["Mumbai", "Pune", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]
# relative price multiplier per city (Mumbai/Bangalore pricier, Kolkata cheaper)
location_factor = {
    "Mumbai": 1.55, "Bangalore": 1.35, "Delhi": 1.30, "Pune": 1.10,
    "Hyderabad": 1.05, "Chennai": 1.00, "Kolkata": 0.85,
}

furnishings = ["Furnished", "Semi-Furnished", "Unfurnished"]
furnishing_factor = {"Furnished": 1.12, "Semi-Furnished": 1.05, "Unfurnished": 1.0}

property_types = ["Apartment", "Villa", "Independent House", "Studio"]
property_factor = {"Apartment": 1.0, "Villa": 1.35, "Independent House": 1.15, "Studio": 0.75}

area_sqft = np.random.randint(400, 6000, N)
bedrooms = np.random.randint(1, 6, N)
bathrooms = np.random.randint(1, 5, N)
balconies = np.random.randint(0, 4, N)
age_years = np.random.randint(0, 30, N)
total_floors = np.random.randint(1, 40, N)
floor = np.array([np.random.randint(0, tf + 1) for tf in total_floors])
parking = np.random.randint(0, 3, N)
location = np.random.choice(locations, N)
furnishing = np.random.choice(furnishings, N)
property_type = np.random.choice(property_types, N)

base_price_per_sqft = 4200  # base INR per sqft

price = (
    area_sqft * base_price_per_sqft
    + bedrooms * 350_000
    + bathrooms * 180_000
    + balconies * 60_000
    + parking * 120_000
    - age_years * 25_000
    + (floor / np.maximum(total_floors, 1)) * 400_000  # higher relative floor -> pricier
)

price = price * np.array([location_factor[l] for l in location])
price = price * np.array([furnishing_factor[f] for f in furnishing])
price = price * np.array([property_factor[p] for p in property_type])

# add noise
noise = np.random.normal(0, 0.08, N)
price = price * (1 + noise)
price = np.clip(price, 500_000, None).round(2)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "balconies": balconies,
    "age_years": age_years,
    "floor": floor,
    "total_floors": total_floors,
    "parking": parking,
    "location": location,
    "furnishing": furnishing,
    "property_type": property_type,
    "price": price,
})

df.to_csv("house_price_dataset.csv", index=False)
print(f"Generated house_price_dataset.csv with {len(df)} rows")
print(df.head())
