import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import streamlit as st
import base64

# Function to add background image from local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add your background
add_bg_from_local("home.jpg")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("house.csv")
    df = df[df.price > 0]
    df = df[df.price <= df.price.quantile(0.99)]
    return df

# Train model if not already
@st.cache_resource
def train_model(df):
    num_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_above', 'sqft_basement', 'floors', 'view', 'condition']
    cat_cols = ['statezip']
    X_num = df[num_cols]
    X_cat = df[cat_cols]

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = ohe.fit_transform(X_cat)

    X = np.hstack([X_num_scaled, X_cat_encoded])
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, ohe, num_cols, cat_cols

# Load Data and Train Model
df = load_data()
model, scaler, ohe, num_cols, cat_cols = train_model(df)

# Streamlit UI
st.title("🏡 House Price Prediction App")

st.sidebar.header("Input House Details")

input_data = {}
for col in num_cols:
    val = st.sidebar.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    input_data[col] = val

statezip = st.sidebar.selectbox("State ZIP", sorted(df['statezip'].unique()))

# Submit button
if st.sidebar.button("Submit"):

  # Prepare input for prediction
  input_df = pd.DataFrame([input_data])
  input_num_scaled = scaler.transform(input_df)
  input_cat_encoded = ohe.transform([[statezip]])
  X_input = np.hstack([input_num_scaled, input_cat_encoded])

# Make prediction
  prediction = model.predict(X_input)[0]

  st.subheader("📊 Predicted House Price")
  st.success(f"${prediction:,.0f}")
