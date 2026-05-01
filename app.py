
import pandas as pd
import streamlit as st
import joblib

# Load files
model = joblib.load("tuned_cost_model.joblib")
scaler = joblib.load("scaler.joblib")
columns = joblib.load("columns.joblib")

st.set_page_config(page_title="Logistics Cost Prediction", page_icon="📦")
st.title("📦 Logistics Cost Prediction")

st.write("Enter shipment details:")

# USER INPUT (Streamlit UI)
weight = st.number_input("Enter Weight (kg)", min_value=0.0, value=1.0)

carrier = st.selectbox(
    "Select Carrier",
    ["Amazon Logistics", "FedEx", "UPS", "DHL", "LaserShip", "USPS", "OnTrac"]
)

origin = st.selectbox(
    "Select Origin Warehouse",
    ["Warehouse_ATL","Warehouse_BOS","Warehouse_CHI","Warehouse_DEN",
     "Warehouse_HOU","Warehouse_LA","Warehouse_MIA","Warehouse_NYC",
     "Warehouse_SEA","Warehouse_SF"]
)

destination = st.selectbox(
    "Select Destination",
    ["Atlanta","Boston","Chicago","Dallas","Denver","Detroit","Houston",
     "Los Angeles","Miami","Minneapolis","New York","Phoenix",
     "Portland","San Francisco","Seattle"]
)

duration = st.slider("Transit Days", 1, 10, 3)

# Predict button
if st.button("Predict Cost"):

    # Create DataFrame
    new_data = pd.DataFrame([{
        'Weight_kg': weight,
        'Carrier': carrier,
        'Origin_Warehouse': origin,
        'Destination': destination,
        'Transit_Days': duration
    }])

    # One-hot encoding
    new_data = pd.get_dummies(new_data)

    # Load saved training columns
    columns = joblib.load("columns.joblib")

    # Match columns
    new_data = new_data.reindex(columns=columns, fill_value=0)

    # Load model + scaler
    model = joblib.load("tuned_cost_model.joblib")
    scaler = joblib.load("scaler.joblib")

    # Scale
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = model.predict(new_data_scaled)

    st.success(f"💰 Estimated Cost: $ {round(prediction[0], 2)}")
