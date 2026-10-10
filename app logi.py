
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("🚚 Delivery Delay Predictor")
st.write("Enter delivery details to predict whether the delivery will be delayed.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('logistic_model.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model file 'logistic_model.sav': {e}")
    st.stop()

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=200.0, value=25.0, step=0.1)
        traffic_congestion = st.slider("Traffic Congestion Level", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition Level", min_value=1, max_value=3, value=2, help="1: Clear, 2: Rainy/Cloudy, 3: Snowy/Stormy")
        delivery_slot = st.slider("Delivery Slot", min_value=1, max_value=3, value=2, help="1: Morning, 2: Afternoon, 3: Evening")
        driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=10)
        num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=5)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
        road_condition_score = st.slider("Road Condition Score", min_value=1, max_value=5, value=3)
        package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=200.0, value=25.0, step=0.1)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=50.0, value=12.0, step=0.1)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, max_value=500, value=60)
        
    submit_btn = st.form_submit_button("Predict Delay Status")

if submit_btn:
    # Prepare the input array in the exact feature order used during training
    input_features = pd.DataFrame([[
        delivery_distance, traffic_congestion, weather_condition,
        delivery_slot, driver_experience, num_stops, vehicle_age,
        road_condition_score, package_weight, fuel_efficiency,
        warehouse_processing_time
    ]], columns=[
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ])
    
    # Predict class and probability
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]
    
    st.markdown("--- ")
    st.subheader("Prediction Results:")
    
    if prediction == 1:
        st.error(f"🚨 **Delayed!** (Probability of delay: {probability:.2%})")
    else:
        st.success(f"✅ **On Time!** (Probability of delay: {probability:.2%})")
