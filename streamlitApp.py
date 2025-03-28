import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost
import sklearn
from PIL import Image

# Set the page configuration of the app
st.set_page_config(page_title="Timelytics", page_icon=":clock10:", layout="wide")

# Display the title and captions for the app
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times."
)

# Load the trained ensemble model from the saved pickle file
@st.cache_resource
def load_model():
    with open("./voting_model.pkl", "rb") as file:
        return pickle.load(file)

voting_model = load_model()

# Define the prediction function
def waitime_predictor(
    purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
    geolocation_state_customer, geolocation_state_seller, distance
):
    prediction = voting_model.predict(
        np.array([[
            purchase_dow, purchase_month, year, product_size_cm3,
            product_weight_g, geolocation_state_customer, geolocation_state_seller, distance
        ]])
    )
    return round(prediction[0])

# Sidebar input fields
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")
    purchase_dow = st.number_input("Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input("Geolocation State of the Customer", value=10)
    geolocation_state_seller = st.number_input("Geolocation State of the Seller", value=20)
    distance = st.number_input("Distance", value=475.35)

# Submit button
if st.button("Predict"):
    prediction = waitime_predictor(
        purchase_dow, purchase_month, year, product_size_cm3, product_weight_g,
        geolocation_state_customer, geolocation_state_seller, distance
    )
    with st.spinner("Calculating wait time..."):
        st.header("Output: Wait Time in Days")
        st.write(prediction)

# Sample dataset
st.header("Sample Dataset")
data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm^3": [37206.0, 63714, 54816],
    "Product Weight in grams": [16250.0, 7249, 9600],
    "Geolocation State Customer": [25, 25, 25],
    "Geolocation State Seller": [20, 7, 20],
    "Distance": [247.94, 250.35, 4.915]
}
df = pd.DataFrame(data)
st.write(df)
