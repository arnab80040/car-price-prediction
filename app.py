import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

df = pd.read_csv("car data.csv")
df.drop_duplicates(inplace=True)
df.drop(86, axis=0, inplace=True)
data = df.drop(columns="Car_Name")

d1 = {"Petrol":int(0), "Diesel":int(1)}
d2 = {"Dealer":2, "Individual":3}
d3 = {"Manual":4, "Automatic":5}

data["Fuel_Type"] = data["Fuel_Type"].map(d1)
data["Selling_type"] = data["Selling_type"].map(d2)
data["Transmission"] = data["Transmission"].map(d3)

data.dropna(inplace=True)

X = data.drop(columns="Selling_Price")
y = data["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model1 = LinearRegression()
model1.fit(X_train, y_train)


def predictor_fn(year, present_price, distance_trav, fuel_type, selling_type, transmission, owner):
    if fuel_type == "Petrol":
        fuel_type = 0
    elif fuel_type == "Diesel":
        fuel_type = 1

    if selling_type == "Dealer":
        selling_type = 2
    elif selling_type == "Individual":
        selling_type = 3

    if transmission == "Manual":
        transmission = 4
    elif transmission == "Automatic":
        transmission = 5
    return model1.predict(np.array([year, present_price, distance_trav, fuel_type, selling_type, transmission, owner]).reshape(1, -1))[0]


fuel_types = ["Diesel", "Petrol"]
selling_types = ["Individual", "Dealer"]
transmission_types = ["Manual", "Automatic"]

st.markdown("<h1 style='text-align: center; color: black;'>Car price predictor</h1>", unsafe_allow_html=True)
input_car_name = st.text_input("Enter the name of car")
input_year = st.number_input("Enter the model year")
present_price = st.number_input("Enter the current market price in *lac rupees")
dist_trav = st.number_input("Enter the distance travelled in Kms")
fuel_type = st.selectbox("Select the fuel type", fuel_types)
selling_type = st.selectbox("Select the selling type", selling_types)
transmission = st.selectbox("Select the transmission type", transmission_types)
ownership = st.number_input("Enter owner")



if st.button("Predict"):
    st.header("Predicted selling price of " + input_car_name + ":")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.header("Rs " + str(round(predictor_fn(input_year, present_price, dist_trav, fuel_type, selling_type, transmission, ownership), 2)) + " lac")

    with col3:
        st.write(' ')