import pandas as pd
import streamlit as st
import pickle

# Loading the data, encoders, scaler and model
data = pd.read_csv("Car_Price_Data.csv")
binary_encoder = pickle.load(open("Binary_encoder.pkl", "rb"))
one_hot_encoder = pickle.load(open("OneHot_encoder.pkl", "rb"))
scaler = pickle.load(open("StandardScaler.pkl", "rb"))
model = pickle.load(open("Car_Price_Model.pkl", "rb"))

# Putting categorical items in a list
names = data["name"].unique().tolist()
fuels = data["fuel"].unique().tolist()
sellers = data["seller_type"].unique().tolist()
transmissions = data["transmission"].unique().tolist()
owners = data["owner"].unique().tolist()
seaters = data["seats"].astype(int).unique().tolist()

# Function to preprocess input data
def preprocessing (input_data):
    # Feature Engineering to create engine_power column from torque and rpm
    def eng_power(torque_value, rpm_value):
        return (torque_value * rpm_value) / 7127 # 7127 is the conversion factor for Nm and rpm to hp.

    # Applying the function to each row in the DataFrame and creating the new engine_power_hp column
    input_data["engine_power(hp)"] = input_data.apply(lambda row: eng_power(row["torque(Nm)"], row["rpm"]), axis = 1)

    # Dropping torque and rpm values
    input_data = input_data.drop(["torque(Nm)", "rpm"], axis = 1)

    # Fitting the encoder to the data
    input_data_encoded = binary_encoder.transform(input_data)
    input_data_encoded = one_hot_encoder.transform(input_data_encoded)

    # Dropping first columns of one_hot_encoded columns to avoid dummy variable trap
    input_data_encoded.drop(["fuel_1", "seller_type_1", "transmission_1", "owner_1"], axis = 1, inplace = True)

    # Scaling the data
    input_data_scaled = pd.DataFrame(scaler.transform(input_data_encoded), columns = input_data_encoded.columns)

    return input_data_scaled

# Building Streamlit
def main():
    # Giving the streamlit app a title
    st.title("Car Price Prediction Model")

    # Creating two columns
    col1, col2 = st.columns(2)
    # Getting user input for first column
    with col1:
        name = st.selectbox("Car Brand", options = names)
        year = st.number_input("Year of Manufacture", min_value = 1980)
        km_driven = st.number_input("Kilometres Driven", step = 1)
        fuel = st.selectbox("Fuel", options=fuels)
        seller_type = st.selectbox("Seller Type", options=sellers)
        transmission = st.selectbox("Transmission", options= transmissions)
        owner = st.selectbox("Owner", options=owners)
    # Getting user input for second column
    with col2:
        mileage = st.number_input("Mileage(kmpl)", format="%.2f")
        engine = st.number_input("Engine Capacity(CC)", format = "%.2f")
        max_power = st.number_input("Maximum Power(bhp)", format = "%.2f")
        seats = st.selectbox("Number of Seats", options = seaters)
        torque = st.number_input("Torque(Nm)", format="%.2f")
        rpm = st.number_input("Rotational Speed(rpm)", format = "%.2f")

    # Creating input data
    input_data = pd.DataFrame({
        "name":[name],
        "year":[year],
        "km_driven":[km_driven],
        "fuel":[fuel],
        "seller_type":[seller_type],
        "transmission":[transmission],
        "owner":[owner],
        "mileage(kmpl)":[mileage],
        "engine(CC)":[engine],
        "max_power(bhp)":[max_power],
        "seats":[seats],
        "torque(Nm)":[torque],
        "rpm":[rpm]
    })

    # Button for prediction
    if st.button("Predict"):   
        # Preprocessing input data
        processed_data = preprocessing(input_data)

        # Making predictions using trained model
        prediction = model.predict(processed_data)

        # Displaying Prediction
        st.write(f"The Price of this car is around ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()