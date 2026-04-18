
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("💼 Job Salary Prediction App")

st.write("Predict salary based on years of experience")

# User Input
exp = st.slider("Years of Experience", 0.0, 20.0, 1.0)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[exp]]))
    st.success(f"💰 Predicted Salary: ₹{prediction[0]:,.2f}")

# Visualization
st.subheader("📊 Data Visualization")

try:
    df = pd.read_csv("salary_data.csv")

    fig, ax = plt.subplots()
    ax.scatter(df["YearsExperience"], df["Salary"], color="blue")
    ax.plot(df["YearsExperience"], model.predict(df[["YearsExperience"]]), color="red")

    ax.set_xlabel("Experience")
    ax.set_ylabel("Salary")

    st.pyplot(fig)

except:
    st.warning("Dataset not found for visualization")