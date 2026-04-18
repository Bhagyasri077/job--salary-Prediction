import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("salary_data.csv")

X = df[["YearsExperience"]]
y = df["Salary"]

model = LinearRegression()
model.fit(X, y)

# SAVE MODEL (important line)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")