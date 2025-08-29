import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# Load your dataset
df = pd.read_csv("data/data.csv")

# Ensure proper formatting
df["Month"] = pd.to_datetime(df["Month"], format='%d-%m-%Y', errors='coerce')
df = df.set_index("Month").sort_index()

# Train-Test Split
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

# Train SARIMAX Model on "Price" column
target_column = "Price"
model = SARIMAX(train_data[target_column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

# Save the trained model to a file
joblib.dump(results, "sarimax_model.pkl")

print("Model training complete and saved to 'sarimax_model.pkl'.")
