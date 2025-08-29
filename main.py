import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = "commodities_prices_weather.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Ensure column names are correct and match the expected structure
if len(df.columns) == 6:  # If there are exactly 6 columns (adjust if necessary)
    df.columns = ['Month', 'Commodities', 'Price', 'TAVG', 'TMIN', 'TMAX']
else:
    st.error(f"Unexpected number of columns: {len(df.columns)}. Expected 6 columns.")
    st.stop()  # Stop execution if column mismatch

# Preprocess data
df['Month'] = pd.to_datetime(df['Month'], format='%d-%m-%Y')  # Convert Month to datetime
df.sort_values(by=['Commodities', 'Month'], inplace=True)  # Sort data
commodities = df['Commodities'].unique()  # List of unique commodities

# Streamlit app setup
st.title("Commodity Price and Rainfall Forecasting")

# Commodity selection
selected_commodity = st.selectbox("Choose a Commodity", commodities)

# Filter data for the selected commodity
commodity_data = df[df['Commodities'] == selected_commodity]
commodity_data.set_index('Month', inplace=True)

# Prepare data for SARIMAX (Price Prediction)
price_data = commodity_data['Price']
exog_data = commodity_data[['TAVG', 'TMIN', 'TMAX']]

# Fit SARIMAX model for price forecasting
price_model = SARIMAX(price_data, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
price_sarimax_model = price_model.fit(disp=False)

# Forecast for the next 5 years (60 months) for price
forecast_steps = 12 * 5
future_exog = exog_data[-forecast_steps:]  # Use last available exog data
price_forecast = price_sarimax_model.get_forecast(steps=forecast_steps, exog=future_exog)
price_forecast_values = price_forecast.predicted_mean

# Prepare forecast dates
price_forecast_dates = pd.date_range(start=price_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Create forecast DataFrame for price
price_forecast_df = pd.DataFrame({'Month': price_forecast_dates, 'Forecasted_Price': price_forecast_values})
price_forecast_df.set_index('Month', inplace=True)

# Rainfall Forecast using Random Forest
# Assuming we use TAVG, TMIN, TMAX for predicting rainfall (modify this based on your data availability)
rainfall_data = commodity_data[['TAVG', 'TMIN', 'TMAX']]  # Features
rainfall_data['Rainfall'] = np.random.normal(0, 5, len(commodity_data))  # Replace with actual Rainfall column if available

# Train a Random Forest model for rainfall prediction
rainfall_model = RandomForestRegressor(random_state=42, n_estimators=100)
rainfall_model.fit(rainfall_data[['TAVG', 'TMIN', 'TMAX']], rainfall_data['Rainfall'])

# Predict rainfall for the dataset and forecast future rainfall
commodity_data['Predicted_Rainfall'] = rainfall_model.predict(rainfall_data[['TAVG', 'TMIN', 'TMAX']])

# Forecast future rainfall for the next 5 years
future_rainfall_exog = rainfall_data[['TAVG', 'TMIN', 'TMAX']].tail(forecast_steps)  # Last 60 months
rainfall_forecast = rainfall_model.predict(future_rainfall_exog)

# Prepare forecast dates for rainfall
rainfall_forecast_dates = pd.date_range(start=commodity_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Create forecast DataFrame for rainfall
rainfall_forecast_df = pd.DataFrame({'Month': rainfall_forecast_dates, 'Forecasted_Rainfall': rainfall_forecast})
rainfall_forecast_df.set_index('Month', inplace=True)

# Display the Price Forecast Table
st.write(f"### {selected_commodity} Price Forecast for Next 5 Years")
st.write(price_forecast_df)

# Plot Price Forecast
plt.figure(figsize=(12, 6))
plt.plot(price_data, label='Actual Prices')
plt.plot(price_forecast_df['Forecasted_Price'], label='Forecasted Prices', color='orange')
plt.title(f'{selected_commodity} Price Forecast')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
st.pyplot(plt)

# Display the Rainfall Forecast Table
st.write(f"### {selected_commodity} Predicted Rainfall Forecast for Next 5 Years")
st.write(rainfall_forecast_df)

# Plot Rainfall Forecast
plt.figure(figsize=(12, 6))
plt.plot(commodity_data.index, commodity_data['Predicted_Rainfall'], label='Actual Rainfall')
plt.plot(rainfall_forecast_df['Forecasted_Rainfall'], label='Forecasted Rainfall', color='green')
plt.title(f'{selected_commodity} Rainfall Forecast')
plt.xlabel('Year')
plt.ylabel('Rainfall')
plt.legend()
st.pyplot(plt)

# Identifying the months with the most rainfall forecast for the next 5 years
rainfall_forecast_month = rainfall_forecast_df.groupby(rainfall_forecast_df.index.year)['Forecasted_Rainfall'].idxmax()

# Display the Month with Most Forecasted Rainfall Year-wise for Next 5 Years
st.write(f"### Month with Most Forecasted Rainfall Year-wise for the Next 5 Years")
rainfall_forecast_max_month = rainfall_forecast_month.to_frame(name='Month')
rainfall_forecast_max_month['Max_Forecasted_Rainfall'] = rainfall_forecast_df.loc[rainfall_forecast_month]['Forecasted_Rainfall'].values
st.write(rainfall_forecast_max_month)

# Model Performance for Price and Rainfall
price_rmse = np.sqrt(((price_data - price_sarimax_model.fittedvalues) ** 2).mean())
rainfall_rmse = np.sqrt(np.mean((rainfall_data['Rainfall'] - commodity_data['Predicted_Rainfall']) ** 2))

st.write(f"### Model Performance")
st.write(f"Price Forecast RMSE: {price_rmse:.2f}")
st.write(f"Rainfall Forecast RMSE: {rainfall_rmse:.2f}")
