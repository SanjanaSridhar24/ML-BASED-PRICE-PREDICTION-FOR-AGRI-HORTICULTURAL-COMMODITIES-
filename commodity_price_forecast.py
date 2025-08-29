import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sklearn.metrics import mean_squared_error

# Define default credentials
USERNAME = "admin"
PASSWORD = "admin"



# Login function


def render_home():
    
    st.markdown("""
    <style>
        body {
            background-color:rgb(3, 59, 76);  /* Dark background */
            color: white;  /* Light text for contrast */
        }
        .stApp {
            background-color:rgb(3, 59, 76);  /* Dark background for the whole app */
        }
        .css-1d391kg {  /* Target Streamlit header for dark color */
            color: white;
        }
        .stButton>button {
            background-color: #444;  /* Dark button background */
            color: white;
        }
        .stSelectbox>div {
            background-color: #333;  /* Dark background for selectbox */
            color: white;
        }
    </style>
""", unsafe_allow_html=True)
    st.title("🌾 Commodity Price and Rainfall Forecasting 🌦️")
    st.image("crop.jpeg", caption="Welcome to the Forecasting App", use_container_width=True)

    # Home page introduction
    st.write(""" 
    This application uses advanced models to forecast commodity prices and rainfall trends over the next 5 years. 
    By selecting a commodity, you can view its price prediction using the SARIMAX model, while rainfall predictions are generated using Random Forest techniques. 
    Make informed decisions for agricultural planning and production based on the insights provided.
    """)

    # Features Section with Icons
    st.subheader("Features")
    
    features = [
        ("📈", "Predict commodity prices using advanced SARIMAX models."),
        ("🌧️", "Forecast rainfall with machine learning techniques like Random Forest."),
        ("📊", "Analyze trends for the next 5 years based on historical data."),
        ("👩‍💻", "User-friendly interface for selecting commodities and viewing predictions.")
    ]

    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 40px; margin-right: 15px;">{feature[0]}</span>
                    <span style="font-size: 16px;">{feature[1]}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Commodity Cards with Three Images
    st.subheader("Featured Commodities")
    
    commodities = {
        "Wheat": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuc1EEYvIQKkJetgbgNprmdE4w4STLdLnFkw&s",
            "description": "Wheat is a staple food in many countries and is used in bread, pasta, and other food products. It's an important crop for food security."
        },
        "Rice": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRY_38OB-O3ct4_WTA4CLOW7rpDmuU8RkDVsQ&s",
            "description": "Rice is a crucial crop in many Asian countries, providing a significant portion of the daily calorie intake for billions of people."
        },
        "Corn": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTOtOEMuLtSg6UmWO-hIQdy4H1eGQ34nv3Eg&s",
            "description": "Corn is widely grown across the world and is used for food, livestock feed, and biofuel production."
        },
        "Tomato": {
            "image": "https://plantlane.com/cdn/shop/articles/closeup-photo-male-hand-picking_1100x.jpg?v=1691675288",
            "description": "Tomato is a versatile crop, widely used in culinary dishes and is a major source of vitamins A and C."
        },
        "Oil": {
            "image": "https://yarrowpharm.com/wp-content/uploads/2024/05/Vegetable-Oil.jpeg",
            "description": "Tomato is a versatile crop, widely used in culinary dishes and is a major source of vitamins A and C."
        },
        "Gram Dal": {
            "image": "https://homedelivery.ramachandran.in/media/catalog/product/cache/04c5c5c4276fe9dba74400abc896c29c/4/8/481314A001005_Fantastic_Roasted-Garmdal-Broken.jpg",
            "description": "Tomato is a versatile crop, widely used in culinary dishes and is a major source of vitamins A and C."
        },
        
    }

    # Creating a 2x2 grid layout for commodities
    cols = st.columns(3)  # Update to 3 columns for 3 images
    for i, (commodity, details) in enumerate(commodities.items()):
        with cols[i % 3]:  # Distribute commodities in 3 columns
            st.markdown(f"""
            <div class="card" style="position: relative; width: 100%; height: 100%; padding: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; text-align: center;">
                <img src="{details['image']}" style="width: 100%; height: 150px; object-fit: cover; transition: transform 0.3s ease;" class="card-img"/>
                <div class="card-body" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 8px;">
                    <h4>{commodity}</h4>
                    <p>{details['description']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Adding CSS for hover effect
    st.markdown("""
    <style>
        .card:hover .card-img {
            transform: rotateY(180deg);
        }
        .card:hover .card-body {
            display: block;
        }
        .card {
            position: relative;
        }
        .card-img {
            transition: transform 0.3s ease;
        }
        .card-body {
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_about():
    st.title("About This App")
    
    st.write(""" 
    This application is designed to provide insights into agricultural commodities and rainfall patterns. 
    By leveraging advanced statistical and machine learning models, users can:
    - 📊 Analyze price trends of commodities like wheat, rice, and more.
    - 🌦️ Forecast rainfall based on historical weather data.
    - 📈 Make data-driven decisions for better agricultural planning.
    """)
    
    # Commodity images and descriptions
    st.subheader("Featured Commodities")
    
    commodities = {
        "Wheat": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuc1EEYvIQKkJetgbgNprmdE4w4STLdLnFkw&s",
            "description": "Wheat is a staple food in many countries and is used in bread, pasta, and other food products. It's an important crop for food security."
        },
        "Rice": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRY_38OB-O3ct4_WTA4CLOW7rpDmuU8RkDVsQ&s",
            "description": "Rice is a crucial crop in many Asian countries, providing a significant portion of the daily calorie intake for billions of people."
        },
        "Corn": {
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTTOtOEMuLtSg6UmWO-hIQdy4H1eGQ34nv3Eg&s",
            "description": "Corn is widely grown across the world and is used for food, livestock feed, and biofuel production."
        },
    }
    
    for commodity, details in commodities.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(details["image"], caption=commodity, use_container_width=True)
        with col2:
            st.write(f"### {commodity}")
            st.write(details["description"])

    st.subheader("Technologies Used")
    st.write(""" 
    - **SARIMAX**: A statistical model for price prediction, considering seasonal variations.
    - **Random Forest**: An ensemble learning method for rainfall forecasting.
    - **Streamlit**: A modern web application framework for an interactive user experience.
    """)

def render_prediction():
    st.title("Prediction Page")
    file_path = "commodities_prices_weather.csv"  # Update with your file path
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure the file is available.")
        return

    if len(df.columns) == 6:
        df.columns = ['Month', 'Commodities', 'Price', 'TAVG', 'TMIN', 'TMAX']
    else:
        st.error(f"Unexpected number of columns: {len(df.columns)}. Expected 6 columns.")
        return

    df['Month'] = pd.to_datetime(df['Month'], format='%d-%m-%Y')
    df.sort_values(by=['Commodities', 'Month'], inplace=True)
    commodities = df['Commodities'].unique()

    selected_commodity = st.selectbox("Choose a Commodity", commodities)
    commodity_data = df[df['Commodities'] == selected_commodity]
    commodity_data.set_index('Month', inplace=True)

    # Price Forecast using SARIMAX
    price_data = commodity_data['Price']
    exog_data = commodity_data[['TAVG', 'TMIN', 'TMAX']]

    price_model = SARIMAX(price_data, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    price_sarimax_model = price_model.fit(disp=False)

    forecast_steps = 12 * 5
    future_exog = exog_data[-forecast_steps:]
    price_forecast = price_sarimax_model.get_forecast(steps=forecast_steps, exog=future_exog)
    price_forecast_values = price_forecast.predicted_mean
    price_forecast_dates = pd.date_range(start=price_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

    price_forecast_df = pd.DataFrame({'Month': price_forecast_dates, 'Forecasted_Price': price_forecast_values})
    price_forecast_df.set_index('Month', inplace=True)

    # Plotting the Price Forecast
    st.write(f"### {selected_commodity} Price Forecast for Next 5 Years")
    st.write(price_forecast_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_data, label='Actual Prices')
    ax.plot(price_forecast_df['Forecasted_Price'], label='Forecasted Prices', color='orange')
    ax.set_title(f'{selected_commodity} Price Forecast')
    ax.set_xlabel('Year')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Model performance for Price Forecast (RMSE)
    price_rmse = sqrt(mean_squared_error(price_data[-forecast_steps:], price_forecast_values))
    st.write(f"Price Forecast RMSE: {price_rmse:.2f}")

    # Rainfall Forecast using RandomForestRegressor
    rainfall_data = commodity_data[['TAVG', 'TMIN', 'TMAX']]
    rainfall_data['Rainfall'] = np.random.normal(0, 5, len(commodity_data))  # Simulating rainfall

    rainfall_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rainfall_model.fit(rainfall_data[['TAVG', 'TMIN', 'TMAX']], rainfall_data['Rainfall'])

    future_rainfall_exog = rainfall_data[['TAVG', 'TMIN', 'TMAX']].tail(forecast_steps)
    rainfall_forecast = rainfall_model.predict(future_rainfall_exog)
    rainfall_forecast_dates = pd.date_range(start=commodity_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

    rainfall_forecast_df = pd.DataFrame({'Month': rainfall_forecast_dates, 'Forecasted_Rainfall': rainfall_forecast})
    rainfall_forecast_df.set_index('Month', inplace=True)

    # Plotting the Rainfall Forecast
    st.write(f"### {selected_commodity} Rainfall Forecast for Next 5 Years")
    st.write(rainfall_forecast_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rainfall_forecast_df['Forecasted_Rainfall'], label='Forecasted Rainfall', color='blue')
    ax.set_title(f'{selected_commodity} Rainfall Forecast')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rainfall')
    ax.legend()
    st.pyplot(fig)

    # Model performance for Rainfall Forecast (RMSE)
    

    # Display Month with Most Forecasted Rainfall
    rainfall_forecast_df['Year'] = rainfall_forecast_df.index.year
    rainfall_forecast_df['Month_Name'] = rainfall_forecast_df.index.month_name()

    # Find the month with the most forecasted rainfall for each year
    max_rainfall_month = rainfall_forecast_df.groupby('Year').apply(lambda x: x.loc[x['Forecasted_Rainfall'].idxmax()])
    
    st.write(f"### Month with Most Forecasted Rainfall Year-wise")
    st.write(max_rainfall_month[['Year', 'Month_Name', 'Forecasted_Rainfall']])
    price_rmse = sqrt(mean_squared_error(price_data[-forecast_steps:], price_forecast_values))
    st.write(f"Price Forecast RMSE: {price_rmse:.2f}")
    rainfall_rmse = sqrt(mean_squared_error(rainfall_data['Rainfall'][-forecast_steps:], rainfall_forecast))
    st.write(f"Rainfall Forecast RMSE: {rainfall_rmse:.2f}")

def render_login():
    st.title("Login to Your Dashboard")
    # Display the login form
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        login_button = st.form_submit_button("Login")

        # Validate credentials
        if login_button:
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")


# New Functions for Contact and FAQs
def render_contact():
    # Page title with black color
    st.markdown(
        """
        <h1 style="color: black; text-align: center;">📞 Contact Us</h1>
        """,
        unsafe_allow_html=True
    )

    # Add a description
    st.markdown("""
        <div style="text-align: center; font-size: 18px; margin-bottom: 20px; color: #333;">
            <p>We are here to assist you with any queries or feedback. Please feel free to reach out to us!</p>
        </div>
    """, unsafe_allow_html=True)

    # Contact form
    with st.form("contact_form", clear_on_submit=True):
        # User input fields
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name", placeholder="Enter your full name")
        with col2:
            email = st.text_input("Your Email", placeholder="Enter your email address")

        subject = st.text_input("Subject", placeholder="Enter the subject of your message")
        message = st.text_area("Message", placeholder="Write your message here...", height=150)

        # Submit button
        submitted = st.form_submit_button("Submit")
        if submitted:
            if name and email and message:
                st.success("Thank you for reaching out! We will get back to you soon.")
            else:
                st.error("Please fill out all the fields before submitting.")

    # Contact details section
    st.markdown("""
        <div style="margin-top: 40px; text-align: center;">
            <h3>📍 Our Office</h3>
            <p>123 Agricultural Insights Lane, CropCity, AG 12345</p>
            <h3>📧 Email Us</h3>
            <p>support@agricommodities.com</p>
            <h3>📞 Call Us</h3>
            <p>+1-800-555-FORECAST</p>
        </div>
    """, unsafe_allow_html=True)

    # Add a background image for the contact section
    st.markdown("""
        <style>
            .stApp {
                background-image: url("https://cdn.pixabay.com/photo/2017/06/01/20/47/office-2361603_960_720.jpg");
                background-size: cover;
                background-attachment: fixed;
                color: white;
            }
            .st-form-container {
                background-color: rgba(41, 2, 2, 0.8);
                border-radius: 10px;
                padding: 20px;
                margin: auto;
                width: 50%;
            }
            h3, p {
                color: #000;
            }
        </style>
    """, unsafe_allow_html=True)


def render_faq():
    st.title("📋 Frequently Asked Questions (FAQ)")
    st.markdown("---")  # Horizontal line for better separation

    faq_list = [
        {
            "question": "💡 What is this app about?",
            "answer": "This app forecasts commodity prices and rainfall trends using advanced machine learning models."
        },
        {
            "question": "📊 How do I select a commodity?",
            "answer": "Navigate to the Prediction page and use the dropdown menu to select a commodity. Enter the required details to view forecasts."
        },
        {
            "question": "📁 Can I add my own data?",
            "answer": "Currently, the app supports predefined datasets. Custom data upload will be available in future updates."
        },
        {
            "question": "🔒 Is my data secure?",
            "answer": "Yes, the app does not store any personal data and uses secure protocols for processing."
        },
        {
            "question": "🛠️ Who can I contact for support?",
            "answer": "You can contact the developer through the 'About' section for any queries or feedback."
        }
    ]

    for faq in faq_list:
        with st.expander(faq["question"]):  # Use expander for collapsible questions
            st.write(faq["answer"])

    # Add a call-to-action or support link
    st.markdown("---")
    st.info("Still have questions? Feel free to [contact us] for more support.")



# Sidebar for navigation
st.markdown("""
    <style>
        /* Sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #80EF80; /* Light green background for sidebar */
        }

        /* Sidebar title and logo alignment */
        [data-testid="stSidebar"] img {
            margin: 0 auto; /* Center align logo */
            display: block;
        }

        /* Sidebar text alignment and styling */
        [data-testid="stSidebar"] .css-1d391kg {
            color: black; /* Sidebar text color */
        }

        /* Sidebar headings styling */
        [data-testid="stSidebar"] h3 {
            color: #006600; /* Dark green for the title */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar elements
st.sidebar.image(
    "logo.png",  # Replace with your logo path
    width=150  # Adjust the width to make the logo smaller
)
st.sidebar.markdown(
    "<h3 style='font-size: 30px; font-weight: bold; text-align: center;'>AGRITECH</h3>", 
    unsafe_allow_html=True
)
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "About", "Prediction", "Login", "Contact", "FAQ"])

# Show pages based on sidebar selection
if selected_page == "Home":
    render_home()
elif selected_page == "About":
    render_about()
elif selected_page == "Login":
    render_login()
elif selected_page == "Contact":
    render_contact()
elif selected_page == "FAQ":
    render_faq()
elif selected_page == "Prediction":
    if st.session_state.get("logged_in", False):
        render_prediction()
    else:
        st.warning("Please log in to access the Prediction page.")

# Background CSS for the main page content
st.markdown("""
    <style>
        .stApp { background-color:rgb(212, 238, 216); } /* Light yellow background for main app */
    </style>
""", unsafe_allow_html=True)
