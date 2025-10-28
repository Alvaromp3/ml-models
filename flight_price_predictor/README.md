# ✈️ Iberia Flight Price Predictor

A Streamlit web application that predicts flight prices for Iberia flights and visualizes flight routes on an interactive map.

## Features

- **Flight Price Prediction**: Uses CatBoost regression model to predict flight prices
- **Interactive Map**: Visualizes flight routes between Spanish cities
- **Real-time Animation**: Shows flight path with animated airplane icon
- **User-friendly Interface**: Modern, responsive design with gradient backgrounds

## Supported Cities

- Madrid (MAD)
- Barcelona (BCN)
- Valencia (VLC)
- Sevilla (SVQ)
- Bilbao (BIO)
- Palma (PMI)
- Las Palmas (LPA)
- Malaga (AGP)
- Alicante (ALC)
- Santiago (SCQ)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

## Usage

1. Select origin and destination cities
2. Choose departure date and flight details
3. Click "Predict Flight Price" to get price estimation
4. View the interactive flight route map
5. Explore different combinations to compare prices

## Data

The app can work with:
- Real CSV data from `iberia_flight_prices_2025.csv` (place in Downloads folder)
- Synthetic data generation if CSV is not available

## Model

- **Algorithm**: CatBoost Regressor
- **Features**: Origin, destination, departure date, days until departure, duration, weekend flag, season
- **Performance**: Displays R² score and Mean Absolute Error

## Technologies

- Streamlit for web interface
- Plotly for interactive maps and visualizations
- CatBoost for machine learning predictions
- Pandas/NumPy for data processing