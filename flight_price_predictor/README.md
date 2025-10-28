# Iberia Flight Price Predictor

## Description
Streamlit web application that predicts Iberia flight prices using CatBoost Regressor with GridSearch optimization and visualizes flight routes on interactive maps.

**Developed by:** Alvaro Martin-Pena

## Motivation
Help travelers predict and compare flight prices across different routes and dates, making travel planning more efficient and cost-effective.

## Features
- **Advanced prediction**: CatBoost regression model with GridSearch optimization
- **Interactive maps**: Plotly visualizations showing flight routes
- **Real-time animation**: Animated flight paths
- **Modern UI**: Professional design with gradient backgrounds
- **Multiple cities**: Support for Spanish cities and international destinations
- **Dynamic demand calculation**: Considers seasonality, weekends, and holidays

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare dataset:
```bash
# Place iberia_flight_prices_2025.csv in the project directory
# Or set environment variable:
export FLIGHT_DATASET_PATH=/path/to/your/dataset.csv
```

3. Run the application:
```bash
streamlit run app.py
```

Or using the run script:
```bash
python run_app.py
```

## Usage
1. Select origin and destination cities
2. Choose departure date and flight details (cabin class, stops, time band)
3. Click "Predict Flight Price" to get price estimation
4. View the interactive flight route map
5. Explore different combinations to compare prices

## Supported Cities
- Madrid (MAD), Barcelona (BCN)
- Valencia (VLC), Sevilla (SVQ)
- Bilbao (BIO), Palma (PMI)
- Las Palmas (LPA), Malaga (AGP)
- Alicante (ALC), Santiago (SCQ)
- International: Lisbon, London, NYC, SF, Bogotá, Miami, São Paulo, and more

## Model Architecture

### Algorithm
- **Model**: CatBoost Regressor
- **Optimization**: GridSearchCV with 5-fold cross-validation
- **Preprocessing**: StandardScaler + OneHotEncoder

### Hyperparameters
- **Iterations**: 300, 500
- **Depth**: 6, 8
- **Learning Rate**: 0.03, 0.05
- **L2 Regularization**: 3, 5

### Features
- Origin, destination
- Distance (calculated)
- Day of week, month
- Demand factor (calculated based on multiple factors)
- Cabin class, stops, time band

### Expected Performance
- **R² Score**: Displays actual score in sidebar
- **Mean Absolute Error**: Shown in interface
- **Real-time predictions**: < 1 second

## Project Structure
```
flight_price_predictor/
├── app.py                          # Streamlit application
├── run_app.py                      # Launch script
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # Documentation
└── iberia_flight_prices_2025.csv  # Dataset (required - not included)
```

## Technical Details

### Dynamic Demand Calculation
The app calculates a demand factor based on:
- Route popularity
- Seasonal adjustments (summer peak, Christmas)
- Day of week impact
- Distance factor
- Cabin class premium
- Holiday periods

### Distance Calculation
Real distance between airports using Haversine formula for accurate geo-distance calculation.

## Author
**Alvaro Martin-Pena**
- Machine Learning Engineer
- Data Scientist

## License
For educational and personal use.
