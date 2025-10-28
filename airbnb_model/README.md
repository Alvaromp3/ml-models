# Airbnb Price Predictor

## Description
Flask and Streamlit web applications to predict Airbnb property prices using an XGBoost Regressor with PCA and feature selection.

**Developed by:** Alvaro Martin-Pena

## Motivation
Predict rental prices for Airbnb properties to help hosts set competitive prices and travelers budget effectively.

## Features
- **Advanced model**: XGBoost Regressor with PCA and SelectKBest
- **Dual interfaces**: Flask and Streamlit web apps
- **Real-time prediction**: Instant price estimates
- **Feature importance**: Model provides insights on price factors
- **High accuracy**: R² > 0.90

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:
```bash
export AIRBNB_DATASET_PATH=/path/to/your/dataset.csv
```

Or place `airbnb_synthetic.csv` in the project directory.

3. Run the application:

**Streamlit (Recommended):**
```bash
streamlit run app_streamlit.py
```

**Flask:**
```bash
python run_app.py
```

Or directly:
```bash
python app.py
```

## Usage

1. Open your browser:
   - Streamlit: `http://localhost:8501`
   - Flask: `http://localhost:5000`
2. Fill in the property details:
   - **Latitude/Longitude**: Location coordinates
   - **Property Type**: Apartment, House, Condominium, Loft
   - **Room Type**: Entire home/apt, Private room, Shared room
   - **Bedrooms**: Number of bedrooms
   - **Bathrooms**: Number of bathrooms
   - **Reviews**: Number of reviews
   - **Availability**: Days available per year
3. Click "Predict Price" or "Analyze Performance"
4. Get the price prediction and metrics

## Model Architecture

### Algorithm
- **Model**: XGBoost Regressor
- **Estimators**: 600
- **Max Depth**: 12
- **Learning Rate**: 0.03

### Preprocessing Pipeline
1. **StandardScaler**: Normalize numerical features
2. **SelectKBest**: Select 6 best features (f_regression)
3. **PCA**: Reduce to 5 principal components
4. **XGBoost**: Gradient boosting regressor

### Expected Performance
- **R² Score**: > 0.90 (Excellent performance)
- **Mean Absolute Error**: Low price deviation
- **RMSE**: Minimal prediction error

### Input Features
- `latitude`, `longitude`
- `bedrooms`, `bathrooms`
- `number_of_reviews`
- `availability_365`
- `property_type` (encoded)
- `room_type` (encoded)

## Project Structure
```
airbnb_model/
├── app.py                  # Flask application
├── app_streamlit.py        # Streamlit application (recommended)
├── modelo_airbnb.py        # ML model class
├── templates/
│   └── index.html          # Flask web interface
├── airbnb_synthetic.csv    # Training dataset
├── run_app.py             # Flask launch script
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore rules
└── README.md               # Documentation
```

## Technical Improvements

### Model Evolution
- **v1**: KNeighborsClassifier (Wrong approach, R² ≈ 0.36)
- **v2**: RandomForestRegressor (R² ≈ 0.75-0.85)
- **v3**: XGBoost with PCA + SelectKBest (R² > 0.90)

### Key Improvements
- Correct regression model (not classifier)
- Advanced feature engineering (PCA + SelectKBest)
- Optimized XGBoost hyperparameters
- Environment variable support for dataset path
- Professional error handling
- Dual web interfaces (Flask + Streamlit)
- English documentation

### Performance Optimization
- Feature selection reduces overfitting
- PCA improves generalization
- XGBoost provides industry-leading accuracy
- Regularization prevents overfitting

## Author
**Alvaro Martin-Pena**
- Machine Learning Engineer
- Data Scientist

## License
For educational and personal use.
