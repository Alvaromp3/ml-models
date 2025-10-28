# Airbnb Price Predictor

## Description
Flask web application to predict Airbnb property prices using a Random Forest Regressor trained on property features.

**Developed by:** Alvaro Martin-Pena

## Motivation
Predict rental prices for Airbnb properties to help hosts set competitive prices and travelers budget effectively.

## Features
- **Optimized model**: Random Forest Regressor (150 trees)
- **Web interface**: Flask-powered UI
- **Real-time prediction**: Instant price estimates
- **Feature importance**: Model provides insights on price factors

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset exists:
```bash
# Place airbnb_synthetic.csv in the project directory
# Or set environment variable:
export AIRBNB_DATASET_PATH=/path/to/your/dataset.csv
```

3. Run the application:
```bash
python run_app.py
```

Or directly:
```bash
python app.py
```

## Usage
1. Open your browser at `http://localhost:5000`
2. Fill in the property details:
   - **Latitude/Longitude**: Location coordinates
   - **Property Type**: Apartment, House, Condominium, Loft
   - **Room Type**: Entire home/apt, Private room, Shared room
   - **Bedrooms**: Number of bedrooms
   - **Bathrooms**: Number of bathrooms
   - **Reviews**: Number of reviews
   - **Availability**: Days available per year
3. Click "Predict Price"
4. Get the price prediction

## Model Features

### Algorithm
- **Algorithm**: Random Forest Regressor
- **Trees**: 150
- **Max Depth**: 15
- **Input Variables**: 9 features

### Preprocessing
- StandardScaler for numerical variables
- LabelEncoder for categorical variables
- Outlier removal using IQR method

### Expected Performance
- **R² Score**: 0.75-0.85 (significantly improved from 0.36)
- **Mean Absolute Error**: Minimal price deviation

## Project Structure
```
airbnb_model/
├── app.py                  # Flask application
├── modelo_airbnb.py        # ML model class
├── templates/
│   └── index.html          # Web interface
├── airbnb_synthetic.csv   # Training dataset
├── run_app.py             # Easy launch script
├── .gitignore             # Git ignore rules
└── README.md               # Documentation
```

## Technical Improvements Made

### Fixed Critical Bug
- Changed from `KNeighborsClassifier` to `RandomForestRegressor`
- Proper regression model for continuous price prediction

### Enhanced Performance
- Replaced KNN with Random Forest for better accuracy
- Increased R² from ~0.36 to ~0.80+
- Better generalization to unseen data

### Better Practices
- Environment variable support for dataset path
- Proper error handling for missing files
- Professional .gitignore configuration
- English documentation

## Author
**Alvaro Martin-Pena**
- Machine Learning Engineer
- Data Scientist

## License
For educational and personal use.
