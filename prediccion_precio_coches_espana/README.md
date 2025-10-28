# Used Car Price Predictor - Spain

## Description
Flask and Streamlit web applications to predict used car prices in Spain using Random Forest Regressor trained on vehicle characteristics.

**Developed by:** Alvaro Martin-Pena

## Motivation
Help buyers and sellers estimate fair market prices for used cars in Spain based on technical, physical, and commercial features.

## Features
- **Advanced model**: Random Forest Regressor (300 trees)
- **Dual interfaces**: Flask and Streamlit web apps
- **Real-time prediction**: Instant price estimates
- **Multiple features**: 14 input variables
- **Dataset included**: Spanish used car dataset

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:

**Streamlit (Recommended):**
```bash
streamlit run app_streamlit.py
```

**Flask:**
```bash
python app.py
```

3. Open your browser:
   - Streamlit: `http://localhost:8501`
   - Flask: `http://localhost:5000`

## Usage

### Streamlit
1. Click "Train Model" to train with the included dataset
2. Fill in car characteristics
3. Click "Predict Price" to get the estimate

### Flask
1. Click "Train Model" button
2. Fill in the form with car characteristics
3. Click "Predict Price" to get the estimate

## Model Features

### Algorithm
- **Model**: Random Forest Regressor
- **Trees**: 300
- **Max Depth**: 20
- **Min Samples Leaf**: 2

### Input Features

**Numeric:**
- Age (years)
- Mileage (km)
- Horsepower (CV)
- Consumption (L/100km)
- Number of owners
- Number of doors

**Categorical:**
- Brand
- Model
- Fuel type
- Transmission
- Condition
- Region
- Color
- Body type

### Preprocessing
- StandardScaler for numerical variables
- OneHotEncoder for categorical variables
- Outlier removal using IQR method
- Duplicate removal

## Expected Performance
- **R² Score**: Displayed after training
- **MAE**: Shown in interface
- **RMSE**: Calculated on test set

## Project Structure
```
prediccion_precio_coches_espana/
├── app.py                    # Flask application
├── app_streamlit.py          # Streamlit application
├── modelo.py                 # ML model functions
├── templates/
│   └── index.html           # Flask interface
├── coches_espana_usados_1500_edad.csv  # Dataset
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Documentation
```

## Technologies
- Python 3.8+
- Flask / Streamlit
- Scikit-learn
- Pandas / NumPy
- Random Forest Regressor

## Author
**Alvaro Martin-Pena**
- Machine Learning Engineer
- Data Scientist

## License
For educational and personal use.
