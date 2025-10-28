## Wine Quality Predictor (Flask)

Flask application to predict wine quality using machine learning.

### Features

- Gradient boosting classifier (XGBoost)
- Robust preprocessing: StandardScaler + OneHotEncoder, IQR outlier removal
- Binary target: High quality (≥6) vs Low (<6)
- Train/test metrics: Accuracy, Precision, Recall, F1
- Portable dataset path via environment variable

### Installation

```bash
pip install -r requirements.txt
```

### Dataset

- Default file: `winequalityN.csv` in this folder
- Or set env var to a custom path:

```bash
export WINE_DATASET_PATH=/absolute/path/to/winequalityN.csv
```

### Run

```bash
python app.py
```

Then open `http://localhost:5000`

### Usage

1. Click “Entrenar Modelo” to train and view train/test metrics
2. Fill the form and submit to get prediction + confidence

### Project Structure

- `app.py` — Flask app
- `modelo.py` — ML logic (training, prediction)
- `templates/index.html` — UI
- `winequalityN.csv` — dataset
- `requirements.txt` — dependencies

### Author

Developed by: Alvaro Martin-Pena
