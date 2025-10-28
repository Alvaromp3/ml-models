# ⚽ Elite Sports Performance Analytics

Advanced machine learning platform for analyzing soccer player performance, preventing injuries, and optimizing team lineups.

## 🎯 Features

### 1. **Data Audit & Quality Control**

- Upload and analyze CSV datasets
- Data quality reports with missing value detection
- Outlier detection and automated cleaning
- Player Load distribution analysis

### 2. **Model Training**

- **Player Load Prediction (Regression)**: Predict player load based on performance metrics
  - High-performance GradientBoosting model
  - R² score typically > 0.90
  - Real-time predictions
- **Injury Risk Classification**: Classify players into low/medium/high injury risk categories
  - LGBM Classifier with SMOTE for balanced data
  - Accuracy typically > 0.90
  - Comprehensive risk assessment

### 3. **Player Load Analysis**

- Average load by player for training and match sessions
- Professional coaching recommendations
- Identification of high-performance athletes
- Development area insights

### 4. **Injury Prevention**

- Automated injury risk assessment for last 2 weeks
- Personalized recommendations based on risk level
- Immediate action items for high-risk players
- Wellness and recovery protocols

### 5. **Team Lineup Calculator**

- Optimized lineup selection based on multiple strategies:
  - Balanced Performance
  - Maximum Energy Output
  - Speed-Focused
  - Endurance-Focused
  - Custom Weight Configuration
- Radar charts for player comparison
- Composite scoring system

### 6. **Performance Analytics**

- Key Performance Indicators (KPIs) dashboard
- Correlation analysis heatmap
- Temporal performance trends
- Speed zone distribution analysis
- Heart Rate Load Zone analysis
- Player comparison matrix
- Comprehensive coaching recommendations

## 🚀 Getting Started

### Installation

1. **Navigate to the project directory:**

```bash
cd graceland_soccer_model
```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Application

**Method 1: Using the run script**

```bash
python3 run_app.py
```

**Method 2: Direct Streamlit command**

```bash
python3 -m streamlit run app.py
```

**Method 3: Alternative (if method 2 doesn't work)**

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

> **Note**: On macOS, you may need to use `python3` instead of `python`

## 📊 Dataset Requirements

The application expects a CSV file with the following columns (or similar):

### Required Columns:

- `Player Name`: Player identifier
- `Player Load`: Target variable for regression
- `Date`: Session date

### Performance Metrics (at least some of these):

- `Work Ratio`
- `Energy (kcal)`
- `Distance (miles)`
- `Sprint Distance (yards)`
- `Top Speed (mph)`
- `Max Acceleration (yd/s/s)`
- `Max Deceleration (yd/s/s)`
- `Distance Per Min (yd/min)`
- `Hr Load`
- `Hr Max (bpm)`
- `Time In Red Zone (min)`
- `Impacts`
- `Impact Zones: > 20 G (Impacts)`
- `Impact Zones: 15 - 20 G (Impacts)`
- `Power Plays`
- `Power Score (w/kg)`
- `Distance in Speed Zone 4 (miles)`
- `Distance in Speed Zone 5 (miles)`
- `Time in HR Load Zone 85% - 96% Max HR (secs)`

## 🎓 How to Use

### Step 1: Upload Your Data

1. Navigate to **📊 Data Audit** page
2. Click "Upload your CSV file"
3. Review the data quality report
4. Click "🧹 Clean Data & Remove Outliers" button

### Step 2: Train Models

1. Navigate to **🤖 Model Training** page
2. Train the regression model for Player Load prediction
3. Train the classification model for Injury Risk assessment
4. Review model performance metrics

### Step 3: Analyze Results

1. **Player Load Analysis**: View player performance rankings
2. **Injury Prevention**: Check current injury risks and get recommendations
3. **Team Lineup Calculator**: Generate optimal lineup based on strategy
4. **Performance Analytics**: Explore detailed performance insights

## 🔧 Technical Details

### Models Used

#### Regression Model (Player Load)

- **Algorithm**: GradientBoostingRegressor
- **Preprocessing**: StandardScaler + Feature Selection + PCA
- **Expected Performance**: R² > 0.90

#### Classification Model (Injury Risk)

- **Algorithm**: LightGBM Classifier
- **Preprocessing**: StandardScaler + SMOTE + Feature Selection
- **Expected Performance**: Accuracy > 0.90

### Data Preprocessing

- **Outlier Handling**: IQR method (1.5 \* IQR)
- **Missing Values**: Imputed using mode (categorical) and median (numeric)
- **Feature Scaling**: StandardScaler for normalization
- **Feature Selection**: SelectKBest / SelectFromModel

### Injury Risk Calculation

Risk categories are determined based on:

- **Low Risk**: Player Load ≤ 250
- **Medium Risk**: 250 < Player Load ≤ 400
- **High Risk**: Player Load > 400 AND high Hr Load AND extended Red Zone time

## 📈 Performance Benchmarks

Elite Performance Targets:

- **Player Load**: 350-450 per session
- **Distance**: 4-7 miles per training
- **Sprint Work**: 200-400 yards high-speed
- **Energy**: 600-900 kcal per hour
- **High-intensity HR**: 15-25 minutes (85-100% max)

## ⚠️ Important Notes

- This system is designed to **support** coaching decisions, not replace professional judgment
- Always combine data insights with coaching experience and medical expertise
- Model predictions are based on training data and may require periodic retraining
- High injury risk alerts should trigger immediate medical assessment

## 🤝 Support

For questions or issues, please refer to:

- The inline help tooltips (💡 **Coach Insight** boxes)
- Professional sports science resources
- Medical staff for injury-related concerns

## 📝 License

This project is developed for sports performance analytics purposes.

## 🎯 Version

**Version**: 1.0.0  
**Last Updated**: 2024  
**Powered by**: Machine Learning & Sports Science
