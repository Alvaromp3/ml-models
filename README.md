# ⚽ Elite Sports Performance Analytics - **DEMO**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-demo-yellow.svg)](https://github.com/Alvaromp3/ml-models)

> **⚠️ NOTE**: This is a **PROTOTYPE/DEMO** version. A professional mobile app for coaches is currently in development.

Advanced machine learning platform for analyzing soccer player performance, preventing injuries, and optimizing team lineups.

**🚀 Upcoming**: Professional mobile application for sports coaches with advanced analytics, real-time updates, and intuitive interface.

## 📸 Screenshots

### Dashboard Overview

![Dashboard](screenshots/dashboard.png)
_Quick overview of KPIs and player risk alerts_

### Model Training Interface

![Model Training](screenshots/model_training.png)
_Train regression and classification models with performance metrics_

### Injury Risk Prevention

![Injury Prevention](screenshots/injury_prevention.png)
_Automated risk assessment and personalized recommendations_

### Player Load Analysis

![Player Load Analysis](screenshots/player_load.png)
_Individual player performance tracking and insights_

### Team Lineup Calculator

![Lineup Calculator](screenshots/lineup_calculator.png)
_Optimize team formations with multiple strategies_

> **Note**: Screenshots will be added in the first release. The application runs locally at `http://localhost:8501`

## 👤 Author

**Alvaro Martin-Pena** | Machine Learning Engineer

- GitHub: [@Alvaromp3](https://github.com/Alvaromp3)
- LinkedIn: [Alvaro Martin-Pena](https://linkedin.com/in/alvaro-martin-pena)

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

### 🌐 Deploy to Streamlit Cloud

**Option 1: Deploy via GitHub**
1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `Alvaromp3/ml-models`
5. Set Main file path: `graceland_soccer_model/app.py`
6. Click "Deploy!"
7. Your app will be live at: `https://your-app-name.streamlit.app`

**Option 2: Deploy from this directory**
```bash
# Install Streamlit CLI (if not already installed)
pip install streamlit

# Deploy using Streamlit Cloud CLI
streamlit deploy
```

### 🐳 Deploy with Docker
```bash
docker build -t graceland-soccer .
docker run -p 8501:8501 graceland-soccer
```

## 📊 Dataset Requirements

**⚠️ IMPORTANT**: This application requires data exported from **Catapult Sports GPS tracking systems**. The CSV file must contain the following columns (or similar):

**💡 Quick Start**: Use the included `sample_catapult_data.csv` to test the application without uploading your own data!

### Required Columns for Catapult Export:

- `Player Name`: Player identifier
- `Player Load`: Target variable for regression
- `Date`: Session date

### Recommended Performance Metrics (from Catapult):

The application works best with the following Catapult metrics included in your CSV export:

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
   - **For testing**: Use `sample_catapult_data.csv` included in the project
   - **For real data**: Upload your Catapult export CSV file
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

Risk categories are determined based on percentiles of Player Load distribution:

- **Low Risk**: Below 60th percentile
- **Medium Risk**: 60th to 95th percentile
- **High Risk**: Above 95th percentile (top 5% of load) OR high load combined with high energy expenditure and impacts

This balanced approach ensures approximately 40% Low, 35% Medium, and 5% High risk players in a typical squad.

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
