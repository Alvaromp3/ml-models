# 🚀 Quick Start Guide

## Installation

```bash
# 1. Navigate to project directory
cd graceland_soccer_model

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python3 run_app.py
```

**Alternative: Direct run (if dependencies already installed)**

```bash
python3 -m streamlit run app.py
```

## First Use

1. **Upload Data**: Go to "📊 Data Audit" and upload your CSV file
2. **Clean Data**: Click "🧹 Clean Data & Remove Outliers"
3. **Train Models**: Go to "🤖 Model Training" and train both models
4. **Analyze**: Explore all the different pages to get insights

## Sample Workflow

```
Data Audit → Clean Data → Train Regression Model → Train Classification Model
    ↓
Player Load Analysis (see top performers)
    ↓
Injury Prevention (check risk levels)
    ↓
Team Lineup Calculator (select optimal lineup)
    ↓
Performance Analytics (deep dive analysis)
```

That's it! 🎉
