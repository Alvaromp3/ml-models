# FIFA Score Predictor ⚽

## Description
Streamlit web application to predict competitive FIFA match scores. The model uses an optimized CatBoost Regressor to deliver accurate predictions from match statistics.

**Developed by:** Alvaro Martin-Pena

## Motivation
I built this project to help competitive FIFA players estimate their score based on their match stats and identify areas to improve.

## Features
- **Optimized model**: CatBoost with GridSearchCV
- **Modern UI**: Professional Streamlit interface
- **Real-time prediction**: Instant results
- **Actionable insights**: Recommendations to improve performance

## Installation

1) Clone the repository:
```bash
git clone https://github.com/Alvaromp3/fifa-score-predictor.git
cd fifa-score-predictor
```

2) Install dependencies:
```bash
pip install -r requirements.txt
```

3) Prepare the dataset:

Place the file `fifa_esports_dataset_no_id.csv` in the project directory, or set an environment variable with its path.

Required columns:
- `goals_scored`
- `shots_on_target`
- `possession_pct`
- `passes_completed`
- `tackles_won`
- `fouls_committed`
- `yellow_cards`
- `red_cards`
- `team_rating`
- `opponent_rating`
- `match_result`
- `score` (target)

4) Run the app:
```bash
streamlit run app.py
```

Optional: custom dataset path
```bash
export FIFA_DATASET_PATH=/path/to/your/file.csv
streamlit run app.py
```

## Usage
1. Enter match statistics in the sidebar
2. Click "Analyze Performance"
3. View the predicted score and improvement tips

## Model Inputs
- Goals scored
- Shots on target
- Ball possession (%)
- Passes completed
- Tackles won
- Fouls committed
- Yellow/Red cards
- Team rating
- Opponent rating
- Match result

## Tech Stack
- **Frontend**: Streamlit
- **Model**: CatBoost
- **Preprocessing**: Scikit-learn
- **Data**: Pandas, NumPy

## Project Structure
```
fifa_score_predictor/
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── .gitignore                      # Git ignore rules
└── fifa_esports_dataset_no_id.csv  # Dataset (required - not included)
```

## Notes
- Dataset not included due to size. Provide it as described above.
- `.gitignore` is configured to ignore CatBoost training artifacts (`catboost_info/`).
- You can set a custom dataset path via `FIFA_DATASET_PATH`.
- First run may take longer due to model training.

## Author
**Alvaro Martin-Pena**
- Machine Learning Engineer
- Sports Analytics Specialist

## License
For educational and personal use.
