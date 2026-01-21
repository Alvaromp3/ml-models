# Graceland Soccer Analytics Web Application

Web application for analyzing soccer player performance, predicting injury risk, and optimizing team lineups.

## Structure

```
graceland_soccer_web/
├── backend/          # FastAPI backend
│   ├── app/         # Application code
│   ├── modelos_graceland/  # ML models
│   └── requirements.txt
└── frontend/         # React + TypeScript frontend
    └── src/          # Source code
```

## Quick Start

### Try it Online

**Live Demo:** https://ml-models-z9jxxmnmkkxtca7eewrywj.streamlit.app/

> **Note**: The demo is fully functional with sample Catapult data included. No installation required!

**Features available:**
- **Dashboard**: KPIs overview with real-time player risk alerts
- **Model Training**: Train and evaluate regression + classification models
- **Injury Prevention**: Automated risk assessment with personalized recommendations
- **Player Load Analysis**: Individual player performance tracking and insights
- **Team Lineup Calculator**: Optimize team formations with multiple strategies

### Local Development

#### Backend

```bash
cd graceland_soccer_web/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd graceland_soccer_web/frontend
npm install
npm run dev
```

## Author

**Alvaro Martin-Pena** | Machine Learning Engineer

- GitHub: [@Alvaromp3](https://github.com/Alvaromp3)
- LinkedIn: [Alvaro Martin-Pena](https://linkedin.com/in/alvaro-martin-pena)

## Features

- Player performance dashboard
- Injury risk prediction
- Player load analysis
- Best lineup generator
- Data audit and outlier cleaning
- AI-powered coaching recommendations (Ollama)

## Models

All ML models are stored in `backend/modelos_graceland/`:
- `regression_model.pkl` - Player Load prediction
- `classification_model.pkl` - Injury Risk classification
- `load_model.joblib` - Load model (joblib format)
- `risk_model.joblib` - Risk model (joblib format)
