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

### Backend

```bash
cd graceland_soccer_web/backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd graceland_soccer_web/frontend
npm install
npm run dev
```

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
