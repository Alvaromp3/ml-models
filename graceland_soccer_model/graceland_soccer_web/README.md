# Elite Sports Performance Analytics - Web Application

Modern web application for sports analytics with React frontend and FastAPI backend.

## Quick Start

### 1. Start Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Start Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

### 3. Open Browser

Navigate to `http://localhost:5173`

## Features

- **Dashboard**: Real-time KPIs, load charts, risk distribution
- **Players**: Player cards with filtering and search
- **Analysis**: Injury risk prediction with ML
- **Training**: Train and evaluate ML models
- **Settings**: Configure thresholds and preferences

## Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite
- Tailwind CSS
- Recharts
- React Query
- React Router

**Backend:**
- FastAPI
- scikit-learn
- XGBoost / LightGBM
- Pandas / NumPy

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/kpis` | GET | Dashboard KPIs |
| `/api/dashboard/load-history` | GET | Load history chart data |
| `/api/dashboard/risk-distribution` | GET | Risk distribution |
| `/api/players` | GET | All players |
| `/api/players/high-risk` | GET | High risk players |
| `/api/analysis/predict-risk` | POST | Predict injury risk |
| `/api/training/train-load` | POST | Train load model |
| `/api/training/train-risk` | POST | Train risk model |
| `/api/data/load-sample` | POST | Load sample data |

## Author

**Alvaro Martin-Pena** | Machine Learning Engineer
