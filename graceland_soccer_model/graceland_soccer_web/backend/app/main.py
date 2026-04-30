import warnings

# Load environment variables from backend/.env (if present).
# Do not fail if python-dotenv is missing.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Keep startup logs clean: ignore known non-fatal dependency/model warnings
# (Filter by message so it applies even during the initial `requests` import.)
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# LightGBM / sklearn: predicting with ndarray when model was trained with feature names
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import dashboard, players, analysis, training, data, settings

app = FastAPI(
    title="Elite Sports Performance Analytics API",
    description="Backend API for sports analytics dashboard",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router, prefix="/api")
app.include_router(players.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(settings.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "Elite Sports Performance Analytics API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
