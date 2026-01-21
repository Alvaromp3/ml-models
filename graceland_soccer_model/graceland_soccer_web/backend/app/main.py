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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
