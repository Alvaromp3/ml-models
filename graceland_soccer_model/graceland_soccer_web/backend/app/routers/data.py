from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from ..models.schemas import ApiResponse
from ..services.data_service import data_service
import os

router = APIRouter(prefix="/data", tags=["Data"])

# Path to sample data
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DATA_PATH = os.path.join(BACKEND_DIR, 'sample_catapult_data.csv')


class CleanDataRequest(BaseModel):
    method: str = 'iqr'  # 'iqr' or 'zscore'
    threshold: float = 3.0  # IQR multiplier (more permissive - only extreme outliers)


@router.post("/upload", response_model=ApiResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        content = await file.read()
        result = data_service.load_from_upload(content)
        return ApiResponse(success=True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-sample", response_model=ApiResponse)
async def load_sample_data():
    """Load sample Catapult data"""
    try:
        if not os.path.exists(SAMPLE_DATA_PATH):
            raise HTTPException(status_code=404, detail=f"Sample data file not found")
        
        result = data_service.load_csv(SAMPLE_DATA_PATH)
        return ApiResponse(success=True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=ApiResponse)
async def get_data_status():
    """Get current data status"""
    try:
        return ApiResponse(success=True, data={
            'loaded': data_service.df is not None,
            'rowCount': len(data_service.df) if data_service.df is not None else 0,
            'players': data_service.players if data_service.df is not None else []
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit", response_model=ApiResponse)
async def get_data_audit():
    """Get data quality audit report"""
    try:
        if data_service.df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        audit = data_service.get_data_audit()
        return ApiResponse(success=True, data=audit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean-outliers", response_model=ApiResponse)
async def clean_outliers(request: CleanDataRequest):
    """Remove outliers from data using IQR method"""
    try:
        if data_service.df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        result = data_service.clean_outliers(method=request.method, threshold=request.threshold)
        return ApiResponse(success=True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", response_model=ApiResponse)
async def reset_data():
    """Reset data to original (undo cleaning)"""
    try:
        if data_service.df is None:
            raise HTTPException(status_code=400, detail="No data loaded")
        
        result = data_service.reset_to_original()
        return ApiResponse(success=True, data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
