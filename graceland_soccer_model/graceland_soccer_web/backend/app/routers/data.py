from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from ..models.schemas import ApiResponse
from ..services.data_service import data_service
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["Data"])

# Path to sample data
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DATA_PATH = os.path.join(BACKEND_DIR, 'sample_catapult_data.csv')


class CleanDataRequest(BaseModel):
    method: str = 'iqr'  # 'iqr' or 'zscore'
    threshold: float = 3.0  # IQR multiplier (more permissive - only extreme outliers)

class UploadRequest(BaseModel):
    team: str = 'mens'  # 'mens' or 'womens'

class UpdatePositionRequest(BaseModel):
    playerName: str
    position: str
    team: Optional[str] = None


@router.post("/upload", response_model=ApiResponse)
async def upload_data(file: UploadFile = File(...), team: str = 'mens'):
    """Upload CSV file for a specific team"""
    try:
        if team not in ['mens', 'womens']:
            raise HTTPException(status_code=400, detail="Team must be 'mens' or 'womens'")
        
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        result = data_service.load_from_upload(content, team)
        return ApiResponse(success=True, data=result)
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        import traceback
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


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

@router.post("/update-position", response_model=ApiResponse)
async def update_player_position(request: UpdatePositionRequest):
    """Update player position"""
    try:
        result = data_service.update_player_position(
            player_name=request.playerName,
            position=request.position,
            team=request.team
        )
        return ApiResponse(success=True, data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating player position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
