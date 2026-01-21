from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.schemas import ApiResponse
from ..services.data_service import data_service

router = APIRouter(prefix="/settings", tags=["Settings"])


class DateReferenceRequest(BaseModel):
    useTodayAsReference: bool


@router.get("/date-reference", response_model=ApiResponse)
async def get_date_reference_setting():
    """Get current date reference setting"""
    try:
        setting = data_service.get_date_reference_setting()
        return ApiResponse(success=True, data=setting)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/date-reference", response_model=ApiResponse)
async def set_date_reference_setting(request: DateReferenceRequest):
    """Set date reference setting"""
    try:
        result = data_service.set_date_reference_setting(request.useTodayAsReference)
        return ApiResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
