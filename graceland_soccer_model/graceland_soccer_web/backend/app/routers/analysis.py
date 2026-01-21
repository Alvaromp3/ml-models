from fastapi import APIRouter, HTTPException
from ..models.schemas import ApiResponse, PredictLoadRequest, PredictRiskRequest, CompareRequest
from ..services.ml_service import ml_service
from ..services.data_service import data_service
from ..services.ollama_service import ollama_service

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/predict-load", response_model=ApiResponse)
async def predict_load(request: PredictLoadRequest):
    """Predict player load"""
    try:
        # Get player data if no features provided
        features = request.features
        if not features:
            player = data_service.get_player_detail(request.playerId)
            if not player:
                raise HTTPException(status_code=404, detail="Player not found")
            features = player['metrics']
        
        prediction, confidence = ml_service.predict_load(features)
        
        player_name = "Unknown"
        player = data_service.get_player_detail(request.playerId)
        if player:
            player_name = player['name']
        
        return ApiResponse(success=True, data={
            'playerId': request.playerId,
            'playerName': player_name,
            'predictedLoad': round(prediction, 2),
            'confidence': confidence,
            'features': features
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-risk", response_model=ApiResponse)
async def predict_risk(request: PredictRiskRequest):
    """Predict injury risk - only uses data from last 45 days (1.5 months)"""
    try:
        player = data_service.get_player_detail(request.playerId)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Check if player has recent data (last 45 days)
        has_recent_data = player.get('hasRecentData', False)
        recent_sessions = player.get('recentSessionCount', 0)
        
        # If no recent data, return low risk automatically
        if not has_recent_data or recent_sessions == 0:
            return ApiResponse(success=True, data={
                'playerId': request.playerId,
                'playerName': player['name'],
                'riskLevel': 'low',
                'probability': 0.0,
                'factors': [f"No training data in the last 45 days ({recent_sessions} sessions)"],
                'recommendations': [
                    "Player has no recent training sessions",
                    "Risk cannot be accurately assessed without recent data",
                    "Consider starting with low intensity training to gather baseline data"
                ],
                'hasRecentData': False,
                'recentSessionCount': recent_sessions
            })
        
        features = player['metrics']
        risk_level, probability, factors, recommendations = ml_service.predict_risk(features)
        
        return ApiResponse(success=True, data={
            'playerId': request.playerId,
            'playerName': player['name'],
            'riskLevel': risk_level,
            'probability': probability,
            'factors': factors,
            'recommendations': recommendations,
            'hasRecentData': True,
            'recentSessionCount': recent_sessions
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ApiResponse)
async def compare_players(request: CompareRequest):
    """Compare multiple players"""
    try:
        results = []
        for player_id in request.playerIds:
            player = data_service.get_player_detail(player_id)
            if player:
                try:
                    result = ml_service.predict_load(player['metrics'], 'training')
                    results.append({
                        'playerId': player_id,
                        'playerName': player['name'],
                        'predictedLoad': round(result['predictedLoad'], 2),
                        'confidence': result['confidence'],
                        'features': player['metrics']
                    })
                except:
                    results.append({
                        'playerId': player_id,
                        'playerName': player['name'],
                        'predictedLoad': player['avgLoad'],
                        'confidence': 0,
                        'features': player['metrics']
                    })
        
        return ApiResponse(success=True, data=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ollama-status", response_model=ApiResponse)
async def get_ollama_status():
    """Get Ollama AI status"""
    try:
        status = ollama_service.get_status()
        return ApiResponse(success=True, data=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai-recommendations", response_model=ApiResponse)
async def get_ai_recommendations(request: PredictRiskRequest):
    """Get AI-powered recommendations for a player using Ollama"""
    try:
        player = data_service.get_player_detail(request.playerId)
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Check for recent data
        has_recent_data = player.get('hasRecentData', False)
        
        # Get risk prediction first
        if not has_recent_data:
            risk_level = 'low'
            risk_factors = ["No recent training data (last 45 days)"]
        else:
            features = player['metrics']
            risk_level, _, risk_factors, _ = ml_service.predict_risk(features)
        
        # Get AI recommendations
        result = ollama_service.get_player_recommendations(
            player_name=player['name'],
            player_data=player,
            risk_level=risk_level,
            risk_factors=risk_factors
        )
        
        return ApiResponse(success=True, data={
            'playerId': request.playerId,
            'playerName': player['name'],
            'riskLevel': risk_level,
            'hasRecentData': has_recent_data,
            'recentSessionCount': player.get('recentSessionCount', 0),
            'aiRecommendations': result.get('recommendations', ''),
            'aiSource': result.get('source', 'fallback'),
            'aiSuccess': result.get('success', False),
            'aiError': result.get('error')
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
