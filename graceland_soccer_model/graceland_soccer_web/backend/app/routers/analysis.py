from fastapi import APIRouter, HTTPException
from ..models.schemas import ApiResponse, PredictLoadRequest, PredictRiskRequest, CompareRequest
from ..services.ml_service import ml_service
from ..services.data_service import data_service
from ..services.ollama_service import ollama_service

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/predict-load", response_model=ApiResponse)
async def predict_load(request: PredictLoadRequest):
    """Predict player load for next session (match or training)."""
    try:
        features = request.features
        if not features:
            player = data_service.get_player_detail(request.playerId)
            if not player:
                raise HTTPException(
                    status_code=404,
                    detail="Player not found. Load data in Dashboard and ensure the player is in the current team."
                )
            features = player['metrics']

        session_type = (request.sessionType or 'match').lower()
        if session_type not in ('match', 'training'):
            session_type = 'match'
        result = ml_service.predict_load(features, session_type)

        player_name = "Unknown"
        player = data_service.get_player_detail(request.playerId)
        if player:
            player_name = player['name']

        return ApiResponse(success=True, data={
            'playerId': request.playerId,
            'playerName': player_name,
            'predictedLoad': round(result['predictedLoad'], 2),
            'confidence': result.get('confidence', 0.8),
            'method': result.get('method', 'ml_model'),
            'sessionType': result.get('sessionType', session_type),
            'features': features,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/team-average", response_model=ApiResponse)
async def get_team_average():
    """Get team average metrics. Returns stub when no data so frontend does not break."""
    try:
        team_avg = data_service.get_team_average_metrics()
        if not team_avg:
            return ApiResponse(success=True, data={
                'id': 'team_average',
                'name': 'Team Average',
                'position': 'TEAM',
                'number': 0,
                'riskLevel': 'low',
                'avgLoad': 0,
                'avgSpeed': 0,
                'sessions': 0,
                'lastSession': None,
                'hasRecentData': False,
                'recentSessionCount': 0,
                'metrics': {},
                'teamStats': {'totalPlayers': 0, 'playersWithRecentData': 0, 'riskDistribution': {'low': 0, 'medium': 0, 'high': 0}},
            })
        return ApiResponse(success=True, data=team_avg)
    except Exception as e:
        return ApiResponse(success=True, data={
            'id': 'team_average',
            'name': 'Team Average',
            'position': 'TEAM',
            'number': 0,
            'riskLevel': 'low',
            'avgLoad': 0,
            'avgSpeed': 0,
            'sessions': 0,
            'hasRecentData': False,
            'recentSessionCount': 0,
            'metrics': {},
            'teamStats': {'totalPlayers': 0, 'playersWithRecentData': 0, 'riskDistribution': {'low': 0, 'medium': 0, 'high': 0}},
        })


@router.post("/predict-risk", response_model=ApiResponse)
async def predict_risk(request: PredictRiskRequest):
    """Predict injury risk - only uses data from last 45 days (1.5 months)"""
    try:
        # Handle team average special case
        if request.playerId == 'team_average':
            team_avg = data_service.get_team_average_metrics()
            if not team_avg:
                return ApiResponse(success=True, data={
                    'playerId': 'team_average',
                    'playerName': 'Team Average',
                    'riskLevel': 'low',
                    'probability': 0.0,
                    'factors': ["No team data available. Upload a CSV in Dashboard to analyze."],
                    'recommendations': [
                        "Upload training data in the Dashboard",
                        "Risk cannot be assessed without data",
                    ],
                    'hasRecentData': False,
                    'recentSessionCount': 0,
                })
            
            # Use team average metrics for prediction
            features = team_avg.get('metrics', {})
            has_recent_data = team_avg.get('hasRecentData', False)
            recent_sessions = team_avg.get('recentSessionCount', 0)
            
            if not has_recent_data or recent_sessions == 0 or not features:
                return ApiResponse(success=True, data={
                    'playerId': 'team_average',
                    'playerName': 'Team Average',
                    'riskLevel': 'low',
                    'probability': 0.0,
                    'factors': [f"No recent training data in the last 45 days ({recent_sessions} sessions)"],
                    'recommendations': [
                        "Team has no recent training sessions",
                        "Risk cannot be accurately assessed without recent data",
                        "Consider starting with low intensity training to gather baseline data"
                    ],
                    'hasRecentData': False,
                    'recentSessionCount': recent_sessions
                })
            
            # Validate features before prediction
            if not isinstance(features, dict) or len(features) == 0:
                return ApiResponse(success=True, data={
                    'playerId': 'team_average',
                    'playerName': 'Team Average',
                    'riskLevel': 'low',
                    'probability': 0.0,
                    'factors': ["Insufficient metrics data for risk prediction"],
                    'recommendations': [
                        "Team metrics are not available",
                        "Upload training data to enable risk analysis"
                    ],
                    'hasRecentData': False,
                    'recentSessionCount': recent_sessions
                })
            
            try:
                risk_level, probability, factors, recommendations = ml_service.predict_risk(features)
            except Exception as e:
                # Fallback if prediction fails
                return ApiResponse(success=True, data={
                    'playerId': 'team_average',
                    'playerName': 'Team Average',
                    'riskLevel': 'low',
                    'probability': 0.0,
                    'factors': [f"Risk prediction unavailable: {str(e)}"],
                    'recommendations': [
                        "Unable to calculate risk with current data",
                        "Ensure training models are properly trained",
                        "Check that sufficient player data is available"
                    ],
                    'hasRecentData': True,
                    'recentSessionCount': recent_sessions
                })
            
            return ApiResponse(success=True, data={
                'playerId': 'team_average',
                'playerName': 'Team Average',
                'riskLevel': risk_level,
                'probability': probability,
                'factors': factors,
                'recommendations': recommendations,
                'hasRecentData': True,
                'recentSessionCount': recent_sessions
            })
        
        player = data_service.get_player_detail(request.playerId)
        if not player:
            return ApiResponse(success=True, data={
                'playerId': request.playerId,
                'playerName': 'Unknown',
                'riskLevel': 'low',
                'probability': 0.0,
                'factors': ["Player not found or no data available."],
                'recommendations': ["Select a player from the list and ensure data is loaded."],
                'hasRecentData': False,
                'recentSessionCount': 0,
            })
        
        player_name = player.get('name', 'Unknown')
        # Check if player has recent data (last 45 days)
        has_recent_data = player.get('hasRecentData', False)
        recent_sessions = player.get('recentSessionCount', 0)
        
        # If no recent data, return low risk automatically
        if not has_recent_data or recent_sessions == 0:
            return ApiResponse(success=True, data={
                'playerId': request.playerId,
                'playerName': player_name,
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
        
        features = player.get('metrics', {})
        
        # Validate features before prediction
        if not isinstance(features, dict) or len(features) == 0:
            return ApiResponse(success=True, data={
                'playerId': request.playerId,
                'playerName': player_name,
                'riskLevel': 'low',
                'probability': 0.0,
                'factors': ["Insufficient metrics data for risk prediction"],
                'recommendations': [
                    "Player metrics are not available",
                    "Upload training data to enable risk analysis"
                ],
                'hasRecentData': True,
                'recentSessionCount': recent_sessions
            })
        
        try:
            risk_level, probability, factors, recommendations = ml_service.predict_risk(features)
        except Exception as e:
            # Fallback if prediction fails
            return ApiResponse(success=True, data={
                'playerId': request.playerId,
                'playerName': player_name,
                'riskLevel': 'low',
                'probability': 0.0,
                'factors': [f"Risk prediction unavailable: {str(e)}"],
                'recommendations': [
                    "Unable to calculate risk with current data",
                    "Ensure training models are properly trained",
                    "Check that sufficient player data is available"
                ],
                'hasRecentData': True,
                'recentSessionCount': recent_sessions
            })
        
        return ApiResponse(success=True, data={
            'playerId': request.playerId,
            'playerName': player_name,
            'riskLevel': risk_level,
            'probability': probability,
            'factors': factors,
            'recommendations': recommendations,
            'hasRecentData': True,
            'recentSessionCount': recent_sessions
        })
    except ValueError as e:
        return ApiResponse(success=True, data={
            'playerId': getattr(request, 'playerId', ''),
            'playerName': 'Unknown',
            'riskLevel': 'low',
            'probability': 0.0,
            'factors': [str(e)],
            'recommendations': ["Check input data and try again."],
            'hasRecentData': False,
            'recentSessionCount': 0,
        })
    except Exception as e:
        return ApiResponse(success=True, data={
            'playerId': getattr(request, 'playerId', ''),
            'playerName': 'Unknown',
            'riskLevel': 'low',
            'probability': 0.0,
            'factors': [f"Analysis error: {str(e)}"],
            'recommendations': ["Something went wrong. Try again or upload data in Dashboard."],
            'hasRecentData': False,
            'recentSessionCount': 0,
        })


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
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Getting AI recommendations for player: {request.playerId}")
        
        # Handle team average special case
        if request.playerId == 'team_average':
            player = data_service.get_team_average_metrics()
            if not player:
                logger.warning("Team average data not available")
                raise HTTPException(status_code=404, detail="No team data available")
        else:
            player = data_service.get_player_detail(request.playerId)
            if not player:
                logger.warning(f"Player not found: {request.playerId}")
                raise HTTPException(status_code=404, detail="Player not found")
        
        # Check for recent data
        has_recent_data = player.get('hasRecentData', False)
        logger.info(f"Player {player['name']} has recent data: {has_recent_data}")
        
        # Get risk prediction first
        if not has_recent_data:
            risk_level = 'low'
            risk_factors = ["No recent training data (last 45 days)"]
        else:
            features = player['metrics']
            risk_level, _, risk_factors, _ = ml_service.predict_risk(features)
        
        logger.info(f"Risk level determined: {risk_level}")
        
        # Get AI recommendations
        result = ollama_service.get_player_recommendations(
            player_name=player['name'],
            player_data=player,
            risk_level=risk_level,
            risk_factors=risk_factors
        )
        
        logger.info(f"AI recommendations result - success: {result.get('success')}, source: {result.get('source')}")
        
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
        logger.error(f"Error getting AI recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
