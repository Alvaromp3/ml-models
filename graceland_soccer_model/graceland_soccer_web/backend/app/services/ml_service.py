import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import joblib
import pickle
import os
import time
import logging
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BACKEND_DIR, 'modelos_graceland')


class MLService:
    def __init__(self):
        self.load_pipeline: Optional[Any] = None
        self.risk_pipeline: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.load_features: List[str] = []
        self.risk_features: List[str] = []
        self.load_metrics: Optional[Dict] = None
        self.risk_metrics: Optional[Dict] = None
        self._load_saved_models()
    
    def _load_saved_models(self):
        regression_path = os.path.join(MODELS_DIR, 'regression_model.pkl')
        classification_path = os.path.join(MODELS_DIR, 'classification_model.pkl')
        
        if os.path.exists(regression_path):
            try:
                with open(regression_path, 'rb') as f:
                    data = pickle.load(f)
                self.load_pipeline = data.get('model')
                self.load_metrics = data.get('metrics')
                self.load_features = data.get('features', [])
                logger.info(f"Loaded regression model from {regression_path}")
            except Exception as e:
                logger.warning(f"Could not load regression model: {e}")
        
        if os.path.exists(classification_path):
            try:
                with open(classification_path, 'rb') as f:
                    data = pickle.load(f)
                self.risk_pipeline = data.get('model')
                self.risk_metrics = data.get('metrics')
                self.risk_features = data.get('features', [])
                logger.info(f"Loaded classification model from {classification_path}")
            except Exception as e:
                logger.warning(f"Could not load classification model: {e}")
        
        if self.load_pipeline is None:
            load_path = os.path.join(MODELS_DIR, 'load_model.joblib')
            if os.path.exists(load_path):
                try:
                    self.load_pipeline = joblib.load(load_path)
                    cols_path = os.path.join(MODELS_DIR, 'load_feature_cols.joblib')
                    if os.path.exists(cols_path):
                        self.load_features = joblib.load(cols_path)
                    logger.info("Loaded local load prediction model")
                except Exception as e:
                    logger.warning(f"Could not load local load model: {e}")
        
        if self.risk_pipeline is None:
            risk_path = os.path.join(MODELS_DIR, 'risk_model.joblib')
            if os.path.exists(risk_path):
                try:
                    self.risk_pipeline = joblib.load(risk_path)
                    cols_path = os.path.join(MODELS_DIR, 'risk_feature_cols.joblib')
                    if os.path.exists(cols_path):
                        self.risk_features = joblib.load(cols_path)
                    logger.info("Loaded local risk prediction model")
                except Exception as e:
                    logger.warning(f"Could not load local risk model: {e}")
    
    def get_available_algorithms(self) -> Dict[str, List[Dict[str, str]]]:
        regression_algos = [
            {'id': 'gradient_boosting', 'name': 'Gradient Boosting', 'available': True},
            {'id': 'random_forest', 'name': 'Random Forest', 'available': True},
            {'id': 'xgboost', 'name': 'XGBoost', 'available': XGBOOST_AVAILABLE},
            {'id': 'lightgbm', 'name': 'LightGBM', 'available': LIGHTGBM_AVAILABLE},
            {'id': 'catboost', 'name': 'CatBoost', 'available': CATBOOST_AVAILABLE},
        ]
        
        classification_algos = [
            {'id': 'random_forest', 'name': 'Random Forest', 'available': True},
            {'id': 'gradient_boosting', 'name': 'Gradient Boosting', 'available': True},
            {'id': 'xgboost', 'name': 'XGBoost', 'available': XGBOOST_AVAILABLE},
            {'id': 'lightgbm', 'name': 'LightGBM', 'available': LIGHTGBM_AVAILABLE},
            {'id': 'catboost', 'name': 'CatBoost', 'available': CATBOOST_AVAILABLE},
        ]
        
        return {
            'regression': regression_algos,
            'classification': classification_algos
        }
    
    def _get_regressor(self, algorithm: str = 'gradient_boosting'):
        return GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    def _get_classifier(self, algorithm: str = 'lightgbm'):
        if LIGHTGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
        else:
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    def train_load_model(self, df: pd.DataFrame, feature_cols: List[str], 
                         algorithm: str = 'gradient_boosting') -> Dict[str, Any]:
        start_time = time.time()
        
        target_col = 'Player Load'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        available_features = [c for c in feature_cols if c in df.columns and c != target_col]
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]
        
        if len(X) < 10:
            raise ValueError(f"Not enough valid data points for training (got {len(X)}, need at least 10)")
        
        self.load_features = available_features
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = self._get_regressor(algorithm)
        
        self.load_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        self.load_pipeline.fit(X_train, y_train)
        
        y_pred_test = self.load_pipeline.predict(X_test)
        y_pred_train = self.load_pipeline.predict(X_train)
        
        test_metrics = {
            'r2Score': round(r2_score(y_test, y_pred_test), 4),
            'mae': round(mean_absolute_error(y_test, y_pred_test), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
            'mse': round(mean_squared_error(y_test, y_pred_test), 4)
        }
        
        train_metrics = {
            'r2Score': round(r2_score(y_train, y_pred_train), 4),
            'mae': round(mean_absolute_error(y_train, y_pred_train), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4),
        }
        
        cv_scores = cross_val_score(self.load_pipeline, X, y, cv=5, scoring='r2')
        cv_metrics = {
            'cvR2Mean': round(np.mean(cv_scores), 4),
            'cvR2Std': round(np.std(cv_scores), 4)
        }
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.load_pipeline, os.path.join(MODELS_DIR, 'load_model.joblib'))
        joblib.dump(self.load_features, os.path.join(MODELS_DIR, 'load_feature_cols.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'regression_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': self.load_pipeline,
                'metrics': test_metrics,
                'features': self.load_features,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        training_time = round(time.time() - start_time, 2)
        self.load_metrics = test_metrics
        
        return {
            'modelType': 'regression',
            'algorithm': algorithm,
            'metrics': test_metrics,
            'trainMetrics': train_metrics,
            'cvMetrics': cv_metrics,
            'trainingTime': training_time,
            'samplesUsed': len(X),
            'featuresUsed': len(available_features),
            'timestamp': datetime.now().isoformat()
        }
    
    def train_risk_model(self, df: pd.DataFrame, feature_cols: List[str],
                         algorithm: str = 'random_forest') -> Dict[str, Any]:
        start_time = time.time()
        
        load_col = 'Player Load'
        if load_col not in df.columns:
            raise ValueError(f"Column '{load_col}' not found")
        
        df = df.copy()
        
        load_values = pd.to_numeric(df[load_col], errors='coerce').dropna()
        q25 = load_values.quantile(0.25)
        q75 = load_values.quantile(0.75)
        
        df['risk_label'] = pd.cut(
            pd.to_numeric(df[load_col], errors='coerce'),
            bins=[0, q25, q75, float('inf')],
            labels=[0, 1, 2]
        )
        
        available_features = [c for c in feature_cols if c in df.columns and c != load_col and c != 'risk_label']
        X = df[available_features].copy()
        y = df['risk_label'].copy()
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask].astype(int)
        
        if len(X) < 10:
            raise ValueError(f"Not enough valid data points for training (got {len(X)}, need at least 10)")
        
        self.risk_features = available_features
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = self._get_classifier(algorithm)
        
        self.risk_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        self.risk_pipeline.fit(X_train, y_train)
        
        y_pred_test = self.risk_pipeline.predict(X_test)
        y_pred_train = self.risk_pipeline.predict(X_train)
        
        test_metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred_test), 4),
            'precision': round(precision_score(y_test, y_pred_test, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred_test, average='weighted', zero_division=0), 4),
            'f1Score': round(f1_score(y_test, y_pred_test, average='weighted', zero_division=0), 4)
        }
        
        train_metrics = {
            'accuracy': round(accuracy_score(y_train, y_pred_train), 4),
            'precision': round(precision_score(y_train, y_pred_train, average='weighted', zero_division=0), 4),
        }
        
        cv_scores = cross_val_score(self.risk_pipeline, X, y, cv=5, scoring='accuracy')
        cv_metrics = {
            'cvAccuracyMean': round(np.mean(cv_scores), 4),
            'cvAccuracyStd': round(np.std(cv_scores), 4)
        }
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.risk_pipeline, os.path.join(MODELS_DIR, 'risk_model.joblib'))
        joblib.dump(self.risk_features, os.path.join(MODELS_DIR, 'risk_feature_cols.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'classification_model.pkl'), 'wb') as f:
            pickle.dump({
                'model': self.risk_pipeline,
                'metrics': test_metrics,
                'features': self.risk_features,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        training_time = round(time.time() - start_time, 2)
        self.risk_metrics = test_metrics
        
        return {
            'modelType': 'classification',
            'algorithm': algorithm,
            'metrics': test_metrics,
            'trainMetrics': train_metrics,
            'cvMetrics': cv_metrics,
            'trainingTime': training_time,
            'samplesUsed': len(X),
            'featuresUsed': len(available_features),
            'classDistribution': {
                'low': int((y == 0).sum()),
                'medium': int((y == 1).sum()),
                'high': int((y == 2).sum())
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_load(self, features: Dict[str, Any], session_type: str = 'match') -> Dict[str, Any]:
        if self.load_pipeline is None:
            avg_load = features.get('Player Load', features.get('avgLoad', 300))
            multiplier = 1.15 if session_type == 'match' else 0.9
            return {
                'predictedLoad': float(avg_load) * multiplier,
                'confidence': 0.6,
                'method': 'average_based',
                'sessionType': session_type
            }
        
        feature_cols = self.load_features if self.load_features else list(features.keys())
        
        X = pd.DataFrame([features])
        
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0
        
        available_cols = [c for c in feature_cols if c in X.columns]
        X = X[available_cols]
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        prediction = self.load_pipeline.predict(X)[0]
        
        if session_type == 'match':
            prediction *= 1.15
        elif session_type == 'training':
            prediction *= 0.9
        
        return {
            'predictedLoad': float(prediction),
            'confidence': 0.85,
            'method': 'ml_model',
            'sessionType': session_type
        }
    
    def predict_risk(self, features: Dict[str, float]) -> Tuple[str, float, List[str], List[str]]:
        if self.risk_pipeline is None:
            return self._rule_based_risk_prediction(features)
        
        try:
            feature_cols = self.risk_features if self.risk_features else list(features.keys())
            
            X = pd.DataFrame([features])
            
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
            
            available_cols = [c for c in feature_cols if c in X.columns]
            
            if not available_cols:
                return self._rule_based_risk_prediction(features)
            
            X = X[available_cols]
            
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            prediction = self.risk_pipeline.predict(X)[0]
            proba = self.risk_pipeline.predict_proba(X)[0]
            
            risk_levels = ['low', 'medium', 'high']
            risk_level = risk_levels[int(prediction)]
            probability = float(max(proba))
            
            factors = self._get_risk_factors(features, risk_level)
            recommendations = self._get_recommendations(risk_level, features)
            
            return risk_level, probability, factors, recommendations
            
        except Exception as e:
            logger.warning(f"ML prediction failed, using rule-based: {e}")
            return self._rule_based_risk_prediction(features)
    
    def _rule_based_risk_prediction(self, features: Dict[str, float]) -> Tuple[str, float, List[str], List[str]]:
        player_load = features.get('Player Load', features.get('playerLoad', 0))
        work_ratio = features.get('Work Ratio', features.get('workRatio', 0))
        top_speed = features.get('Top Speed (mph)', features.get('topSpeed', 0))
        sprint_dist = features.get('Sprint Distance (yards)', features.get('sprintDistance', 0))
        
        risk_score = 0
        factors = []
        
        if player_load > 500:
            risk_score += 3
            factors.append(f"High player load ({player_load:.0f}) exceeds safe threshold")
        elif player_load > 300:
            risk_score += 1
            factors.append(f"Moderate player load ({player_load:.0f})")
        
        if work_ratio > 25:
            risk_score += 2
            factors.append(f"High work ratio ({work_ratio:.1f}) indicates fatigue")
        elif work_ratio > 15:
            risk_score += 1
            factors.append(f"Elevated work ratio ({work_ratio:.1f})")
        
        if top_speed > 20:
            risk_score += 1
            factors.append(f"High speed sessions ({top_speed:.1f} mph) increase strain")
        
        if sprint_dist > 600:
            risk_score += 1
            factors.append(f"High sprint distance ({sprint_dist:.0f} yards)")
        
        if risk_score >= 4:
            risk_level = 'high'
            probability = 0.85
        elif risk_score >= 2:
            risk_level = 'medium'
            probability = 0.65
        else:
            risk_level = 'low'
            probability = 0.80
            factors = ["All metrics within normal parameters"]
        
        recommendations = self._get_recommendations(risk_level, features)
        
        return risk_level, probability, factors, recommendations
    
    def _get_risk_factors(self, features: Dict[str, float], risk_level: str) -> List[str]:
        factors = []
        
        work_ratio = features.get('Work Ratio', features.get('workRatio', 0))
        sprint_dist = features.get('Sprint Distance (yards)', features.get('sprintDistance', 0))
        top_speed = features.get('Top Speed (mph)', features.get('topSpeed', 0))
        distance = features.get('Distance (miles)', features.get('distance', 0))
        player_load = features.get('Player Load', features.get('playerLoad', 0))
        
        if work_ratio > 20:
            factors.append(f"High work ratio ({work_ratio:.1f}) indicates fatigue accumulation")
        if sprint_dist > 500:
            factors.append(f"Elevated sprint distance ({sprint_dist:.0f} yards) increases muscle strain risk")
        if top_speed > 18:
            factors.append(f"High top speed ({top_speed:.1f} mph) sessions require adequate recovery")
        if distance > 5:
            factors.append(f"High total distance ({distance:.2f} miles) may lead to overload")
        if player_load > 500:
            factors.append(f"Accumulated load ({player_load:.0f}) above optimal threshold")
        
        if not factors:
            if risk_level == 'low':
                factors.append("All metrics within safe parameters")
            elif risk_level == 'medium':
                factors.append("Moderate training load detected")
            else:
                factors.append("Multiple metrics elevated")
        
        return factors
    
    def _get_recommendations(self, risk_level: str, features: Dict[str, float]) -> List[str]:
        if risk_level == 'high':
            return [
                "Reduce training intensity by 20-30%",
                "Focus on active recovery and regeneration",
                "Consider rest day or low-intensity session",
                "Monitor for signs of fatigue or discomfort",
                "Increase sleep and nutrition focus"
            ]
        elif risk_level == 'medium':
            return [
                "Maintain current training load with monitoring",
                "Ensure adequate recovery between sessions",
                "Monitor work ratio trends closely",
                "Consider preventive mobility work"
            ]
        else:
            return [
                "Player is in optimal condition",
                "Can maintain or slightly increase training load",
                "Continue monitoring key metrics",
                "Good foundation for high-intensity work"
            ]
    
    def get_model_status(self) -> Dict[str, Any]:
        load_status = {
            'trained': self.load_pipeline is not None,
            'algorithm': None,
            'metrics': self.load_metrics,
            'features': len(self.load_features) if self.load_features else 0
        }
        
        risk_status = {
            'trained': self.risk_pipeline is not None,
            'algorithm': None,
            'metrics': self.risk_metrics,
            'features': len(self.risk_features) if self.risk_features else 0
        }
        
        if self.load_pipeline is not None:
            try:
                if hasattr(self.load_pipeline, 'named_steps') and 'model' in self.load_pipeline.named_steps:
                    load_status['algorithm'] = type(self.load_pipeline.named_steps['model']).__name__
                elif hasattr(self.load_pipeline, 'steps'):
                    load_status['algorithm'] = type(self.load_pipeline.steps[-1][1]).__name__
            except:
                pass
        
        if self.risk_pipeline is not None:
            try:
                if hasattr(self.risk_pipeline, 'named_steps') and 'model' in self.risk_pipeline.named_steps:
                    risk_status['algorithm'] = type(self.risk_pipeline.named_steps['model']).__name__
                elif hasattr(self.risk_pipeline, 'steps'):
                    risk_status['algorithm'] = type(self.risk_pipeline.steps[-1][1]).__name__
            except:
                pass
        
        return {
            'loadModel': load_status['trained'],
            'riskModel': risk_status['trained'],
            'loadModelDetails': load_status,
            'riskModelDetails': risk_status,
            'availableAlgorithms': self.get_available_algorithms()
        }


ml_service = MLService()
