import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
import json
import requests
import re
import concurrent.futures
import logging
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PDF export requires reportlab. Install with: pip install reportlab")

from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Advanced ML extensions
try:
    from advanced_ml_extensions import (
        compute_cross_validation_metrics,
        generate_shap_analysis,
        plot_shap_summary,
        log_to_mlflow,
        display_cv_metrics,
        create_cv_box_plot,
        SHAP_AVAILABLE,
        MLFLOW_AVAILABLE
    )
except ImportError:
    SHAP_AVAILABLE = False
    MLFLOW_AVAILABLE = False
    logging.warning("Advanced ML extensions not available")

# Ollama integration for AI Coach Assistant
# OLLAMA_HOST will be read from secrets or environment when needed
# Default to localhost if not set
if "OLLAMA_HOST" not in os.environ:
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

# Set page config
st.set_page_config(
    page_title="Elite Sports Performance Analytics | Alvaro Martin-Pena",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - Modern UI Design System
st.markdown("""
<style>
    /* ===== Color Palette & Variables ===== */
    :root {
        --primary-blue: #1E88E5;
        --primary-green: #43A047;
        --accent-purple: #667eea;
        --accent-purple-dark: #764ba2;
        --success-green: #28A745;
        --warning-yellow: #FFC107;
        --danger-red: #DC3545;
        --info-blue: #2196F3;
        --bg-light: #F8F9FA;
        --bg-card: #FFFFFF;
        --text-primary: #212529;
        --text-secondary: #6C757D;
        --border-light: #E9ECEF;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
        --gradient-primary: linear-gradient(135deg, #1E88E5 0%, #43A047 100%);
        --gradient-card: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ===== Main Header ===== */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 2rem;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeInDown 0.6s ease-out;
    }
    
    /* ===== Metric Cards ===== */
    .metric-card {
        background: var(--gradient-card);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .metric-card-enhanced {
        background: var(--bg-card);
        border: 2px solid var(--border-light);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }
    
    .metric-card-enhanced:hover {
        border-color: var(--primary-blue);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* ===== Alert Boxes ===== */
    .warning-box {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFE69C 100%);
        border-left: 5px solid var(--warning-yellow);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.4s ease-out;
    }
    
    .success-box {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
        border-left: 5px solid var(--success-green);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.4s ease-out;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%);
        border-left: 5px solid var(--danger-red);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.4s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 5px solid var(--info-blue);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.4s ease-out;
    }
    
    /* ===== Chat Bubbles ===== */
    .chat-bubble-user {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--info-blue) 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: var(--shadow-md);
        animation: slideInRight 0.3s ease-out;
    }
    
    .chat-bubble-assistant {
        background: var(--bg-card);
        color: var(--text-primary);
        padding: 1rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* ===== Sidebar Enhancements ===== */
    .sidebar .stRadio > div {
        gap: 0.5rem;
    }
    
    .sidebar [data-testid="stRadio"] > div > label {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
    }
    
    .sidebar [data-testid="stRadio"] > div > label:hover {
        background-color: var(--bg-light);
    }
    
    .sidebar [data-testid="stRadio"] > div > label > div:first-child {
        border: 2px solid var(--primary-blue);
    }
    
    /* ===== Buttons ===== */
    .stButton > button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* ===== Tables ===== */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe thead {
        background: var(--gradient-primary);
        color: white;
    }
    
    .dataframe tbody tr:hover {
        background-color: var(--bg-light);
        transition: background-color 0.2s ease;
    }
    
    /* ===== Cards & Containers ===== */
    .section-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-light);
        animation: fadeIn 0.5s ease-out;
    }
    
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    /* ===== Animations ===== */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    /* ===== Loading States ===== */
    .loading-skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s ease-in-out infinite;
        border-radius: 8px;
        height: 20px;
        margin: 0.5rem 0;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* ===== Badges ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .badge-success {
        background-color: var(--success-green);
        color: white;
    }
    
    .badge-warning {
        background-color: var(--warning-yellow);
        color: #212529;
    }
    
    .badge-danger {
        background-color: var(--danger-red);
        color: white;
    }
    
    .badge-info {
        background-color: var(--info-blue);
        color: white;
    }
    
    /* ===== Scrollbar Styling ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-light);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-blue);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-green);
    }
    
    /* ===== Typography ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* ===== Spacing Improvements ===== */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* ===== Risk Cards ===== */
    .risk-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .risk-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .risk-card-high {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%);
        border: 2px solid var(--danger-red);
    }
    
    .risk-card-medium {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFE69C 100%);
        border: 2px solid var(--warning-yellow);
    }
    
    .risk-card-low {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
        border: 2px solid var(--success-green);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'regression_model' not in st.session_state:
    st.session_state.regression_model = None
if 'classification_model' not in st.session_state:
    st.session_state.classification_model = None
if 'regression_metrics' not in st.session_state:
    st.session_state.regression_metrics = None
if 'classification_metrics' not in st.session_state:
    st.session_state.classification_metrics = None
if 'regression_features' not in st.session_state:
    st.session_state.regression_features = None
if 'classification_features' not in st.session_state:
    st.session_state.classification_features = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_chat_player' not in st.session_state:
    st.session_state.selected_chat_player = None

# Setup logging and directories
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

logging.basicConfig(
    filename='logs/graceland_analytics.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Helper Functions for Model Persistence
def save_model(model, model_type, metrics=None, features=None):
    """Save trained model to disk"""
    try:
        model_path = f"models/{model_type}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'metrics': metrics,
                'features': features,
                'timestamp': datetime.now().isoformat()
            }, f)
        logging.info(f"Model saved: {model_path}")
        return model_path
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        return None

def load_model(model_type):
    """Load trained model from disk"""
    try:
        model_path = f"models/{model_type}_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Model loaded: {model_path}")
            return data['model'], data.get('metrics'), data.get('features')
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
    return None, None, None

def log_training_results(model_type, metrics, hyperparams=None):
    """Log model training results"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'metrics': metrics,
            'hyperparams': hyperparams
        }
        logging.info(f"Training {model_type} - Metrics: {metrics}")
        
        # Save to JSON
        log_file = 'logs/training_history.json'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Error logging training results: {str(e)}")

def plot_feature_importance(model, feature_names, top_n=15):
    """Visualize feature importance"""
    try:
        # Try to get feature importances from model or pipeline
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importances = model.named_steps['model'].feature_importances_
        
        if importances is None or len(importances) != len(feature_names):
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    except Exception as e:
        logging.error(f"Error plotting feature importance: {str(e)}")
        return None

def plot_overfitting_metrics(metrics):
    """Visualize train vs test metrics with Plotly"""
    try:
        fig = go.Figure()
        metrics_labels = ['R²', 'MAE (normalized)', 'RMSE (normalized)']
        
        # Train metrics
        fig.add_trace(go.Bar(
            name='Train',
            x=metrics_labels,
            y=[
                metrics['train'].get('R2', 0),
                metrics['train'].get('MAE', 0) / 100,
                metrics['train'].get('RMSE', 0) / 100
            ],
            marker_color='#1E88E5'
        ))
        
        # Test metrics
        fig.add_trace(go.Bar(
            name='Test',
            x=metrics_labels,
            y=[
                metrics['test'].get('R2', 0),
                metrics['test'].get('MAE', 0) / 100,
                metrics['test'].get('RMSE', 0) / 100
            ],
            marker_color='#43A047'
        ))
        
        fig.update_layout(
            title='Train vs Test Metrics (Overfitting Analysis)',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            height=400
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error plotting overfitting metrics: {str(e)}")
        return None

def safe_get_metric(value, default=0, format_str="{:.1f}"):
    """Safely get a metric value, handling NaN and None"""
    try:
        if pd.isna(value) or value is None or value == '':
            return default
        if isinstance(value, str):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

# Helper Functions
def limpiar_datos_regression(df):
    """Clean data for regression"""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    for col in cat_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] < limite_inferior, limite_inferior,
                                 np.where(df_clean[col] > limite_superior, limite_superior, df_clean[col]))
    
    return df_clean

def limpiar_datos_classification(df):
    """Clean data for classification with improved risk calculation"""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_clean[col] = np.where(df_clean[col] < limite_inferior, limite_inferior,
                                 np.where(df_clean[col] > limite_superior, limite_superior, df_clean[col]))
    
    # Professional injury risk calculation with balanced thresholds
    if 'Player Load' in df_clean.columns:
        # Calculate percentiles for clinically-based risk assessment
        load_95th = df_clean['Player Load'].quantile(0.95)  # Top 5% - truly elevated load
        load_80th = df_clean['Player Load'].quantile(0.80)  # Upper moderate
        load_60th = df_clean['Player Load'].quantile(0.60)  # Balanced threshold for medium risk
        
        # Check for additional risk factors
        has_energy = 'Energy (kcal)' in df_clean.columns
        has_impacts = 'Impacts' in df_clean.columns
        
        # Professional multi-factor risk assessment
        if has_energy and has_impacts:
            energy_85th = df_clean['Energy (kcal)'].quantile(0.85)
            impacts_high = df_clean['Impacts'].quantile(0.85)
            
            # HIGH RISK: Multiple risk factors present
            df_clean['injury_risk'] = np.where(
                # Severe high load OR (high load + high energy + high impacts)
                (df_clean['Player Load'] > load_95th) |
                ((df_clean['Player Load'] > load_80th) & 
                 (df_clean['Energy (kcal)'] > energy_85th) &
                 (df_clean['Impacts'] > impacts_high)),
                'high',
                np.where(
                    # MEDIUM RISK: Moderate elevation (60th to 95th percentile)
                    df_clean['Player Load'] > load_60th,
                    'medium',
                    # LOW RISK: Low load (below 60th percentile)
                    'low'
                )
            )
        elif has_energy:
            energy_85th = df_clean['Energy (kcal)'].quantile(0.85)
            df_clean['injury_risk'] = np.where(
                (df_clean['Player Load'] > load_95th) |
                ((df_clean['Player Load'] > load_80th) & (df_clean['Energy (kcal)'] > energy_85th)),
                'high',
                np.where(df_clean['Player Load'] > load_60th, 'medium', 'low')
            )
        else:
            # Simple risk based only on load
            df_clean['injury_risk'] = np.where(
                df_clean['Player Load'] > load_95th,
                'high',
                np.where(df_clean['Player Load'] > load_60th, 'medium', 'low')
            )
    else:
        df_clean['injury_risk'] = 'medium'
    
    return df_clean

def generate_intelligent_recommendations(player_name, player_data, recent_data):
    """
    Generate intelligent, data-driven recommendations for a specific player
    
    Parameters:
    -----------
    player_name : str
        Name of the player
    player_data : DataFrame
        Player's recent session data
    recent_data : DataFrame
        All recent data for comparison
    
    Returns:
    --------
    recommendations : dict
        Intelligent recommendations based on player's data
    """
    recommendations = {
        'risk_level': 'Medium',
        'key_insights': [],
        'specific_actions': [],
        'comparison_to_team': {},
        'trends': {},
        'personalized_plan': []
    }
    
    if 'Player Name' in player_data.columns:
        # Calculate comprehensive player metrics
        avg_load = player_data['Player Load'].mean() if 'Player Load' in player_data.columns else 0
        max_load = player_data['Player Load'].max() if 'Player Load' in player_data.columns else 0
        min_load = player_data['Player Load'].min() if 'Player Load' in player_data.columns else 0
        avg_energy = player_data['Energy (kcal)'].mean() if 'Energy (kcal)' in player_data.columns else 0
        max_speed = player_data['Top Speed (mph)'].max() if 'Top Speed (mph)' in player_data.columns else 0
        avg_speed = player_data['Top Speed (mph)'].mean() if 'Top Speed (mph)' in player_data.columns else 0
        total_impacts = player_data['Impacts'].sum() if 'Impacts' in player_data.columns else 0
        avg_impacts = player_data['Impacts'].mean() if 'Impacts' in player_data.columns else 0
        avg_distance = player_data['Distance (miles)'].mean() if 'Distance (miles)' in player_data.columns else 0
        avg_sprint = player_data['Sprint Distance (yards)'].mean() if 'Sprint Distance (yards)' in player_data.columns else 0
        avg_hr_load = player_data['Hr Load'].mean() if 'Hr Load' in player_data.columns else 0
        avg_power = player_data['Power Score (w/kg)'].mean() if 'Power Score (w/kg)' in player_data.columns else 0
        
        # Calculate team averages for comparison
        if 'Player Load' in recent_data.columns:
            team_avg_load = recent_data['Player Load'].mean()
            team_max_load = recent_data['Player Load'].max()
            team_median_load = recent_data['Player Load'].median()
            
            recommendations['comparison_to_team'] = {
                'load_vs_team': f"{((avg_load / team_avg_load - 1) * 100):.1f}% vs team average",
                'load_percentile': f"{(recent_data['Player Load'] > avg_load).mean() * 100:.0f}%"
            }
        
        # Team metrics for comparison
        if 'Energy (kcal)' in recent_data.columns:
            team_avg_energy = recent_data['Energy (kcal)'].mean()
            team_max_energy = recent_data['Energy (kcal)'].max()
        if 'Top Speed (mph)' in recent_data.columns:
            team_max_speed = recent_data['Top Speed (mph)'].max()
            team_avg_speed = recent_data['Top Speed (mph)'].mean()
        if 'Distance (miles)' in recent_data.columns:
            team_avg_distance = recent_data['Distance (miles)'].mean()
        if 'Sprint Distance (yards)' in recent_data.columns:
            team_avg_sprint = recent_data['Sprint Distance (yards)'].mean()
        if 'Hr Load' in recent_data.columns:
            team_avg_hr = recent_data['Hr Load'].mean()
        if 'Power Score (w/kg)' in recent_data.columns:
            team_avg_power = recent_data['Power Score (w/kg)'].mean()
        
        # Identify trends (if we have multiple sessions)
        if len(player_data) >= 2 and 'Date' in player_data.columns:
            player_data_sorted = player_data.sort_values('Date')
            
            # Check if load is increasing
            if 'Player Load' in player_data.columns:
                recent_load = player_data_sorted['Player Load'].tail(3).mean()
                older_load = player_data_sorted['Player Load'].head(3).mean()
                
                if recent_load > older_load * 1.15:
                    recommendations['trends']['load_trend'] = '⚠️ INCREASING LOAD TREND'
                    recommendations['trends']['load_change'] = f"{((recent_load / older_load - 1) * 100):.1f}% increase"
                elif recent_load < older_load * 0.85:
                    recommendations['trends']['load_trend'] = '✅ DECREASING LOAD TREND'
                    recommendations['trends']['load_change'] = f"{((1 - recent_load / older_load) * 100):.1f}% decrease"
                else:
                    recommendations['trends']['load_trend'] = '➡️ STABLE LOAD'
        
        # Generate specific insights based on data
        insights = []
        actions = []
        plan = []
        
        # Load-based insights with specific numbers
        if avg_load > 0:
            load_vs_team_pct = ((avg_load / team_avg_load - 1) * 100) if team_avg_load > 0 else 0
            if abs(load_vs_team_pct) > 15:
                insights.append(f"📊 Load Status: {avg_load:.1f} avg ({load_vs_team_pct:+.1f}% vs team avg of {team_avg_load:.1f})")
                
                if load_vs_team_pct > 15:
                    actions.append(f"⚠️ REDUCE intensity: Target {avg_load * 0.7:.0f} load in next 3 sessions (currently {avg_load:.0f})")
                    plan.append(f"📉 Gradual load reduction: Days 1-2 at {avg_load * 0.6:.0f}, Day 3 at {avg_load * 0.8:.0f}")
                elif load_vs_team_pct < -15:
                    insights.append("📈 Load is below team average - can increase intensity")
                    actions.append(f"💪 INCREASE intensity: Target {team_avg_load:.0f} load to reach team baseline")
                    plan.append(f"📈 Progressive overload: Increase by 10% weekly until reaching {team_avg_load:.0f}")
        
        if max_load > avg_load * 1.6:
            load_spike = ((max_load / avg_load) * 100) - 100
            insights.append(f"⚠️ LOAD SPIKES: Max ({max_load:.0f}) is {load_spike:.0f}% above average ({avg_load:.0f})")
            actions.append(f"🚨 Implement CAP: Don't exceed {avg_load * 0.8:.0f} in any session")
            plan.append(f"📋 Load management protocol: 3 sessions at max {avg_load * 0.85:.0f}, then reassess")
        
        # Consistency analysis
        if len(player_data) >= 3:
            loads = player_data['Player Load'].tolist() if 'Player Load' in player_data.columns else []
            if loads:
                cv = np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0
                load_variance = max_load - min_load
                
                if cv > 0.4:
                    insights.append(f"⚠️ INCONSISTENT TRAINING: {cv*100:.0f}% variation (range: {min_load:.0f}-{max_load:.0f})")
                    actions.append(f"📊 Standardize loads: Target {avg_load:.0f} ±{avg_load*0.2:.0f} for next 5 sessions")
                    plan.append(f"🎯 Load consistency program: Current variance {load_variance:.0f}, target {avg_load*0.3:.0f}")
        
        # Speed-specific recommendations
        if 'Top Speed (mph)' in recent_data.columns and max_speed > 0 and team_max_speed > 0:
            speed_deficit_pct = ((max_speed / team_max_speed) - 1) * 100
            if max_speed < team_max_speed * 0.85:
                insights.append(f"⚡ SPEED GAP: {max_speed:.1f} mph vs team max {team_max_speed:.1f} mph ({speed_deficit_pct:.0f}% below)")
                actions.append(f"🏃 Sprint development: Add 4-6 x 30m max speed sprints 2x/week")
                plan.append(f"🎯 Speed training plan: Target max speed {team_max_speed * 0.95:.1f} mph within 4 weeks")
            elif max_speed >= team_max_speed * 0.95:
                insights.append(f"✅ ELITE SPEED: {max_speed:.1f} mph ({((max_speed/team_max_speed - 1)*100):+.0f}% vs team max)")
                plan.append("⭐ Maintain speed: Preserve sprint ability with 1-2 maintenance sessions/week")
            
            # Sprint volume analysis
            if avg_sprint > 0 and 'Sprint Distance (yards)' in recent_data.columns:
                sprint_vs_team = (avg_sprint / team_avg_sprint) if team_avg_sprint > 0 else 1
                if avg_sprint < team_avg_sprint * 0.8:
                    plan.append(f"📊 Low sprint volume: {avg_sprint:.0f} yds vs team avg {team_avg_sprint:.0f} yds - Increase to 200-300yds/session")
                elif avg_sprint > team_avg_sprint * 1.2:
                    plan.append(f"⚡ High sprint volume: {avg_sprint:.0f} yds - Monitor for fatigue")
        
        # Energy efficiency analysis
        if avg_energy > 0 and avg_distance > 0:
            energy_per_mile = avg_energy / avg_distance
            team_avg_energy = recent_data['Energy (kcal)'].mean()
            team_avg_distance = recent_data['Distance (miles)'].mean()
            team_efficiency = team_avg_energy / team_avg_distance if team_avg_distance > 0 else 0
            
            if team_efficiency > 0:
                efficiency_ratio = energy_per_mile / team_efficiency
                if efficiency_ratio > 1.2:
                    insights.append(f"⚠️ LOW EFFICIENCY: {energy_per_mile:.0f} kcal/mile vs team {team_efficiency:.0f} ({efficiency_ratio:.1f}x)")
                    plan.append(f"💪 Efficiency improvement: {avg_energy:.0f} kcal over {avg_distance:.1f}mi - Focus on anaerobic threshold work")
                elif efficiency_ratio < 0.9:
                    insights.append(f"✅ EFFICIENT MOVEMENT: {energy_per_mile:.0f} kcal/mile ({((1-efficiency_ratio)*100):.0f}% more efficient)")
        
        # Impact analysis
        team_avg_impacts = recent_data['Impacts'].mean() if 'Impacts' in recent_data.columns else 0
        if total_impacts > 0 and team_avg_impacts > 0:
            impact_vs_team = avg_impacts / team_avg_impacts
            if impact_vs_team > 1.3:
                insights.append(f"⚡ HIGH IMPACTS: {avg_impacts:.0f}/session vs team {team_avg_impacts:.0f} ({impact_vs_team:.1f}x)")
                actions.append("🦵 Monitor legs: High impact load detected - prioritize recovery protocols")
                plan.append(f"🔄 Impact management: Current {avg_impacts:.0f} impacts/session - Add 1 extra recovery day this week")
            elif impact_vs_team < 0.7:
                plan.append("⚪ Low impact exposure - may need more contact training")
        
        # HR Load analysis
        if avg_hr_load > 0 and 'Hr Load' in recent_data.columns:
            hr_vs_team = (avg_hr_load / team_avg_hr) if team_avg_hr > 0 else 1
            if avg_hr_load > team_avg_hr * 1.3:
                insights.append(f"❤️ HIGH HR STRESS: {avg_hr_load:.1f} vs team {team_avg_hr:.1f} ({hr_vs_team:.1f}x)")
                plan.append(f"🫀 Cardiovascular stress high - Ensure 48h recovery between intense sessions")
            elif avg_hr_load < team_avg_hr * 0.8:
                plan.append(f"❤️ HR Capacity: {avg_hr_load:.1f} below team avg - Consider more high-intensity work")
        
        # Power analysis
        if avg_power > 0 and 'Power Score (w/kg)' in recent_data.columns:
            power_vs_team = (avg_power / team_avg_power) if team_avg_power > 0 else 1
            if avg_power < team_avg_power * 0.9:
                plan.append(f"💥 Power development: {avg_power:.2f} w/kg vs {team_avg_power:.2f} - Add plyometric training 2x/week")
            elif avg_power >= team_avg_power * 1.1:
                plan.append(f"✅ Power output: {avg_power:.2f} w/kg - Maintain with explosive training")
        
        # Session frequency and volume
        sessions_count = len(player_data)
        if sessions_count >= 3:
            avg_sessions_per_week = sessions_count / 2  # Assuming 2-week period
            plan.append(f"📅 Training frequency: {avg_sessions_per_week:.1f} sessions/week detected")
            if avg_sessions_per_week < 3:
                plan.append("⚙️ OPTIMIZATION: Consider increasing to 4-5 sessions/week for better adaptation")
        
        recommendations['key_insights'] = insights
        recommendations['specific_actions'] = actions
        recommendations['personalized_plan'] = plan
    
    return recommendations

def train_regression_model_fast(df, use_early_stopping=True, use_saved_model=True):
    """Fast regression model training with train/test split, early stopping, and model persistence"""
    df_clean = limpiar_datos_regression(df)
    target = 'Player Load'
    
    features = [
        'Work Ratio', 'Energy (kcal)', 'Distance (miles)', 'Sprint Distance (yards)',
        'Top Speed (mph)', 'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)',
        'Distance Per Min (yd/min)', 'Hr Load', 'Hr Max (bpm)', 'Time In Red Zone (min)',
        'Impacts', 'Impact Zones: > 20 G (Impacts)', 'Impact Zones: 15 - 20 G (Impacts)',
        'Power Plays', 'Power Score (w/kg)'
    ]
    
    features = [c for c in features if c in df_clean.columns]
    X = df_clean[features]
    y = df_clean[target]
    
    # Try to load existing model
    if use_saved_model:
        pipe, loaded_metrics, loaded_features = load_model('regression')
        if pipe is not None:
            logging.info("Using saved regression model")
            return pipe, loaded_metrics, loaded_features
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
    
    # Use early stopping if requested
    if use_early_stopping:
        model = GradientBoostingRegressor(
            n_estimators=1000,  # Large number for early stopping
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            validation_fraction=0.1,
            n_iter_no_change=10,  # Stop if no improvement for 10 iterations
            tol=1e-4,
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    
    # Predict on train and test
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    # Calculate metrics for both sets
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    metrics = {
        'train': {'R2': r2_train, 'MAE': mae_train, 'RMSE': rmse_train},
        'test': {'R2': r2_test, 'MAE': mae_test, 'RMSE': rmse_test}
    }
    
    # Log training results
    log_training_results('regression', metrics, {'use_early_stopping': use_early_stopping})
    
    # Compute cross-validation metrics (skip if data is too large for performance)
    cv_metrics = None
    if len(X) < 500 and 'compute_cross_validation_metrics' in globals():
        try:
            cv_metrics = compute_cross_validation_metrics(pipe, X, y, cv_folds=5, model_type='regression')
            metrics['cv'] = cv_metrics
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE and 'log_to_mlflow' in globals():
        try:
            log_to_mlflow(
                pipe, 
                'regression_model',
                metrics,
                params={'use_early_stopping': use_early_stopping}
            )
        except Exception as e:
            logging.warning(f"MLflow logging failed: {e}")
    
    # Save model
    save_model(pipe, 'regression', metrics, features)
    
    # Store in session state
    st.session_state.regression_metrics = metrics
    st.session_state.regression_features = features
    
    return pipe, metrics, features

def train_classification_model_fast(df, use_saved_model=True):
    """Fast classification model training with high accuracy and model persistence"""
    # Try to load existing model
    if use_saved_model:
        pipe, loaded_metrics, loaded_features = load_model('classification')
        if pipe is not None:
            logging.info("Using saved classification model")
            return pipe, loaded_metrics, loaded_features
    
    df_clean = limpiar_datos_classification(df)
    
    features = [
        'Work Ratio', 'Energy (kcal)', 'Distance (miles)', 'Sprint Distance (yards)',
        'Top Speed (mph)', 'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)',
        'Distance Per Min (yd/min)', 'Hr Load', 'Hr Max (bpm)', 'Time In Red Zone (min)',
        'Impacts', 'Impact Zones: > 20 G (Impacts)', 'Impact Zones: 15 - 20 G (Impacts)',
        'Power Plays', 'Power Score (w/kg)', 'Distance in Speed Zone 4 (miles)',
        'Distance in Speed Zone 5 (miles)', 'Time in HR Load Zone 85% - 96% Max HR (secs)'
    ]
    
    df_clean['injury_risk'] = df_clean['injury_risk'].map({'low': 0, 'medium': 1, 'high': 2})
    features = [f for f in features if f in df_clean.columns]
    
    X = df_clean[features]
    y = df_clean['injury_risk']
    
    # Split train/test (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    preprocessor = ColumnTransformer([('num', StandardScaler(), numeric_cols)])
    
    model = LGBMClassifier(
        num_leaves=31,
        max_depth=10,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(f_classif, k=15)),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    
    # Predict on train and test
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    # Calculate metrics for both sets
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    precision_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    recall_train = recall_score(y_train, y_pred_train, average='weighted')
    recall_test = recall_score(y_test, y_pred_test, average='weighted')
    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    metrics = {
        'train': {'Accuracy': acc_train, 'Precision': precision_train, 'Recall': recall_train, 'F1': f1_train},
        'test': {'Accuracy': acc_test, 'Precision': precision_test, 'Recall': recall_test, 'F1': f1_test}
    }
    
    # Log training results
    log_training_results('classification', metrics)
    
    # Compute cross-validation metrics (skip if data is too large for performance)
    if len(X) < 500 and 'compute_cross_validation_metrics' in globals():
        try:
            cv_metrics = compute_cross_validation_metrics(pipe, X, y, cv_folds=5, model_type='classification')
            metrics['cv'] = cv_metrics
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE and 'log_to_mlflow' in globals():
        try:
            log_to_mlflow(
                pipe,
                'classification_model',
                metrics
            )
        except Exception as e:
            logging.warning(f"MLflow logging failed: {e}")
    
    # Save model
    save_model(pipe, 'classification', metrics, features)
    
    # Store in session state
    st.session_state.classification_metrics = metrics
    st.session_state.classification_features = features
    
    return pipe, metrics, features

# Ollama AI Coach Assistant Functions
def find_player_in_dataframe(df_clean, player_name):
    """Find player in dataframe with flexible matching (case-insensitive, trimmed, fuzzy)"""
    if not player_name or 'Player Name' not in df_clean.columns:
        return None
    
    # Clean input name
    search_name = str(player_name).strip().lower()
    if not search_name:
        return None
    
    # Get all unique player names from dataframe
    available_players = df_clean['Player Name'].unique()
    
    # Try exact match (case-insensitive)
    for name in available_players:
        if pd.isna(name):
            continue
        name_clean = str(name).strip().lower()
        if name_clean == search_name:
            return str(name).strip()
    
    # Try contains match (case-insensitive) - both directions
    for name in available_players:
        if pd.isna(name):
            continue
        name_clean = str(name).strip().lower()
        if search_name in name_clean or name_clean in search_name:
            return str(name).strip()
    
    # Try word-by-word matching for multi-word names
    search_words = [w.strip() for w in search_name.split() if len(w.strip()) > 1]
    
    # For two-word names, check if both words match (order-independent)
    if len(search_words) == 2:
        for name in available_players:
            if pd.isna(name):
                continue
            name_str = str(name).strip().lower()
            name_words = [w.strip() for w in name_str.split() if len(w.strip()) > 1]
            if len(name_words) == 2:
                # Both words from search must be in name (order doesn't matter)
                if search_words[0] in name_words and search_words[1] in name_words:
                    return str(name).strip()
                # Or vice versa
                if name_words[0] in search_words and name_words[1] in search_words:
                    return str(name).strip()
    
    # Try fuzzy match - check if all significant words in search name are in player name
    significant_words = [w for w in search_words if len(w) > 2]
    if significant_words:
        for name in available_players:
            if pd.isna(name):
                continue
            name_str = str(name).strip().lower()
            # Check if all significant words from search are in the player name
            if all(word in name_str for word in significant_words):
                return str(name).strip()
    
    # Try reverse - all words from player name are in search
    for name in available_players:
        if pd.isna(name):
            continue
        name_str = str(name).strip().lower()
        name_words = [w.strip() for w in name_str.split() if len(w.strip()) > 1]
        significant_name_words = [w for w in name_words if len(w) > 2]
        if significant_name_words and all(word in search_name for word in significant_name_words):
            return str(name).strip()
    
    return None

def extract_player_names_from_text(text, available_players):
    """Extract player names mentioned in user text with improved matching"""
    if not text or not available_players:
        return []
    
    text_lower = text.lower()
    found_players = []
    
    # Check for each available player name with multiple matching strategies
    for player in available_players:
        player_str = str(player).strip()
        if not player_str or pd.isna(player):
            continue
        player_lower = player_str.lower()
        
        # Strategy 1: Exact match (case-insensitive) - most reliable
        if player_lower in text_lower:
            found_players.append(player_str)
            continue
        
        # Strategy 2: For two-word names, check if both words appear (order-independent)
        player_words = [w.strip() for w in player_lower.split() if len(w.strip()) > 1]
        if len(player_words) == 2:
            # Both words must be in text (they can be separated)
            word1_found = player_words[0] in text_lower
            word2_found = player_words[1] in text_lower
            if word1_found and word2_found:
                found_players.append(player_str)
                continue
        
        # Strategy 3: For names with 3+ words, check if at least 2 words match
        if len(player_words) >= 3:
            words_found = sum(1 for word in player_words if word in text_lower)
            if words_found >= 2:
                found_players.append(player_str)
                continue
        
        # Strategy 4: For single-word names, use word boundary matching
        if len(player_words) == 1 and len(player_words[0]) > 3:
            # Check if it appears as a whole word
            pattern = r'\b' + re.escape(player_words[0]) + r'\b'
            if re.search(pattern, text_lower):
                found_players.append(player_str)
                continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_players = []
    for player in found_players:
        player_lower_check = str(player).strip().lower()
        if player_lower_check not in seen:
            seen.add(player_lower_check)
            unique_players.append(player)
    
    return unique_players

def format_player_data_for_context(df_clean, player_name=None):
    """Format player data as context string for Ollama"""
    try:
        available_players = []
        if 'Player Name' in df_clean.columns:
            available_players = sorted(list(df_clean['Player Name'].unique()))
        
        if player_name and 'Player Name' in df_clean.columns:
            # Use flexible player search
            matched_player_name = find_player_in_dataframe(df_clean, player_name)
            if matched_player_name:
                player_data = df_clean[df_clean['Player Name'] == matched_player_name].copy()
            else:
                # Player not found - return info about available players
                return f"AVAILABLE_PLAYERS: {', '.join(available_players[:20])}{' (and more)' if len(available_players) > 20 else ''}\nREQUESTED_PLAYER: {player_name} NOT FOUND in dataset."
        else:
            player_data = df_clean.copy()
        
        # Key metrics to include
        key_metrics = [
            'Player Load', 'Energy (kcal)', 'Distance (miles)', 
            'Top Speed (mph)', 'Hr Load', 'Sprint Distance (yards)',
            'Power Score (w/kg)', 'Impacts', 'Work Ratio'
        ]
        
        available_metrics = [m for m in key_metrics if m in player_data.columns]
        
        # Calculate statistics
        stats_summary = []
        stats_summary.append(f"Number of sessions: {len(player_data)}")
        
        for metric in available_metrics:
            try:
                mean_val = player_data[metric].mean()
                max_val = player_data[metric].max()
                min_val = player_data[metric].min()
                stats_summary.append(f"{metric}: Average={mean_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f}")
            except:
                continue
        
        # Add speed zones if available
        speed_zone_cols = [col for col in player_data.columns if 'Speed Zone' in col and 'Distance' in col]
        if speed_zone_cols:
            stats_summary.append(f"\nSpeed Zone Distribution:")
            for zone_col in speed_zone_cols[:5]:  # First 5 zones
                try:
                    total_distance = player_data[zone_col].sum()
                    avg_distance = player_data[zone_col].mean()
                    stats_summary.append(f"  {zone_col}: Total={total_distance:.2f} miles, Avg={avg_distance:.2f} miles")
                except:
                    continue
        
        # Limit context size to avoid long prompts
        context_lines = []
        for line in stats_summary:
            context_lines.append(line)
            if len("\n".join(context_lines)) > 2000:
                break
        context = "\n".join(context_lines)
        
        if player_name:
            # Use the matched name if available
            matched_name = find_player_in_dataframe(df_clean, player_name)
            display_name = matched_name if matched_name else player_name
            context = f"Player: {display_name}\n{context}"
        else:
            context = f"Team Overall Data:\n{context}\nAvailable Players: {', '.join(available_players[:15])}{' (and more)' if len(available_players) > 15 else ''}"
            
        return context
    except Exception as e:
        logging.error(f"Error formatting player data: {str(e)}")
        return None

# Ollama configuration - supports both local and cloud deployments
def get_ollama_base_url() -> str:
    """Get Ollama base URL from Streamlit secrets, environment, or default to localhost."""
    # First try to get from Streamlit secrets (for Streamlit Cloud)
    try:
        if hasattr(st, 'secrets'):
            # Try dictionary-style access
            if isinstance(st.secrets, dict) and 'general' in st.secrets:
                ollama_host = st.secrets['general'].get('OLLAMA_HOST', None)
                if ollama_host:
                    return str(ollama_host).rstrip('/')
            # Try attribute-style access
            elif hasattr(st.secrets, 'general'):
                ollama_host = getattr(st.secrets.general, 'OLLAMA_HOST', None)
                if ollama_host:
                    return str(ollama_host).rstrip('/')
    except (AttributeError, KeyError, TypeError, Exception):
        pass
    
    # Fall back to environment variable
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip('/')

def is_ollama_reachable() -> bool:
    """Check if Ollama server is reachable (supports both local and remote URLs)."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        # Try connecting to configured Ollama URL
        base_url = get_ollama_base_url()
        # Use longer timeout for cloud deployments (Render free tier can be slow)
        is_cloud = not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost")
        timeout = 60 if is_cloud else 2  # 60 seconds for cloud, 2 for local
        requests.get(f"{base_url}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False

def check_model_status(model="llama3.2"):
    """Check if a specific model is installed and return status information."""
    if not OLLAMA_AVAILABLE:
        return {
            "available": False,
            "status": "error",
            "message": "Ollama library is not installed"
        }
    
    try:
        base_url = get_ollama_base_url()
        is_cloud = not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost")
        timeout = 60 if is_cloud else 2
        
        # Check if server is reachable
        try:
            health_response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if health_response.status_code != 200:
                return {
                    "available": False,
                    "status": "error",
                    "message": f"Ollama server returned status {health_response.status_code}"
                }
        except requests.exceptions.Timeout:
            return {
                "available": False,
                "status": "timeout",
                "message": "Ollama server is taking too long to respond. This is normal for free tier Render services - wait 30-60 seconds."
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "message": f"Cannot connect to Ollama: {str(e)}"
            }
        
        # Get list of models
        try:
            models_response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            models_data = models_response.json()
            models = models_data.get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check if our model is in the list
            model_found = model in model_names
            
            if model_found:
                # Get model details
                model_info = next((m for m in models if m.get('name') == model), {})
                size = model_info.get('size', 0)
                size_mb = size / (1024 * 1024) if size > 0 else 0
                
                return {
                    "available": True,
                    "status": "installed",
                    "message": f"✅ Model '{model}' is installed and ready",
                    "models": model_names,
                    "size_mb": size_mb
                }
            else:
                # Check if there's an active download (by checking if server is responsive but model missing)
                return {
                    "available": False,
                    "status": "missing",
                    "message": f"⚠️ Model '{model}' is not installed. Available models: {', '.join(model_names[:3]) if model_names else 'None'}",
                    "models": model_names,
                    "download_available": True
                }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "message": f"Error checking models: {str(e)}"
            }
            
    except Exception as e:
        return {
            "available": False,
            "status": "error",
            "message": f"Error checking model status: {str(e)}"
        }

def download_model_with_progress(model="llama3.2", progress_container=None):
    """Download Ollama model with progress bar display."""
    if not OLLAMA_AVAILABLE:
        return False, "Ollama library is not installed"
    
    try:
        base_url = get_ollama_base_url()
        is_cloud = not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost")
        timeout = 60 if is_cloud else 2
        
        # Check if model already exists
        try:
            models_response = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = models_data.get('models', [])
                model_names = [m.get('name', '') for m in models]
                if model in model_names:
                    return True, f"Model '{model}' is already installed!"
        except:
            pass
        
        # Initialize progress bar and status text FIRST (before any operations)
        progress_bar = None
        status_text = None
        if progress_container:
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()
            status_text.info("🔄 Initiating model download...")
            progress_bar.progress(0.05)
        
        # Wake up server first (especially important for free tier Render)
        if is_cloud:
            if status_text:
                status_text.info("⏳ Connecting to server...")
                progress_bar.progress(0.08) if progress_bar else None
            
            # Try to wake up the server with a simple health check
            try:
                wake_up_response = requests.get(f"{base_url}/api/tags", timeout=60)
                if wake_up_response.status_code == 200:
                    if status_text:
                        progress_bar.progress(0.12) if progress_bar else None
            except:
                # Ignore wake-up errors, try download anyway
                if status_text:
                    progress_bar.progress(0.12) if progress_bar else None
                pass
        
        # Start download using streaming API
        try:
            import time
            pull_url = f"{base_url}/api/pull"
            
            # Start the pull request (non-blocking) with better error handling
            download_initiated = False
            try:
                if status_text:
                    status_text.info("🔄 Starting download...")
                    progress_bar.progress(0.15) if progress_bar else None
                
                # Use longer timeout for cloud services
                download_timeout = 60 if is_cloud else 15
                
                response = requests.post(
                    pull_url,
                    json={"name": model},
                    timeout=download_timeout  # Much longer timeout for cloud services
                )
                
                if response.status_code in [200, 201]:
                    # Download initiated successfully
                    download_initiated = True
                    if status_text:
                        progress_bar.progress(0.15) if progress_bar else None
                else:
                    # Status code not 200/201, but download might have started anyway
                    if status_text:
                        progress_bar.progress(0.15) if progress_bar else None
                    
                    # Wait a moment and check if model appeared
                    time.sleep(2)
                    try:
                        check = requests.get(f"{base_url}/api/tags", timeout=10)
                        if check.status_code == 200:
                            check_models = check.json().get('models', [])
                            check_names = [m.get('name', '') for m in check_models]
                            if model in check_names:
                                if status_text:
                                    progress_bar.progress(1.0) if progress_bar else None
                                    status_text.success(f"✅ Model '{model}' is already available!")
                                return True, f"Model '{model}' is already available!"
                    except:
                        pass
            except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as timeout_err:
                # Timeout is OK - the download may still have started (especially for free tier Render)
                download_initiated = True  # Assume it started
                if status_text:
                    progress_bar.progress(0.2) if progress_bar else None
                    status_text.warning("⏳ Timeout (normal for free tier). Download may have started. Wait 1-2 min then click 'Check Model Status'.")
                
                # Give it one more check after a short wait
                time.sleep(5)
                try:
                    final_check = requests.get(f"{base_url}/api/tags", timeout=10)
                    if final_check.status_code == 200:
                        final_models = final_check.json().get('models', [])
                        final_names = [m.get('name', '') for m in final_models]
                        if model in final_names:
                            if status_text:
                                progress_bar.progress(1.0) if progress_bar else None
                                status_text.success(f"✅ Model '{model}' is available!")
                            return True, f"Model '{model}' is available!"
                except:
                    pass
                    
            except requests.exceptions.ConnectionError:
                if status_text:
                    progress_bar.progress(0.5) if progress_bar else None
                    status_text.error("❌ Cannot connect to server. Wait 30-60 sec (free tier) then try again.")
                return False, "Cannot connect to server. Wait 30-60 seconds (free tier) then try again."
            except Exception as e:
                error_msg = str(e)
                # Check if it's a timeout-related error
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    download_initiated = True
                    if status_text:
                        progress_bar.progress(0.2) if progress_bar else None
                        status_text.warning("⏳ Timeout (normal for free tier). Download may have started. Wait 1-2 min then check status.")
                else:
                    if status_text:
                        progress_bar.progress(0.3) if progress_bar else None
                        status_text.warning(f"⚠️ Error: {error_msg[:80]}...")
                    return False, f"Error: {error_msg[:100]}. Download may still have started - check status."
            
            # Monitor progress with polling (progress_bar and status_text already initialized above)
            if progress_container and download_initiated:
                if status_text:
                    status_text.info("📥 Download in progress (5-10 minutes)...")
                    progress_bar.progress(0.2) if progress_bar else None
                
                # Poll for progress (every 5 seconds, up to 15 checks = ~75 seconds)
                # After that, inform user to check manually (download continues in background)
                max_polls = 15  # 15 * 5 = 75 seconds of active monitoring
                poll_count = 0
                
                while poll_count < max_polls:
                    time.sleep(5)  # Wait 5 seconds between checks
                    poll_count += 1
                    
                    try:
                        # Check if model is now available
                        check_response = requests.get(f"{base_url}/api/tags", timeout=10)
                        if check_response.status_code == 200:
                            check_models = check_response.json().get('models', [])
                            check_names = [m.get('name', '') for m in check_models]
                            
                            if model in check_names:
                                # Success!
                                progress_bar.progress(1.0)
                                status_text.success(f"✅ Model '{model}' is now available!")
                                return True, f"Model '{model}' downloaded successfully!"
                            else:
                                # Still downloading - update progress estimate
                                # Estimate based on time (assuming 5-10 min average, but we only monitor for 75 sec)
                                estimated_progress = min(0.2 + (poll_count / max_polls) * 0.15, 0.35)
                                progress_bar.progress(estimated_progress)
                                elapsed_sec = poll_count * 5
                                if poll_count % 3 == 0:  # Update every 15 seconds
                                    status_text.info(f"📥 Downloading... ({elapsed_sec}s / ~5-10 min)")
                        else:
                            # Server issue, but continue checking
                            estimated_progress = min(0.15 + (poll_count / max_polls) * 0.20, 0.35)
                            progress_bar.progress(estimated_progress)
                            status_text.warning(f"Checking download status... ({poll_count}/{max_polls} checks)")
                    except Exception as e:
                        # Network issue, but continue
                        estimated_progress = min(0.15 + (poll_count / max_polls) * 0.20, 0.35)
                        progress_bar.progress(estimated_progress)
                        if poll_count % 3 == 0:  # Show update every 15 seconds
                            status_text.warning(f"Checking download status... ({poll_count}/{max_polls} checks)")
                    
                    # Update progress bar every iteration to show activity
                    estimated = min(0.15 + (poll_count / max_polls) * 0.20, 0.35)
                    progress_bar.progress(estimated)
                
                # Final check after monitoring period
                status_text.info("⏳ Monitoring complete. Checking final status...")
                progress_bar.progress(0.4)
                
                final_check = requests.get(f"{base_url}/api/tags", timeout=15)
                if final_check.status_code == 200:
                    final_models = final_check.json().get('models', [])
                    final_names = [m.get('name', '') for m in final_models]
                    if model in final_names:
                        progress_bar.progress(1.0)
                        status_text.success(f"✅ Model '{model}' is now available!")
                        return True, f"Model '{model}' downloaded successfully!"
                    else:
                        progress_bar.progress(0.5)
                        status_text.success("""
                        ⏳ **Download is running in the background**
                        
                        The model download has been initiated and continues even after monitoring stops.
                        
                        **What to do now:**
                        1. Wait 5-10 minutes for the download to complete (~2GB)
                        2. Click '🔄 Check Model Status' periodically to verify
                        3. The download continues even if you navigate away
                        
                        **Note:** This is normal behavior - large downloads take time, especially on free tier services.
                        """)
                        return False, "Download in progress. Check status in 5-10 minutes."
                else:
                    progress_bar.progress(0.4)
                    status_text.warning("⏳ Download initiated. Use 'Check Model Status' to verify progress.")
                    return False, "Download started. Please check status in a few minutes."
                    
            else:
                # No progress container, just wait a bit and check
                time.sleep(5)
                check_response = requests.get(f"{base_url}/api/tags", timeout=10)
                if check_response.status_code == 200:
                    check_models = check_response.json().get('models', [])
                    check_names = [m.get('name', '') for m in check_models]
                    if model in check_names:
                        return True, f"Model '{model}' downloaded successfully!"
                return False, "Download initiated. Please wait 5-10 minutes and check status."
                
        except requests.exceptions.Timeout:
            return False, "Download timeout - but download may still be in progress. Please wait 5-10 minutes."
        except Exception as e:
            return False, f"Download error: {str(e)}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_ollama_response(user_message, player_context=None, chat_history=None, model="llama3.2"):
    """Get response from Ollama with player context, using timeouts to prevent blocking"""
    try:
        if not OLLAMA_AVAILABLE:
            return "Ollama library is not installed. Install it with: pip install ollama"

        # Health check to ensure Ollama server is responsive
        base_url = get_ollama_base_url()
        is_cloud = not base_url.startswith("http://127.0.0.1") and not base_url.startswith("http://localhost")
        health_timeout = 60 if is_cloud else 2  # Longer timeout for cloud services
        
        try:
            health_response = requests.get(f"{base_url}/api/tags", timeout=health_timeout)
            # Check if models are available
            if health_response.status_code == 200:
                try:
                    models = health_response.json().get('models', [])
                    model_names = [m.get('name', '') for m in models]
                    if model and model not in model_names:
                        # Model not found - try to download it automatically
                        logging.info(f"Model '{model}' not found, attempting to download...")
                        try:
                            # Use Ollama API to pull the model (start download, don't wait for completion)
                            # Use a shorter timeout to start the download, then let it continue in background
                            pull_response = requests.post(
                                f"{base_url}/api/pull",
                                json={"name": model},
                                timeout=10  # Short timeout just to initiate the download
                            )
                            if pull_response.status_code in [200, 201]:
                                logging.info(f"Successfully initiated download of model '{model}'")
                                return f"⏳ Model '{model}' download has been initiated (this may take 5-10 minutes).\n\n💡 **What to do:**\n1. Wait 5-10 minutes for the download to complete\n2. Check the '🔄 Check Model Status' button above to see progress\n3. Try your question again after a few minutes\n\n📊 You can also check download progress in Render logs: https://dashboard.render.com"
                            else:
                                # If pull fails immediately, try to check status again
                                try:
                                    # Wait a moment and check if download started anyway
                                    import time
                                    time.sleep(2)
                                    recheck = requests.get(f"{base_url}/api/tags", timeout=health_timeout)
                                    if recheck.status_code == 200:
                                        recheck_models = recheck.json().get('models', [])
                                        recheck_names = [m.get('name', '') for m in recheck_models]
                                        if model in recheck_names:
                                            # Model appeared! Continue with the request
                                            pass  # Will fall through to continue processing
                                        else:
                                            return f"Model '{model}' download may be in progress. Status code: {pull_response.status_code}. Please wait 5-10 minutes and try again, or check Render logs."
                                    else:
                                        return f"Model '{model}' is not installed. Download attempt returned status {pull_response.status_code}. Please wait a few minutes and try again."
                                except:
                                    return f"Model '{model}' download initiated. Please wait 5-10 minutes and try again. Status: {pull_response.status_code}"
                        except requests.exceptions.Timeout:
                            # Timeout is OK - download may have started
                            return f"⏳ Model '{model}' download initiated (timeout is normal - download continues in background).\n\nPlease wait 5-10 minutes, then:\n1. Click '🔄 Check Model Status' above to verify\n2. Try your question again\n\n📊 Check progress: https://dashboard.render.com → ollama-server → Logs"
                        except Exception as pull_error:
                            logging.warning(f"Failed to auto-download model: {str(pull_error)}")
                            # Check one more time if model magically appeared
                            try:
                                final_check = requests.get(f"{base_url}/api/tags", timeout=5)
                                if final_check.status_code == 200:
                                    final_models = final_check.json().get('models', [])
                                    final_names = [m.get('name', '') for m in final_models]
                                    if model in final_names:
                                        # Model is there! Continue
                                        pass
                                    else:
                                        return f"⚠️ Model '{model}' download error: {str(pull_error)}. The download may still be in progress. Please wait 5-10 minutes and try again. Check Render logs for details."
                            except:
                                return f"⚠️ Model '{model}' download issue: {str(pull_error)}. Please wait a few minutes and try again, or check Render logs."
                except:
                    pass  # If parsing fails, continue anyway
        except requests.exceptions.Timeout:
            return f"Ollama server at {base_url} is taking too long to respond. This is normal for free tier Render services (they spin down after inactivity). Please wait 30-60 seconds and try again."
        except Exception as e:
            return f"Cannot connect to Ollama at {base_url}. Error: {str(e)}. For cloud deployments, ensure the service is running and OLLAMA_HOST is set correctly."

        # Build system prompt
        system_prompt = (
            "You are an expert football (soccer) performance analytics assistant. "
            "Your job is to explain performance metrics, analyze player data, and provide useful coaching insights. "
            "Always answer in clear, professional, and concise English. "
            "When player data is provided, ground your explanations and recommendations in that data. "
            "Be friendly and conversational for casual questions like greetings. "
            "IMPORTANT: If the context mentions 'NOT FOUND' or 'AVAILABLE_PLAYERS', that means the requested player doesn't exist in the dataset. "
            "In that case, politely inform the user that the player is not in the dataset and mention some available players from the list provided. "
            "Never make up data for players that don't exist. "
            "When multiple players are mentioned (e.g., 'compare X and Y'), analyze ALL players mentioned using their provided data sections. "
            "If you see multiple '---PLAYER: [Name]---' sections, each contains data for a different player - use ALL of them in your response."
        )

        # Prepare messages (include system)
        messages = [{"role": "system", "content": system_prompt}]

        # Limit chat history to last 4 messages to keep prompt small
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
            messages.extend(recent_history)

        # Build user message with compact context (always include team data if available)
        context_prefix = ""
        if player_context:
            # Truncate context to ~1000 chars max for speed
            trimmed_context = player_context[:1000]
            context_prefix = f"DATA:\n{trimmed_context}\n\n"

        user_prompt = f"{context_prefix}Q: {user_message}\nA:"
        messages.append({"role": "user", "content": user_prompt})

        # Optimized direct call - allow longer responses (2000+ characters)
        # The ollama library automatically uses OLLAMA_HOST environment variable
        try:
            # Use llama3.2 with increased token limit for detailed responses
            # Note: ollama library handles timeouts internally, but cloud services may need patience
            resp = ollama.chat(
                model="llama3.2",
                messages=messages,
                options={
                    'temperature': 0.7, 
                    'num_predict': 512,  # Reduced for faster responses on cloud
                    'top_k': 40,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            if resp and 'message' in resp and 'content' in resp['message']:
                content = resp['message']['content'].strip()
                return content if content else "I'm ready to help! Please ask your question."
        except Exception as e1:
            error_msg = str(e1)
            logging.warning(f"Error with llama3.2: {error_msg}")
            
            # Check if it's a connection/timeout error
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                return f"Ollama server at {base_url} is slow to respond. For Render free tier, this is normal. Please wait 30-60 seconds for the service to wake up, then try again."
            
            # Quick fallback - try with simpler message
            try:
                resp = ollama.chat(
                    model="llama3.2",
                    messages=[messages[-1]],  # Just last message, no history for speed
                    options={'num_predict': 256}  # Even shorter for fallback
                )
                if resp and 'message' in resp and 'content' in resp['message']:
                    return resp['message']['content'].strip()
            except Exception as e2:
                logging.warning(f"Error with fallback: {str(e2)}")
                return f"Could not get response from Ollama. Error: {str(e2)}. If using Render free tier, wait 30-60 seconds and try again."

        return "Could not get a response. Please try again. If using Render free tier, wait 30-60 seconds for the service to wake up."
    except Exception as e:
        logging.error(f"Error getting Ollama response: {str(e)}")
        return f"Assistant temporarily unavailable. {str(e)}"

# Enhanced Sidebar Navigation
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <h1 style="font-size: 1.5rem; font-weight: 700; color: #1E88E5; margin: 0;">Elite Sports Analytics</h1>
    <p style="font-size: 0.85rem; color: #6C757D; margin-top: 0.5rem;">Performance Intelligence Platform</p>
    <hr style="margin: 1rem 0; border: none; border-top: 1px solid #E9ECEF;">
    <p style="font-size: 0.75rem; color: #6C757D; margin-top: 0.5rem; font-style: italic;">Developed by <strong>Alvaro Martin-Pena</strong></p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Group navigation items
st.sidebar.markdown("**Main Sections**")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Audit", "Model Training", "Player Load Analysis", 
     "Injury Prevention", "Team Lineup Calculator", "Performance Analytics", "Load Prediction"],
    label_visibility="collapsed"
)

# Add status indicators and interactive filters in sidebar if data is loaded
if st.session_state.df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Status**")
    df_status = st.session_state.df
    total_players = df_status['Player Name'].nunique() if 'Player Name' in df_status.columns else 0
    st.sidebar.markdown(f"""
    <div style="background: #E3F2FD; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
        <div style="font-size: 0.85rem; color: #2196F3; font-weight: 600;">Data Loaded</div>
        <div style="font-size: 0.75rem; color: #6C757D; margin-top: 0.25rem;">{total_players} players • {df_status.shape[0]:,} sessions</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.regression_model and st.session_state.classification_model:
        st.sidebar.markdown(f"""
        <div style="background: #D4EDDA; padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
            <div style="font-size: 0.85rem; color: #28A745; font-weight: 600;">Models Ready</div>
        </div>
        """, unsafe_allow_html=True)
    


# Main Content
if page == "Dashboard":
    st.markdown('<h1 class="main-header">Dashboard - Quick Overview</h1>', unsafe_allow_html=True)
    st.markdown("**Welcome to Elite Sports Performance Analytics | Developed by Alvaro Martin-Pena**")
    st.markdown("---")
    
    if st.session_state.df is None:
        st.info("Please upload your dataset in the Data Audit section to begin.")
        
        st.markdown("### Quick Start Guide")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1. Upload Data
            Go to **Data Audit** section and upload your CSV file with player performance data.
            """)
        
        with col2:
            st.markdown("""
            #### 2. Train Models
            Train the regression and classification models in **Model Training** section.
            """)
        
        with col3:
            st.markdown("""
            #### 3. Analyze
            Explore player loads, injury risks, and performance analytics!
            """)
    else:
        df = st.session_state.df
        df_filtered = df.copy()
        
        # Interactive Filter Section
        with st.expander("Advanced Filters & Settings", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Player search
                if 'Player Name' in df_filtered.columns:
                    all_player_names = ['All Players'] + sorted(list(df_filtered['Player Name'].unique()))
                    search_term = st.text_input("Search Player", value="", key="player_search", placeholder="Type player name...")
                    if search_term:
                        all_player_names = [p for p in all_player_names if search_term.lower() in p.lower()]
                    selected_player_filter = st.selectbox("Filter by Player", all_player_names, key="dashboard_player_filter")
                    if selected_player_filter != "All Players":
                        df_filtered = df_filtered[df_filtered['Player Name'] == selected_player_filter]
                else:
                    selected_player_filter = "All Players"
            
            with col2:
                # Metric selection for visualization
                if 'Player Load' in df_filtered.columns:
                    metric_options = ['Player Load', 'Energy (kcal)', 'Distance (miles)', 
                                     'Top Speed (mph)', 'Hr Load'] if all(m in df_filtered.columns for m in ['Energy (kcal)', 'Distance (miles)', 'Top Speed (mph)', 'Hr Load']) else ['Player Load']
                    primary_metric = st.selectbox("Primary Metric", metric_options, key="dashboard_primary_metric")
                else:
                    primary_metric = None
            
            with col3:
                # Show comparison
                show_comparison = st.checkbox("Show Team Comparison", value=False, key="dashboard_show_comparison")
            
            with col4:
                # Time period
                time_period = st.selectbox("Time Period", ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"], key="dashboard_time_period")
                
                # Export data button
                if st.button("Export Filtered Data", key="export_dashboard"):
                    csv = df_filtered.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Apply time period filter
        if time_period != "All Time" and 'Date' in df_filtered.columns:
            try:
                df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
                days = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}[time_period]
                cutoff_date = datetime.now() - timedelta(days=days)
                df_filtered = df_filtered[df_filtered['Date'] >= cutoff_date]
            except:
                pass
        
        # Key Metrics Dashboard - Enhanced with cards
        st.markdown("### Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_players = df_filtered['Player Name'].nunique() if 'Player Name' in df_filtered.columns else 0
            st.markdown(f"""
            <div class="metric-card-enhanced" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.25rem;">{total_players}</div>
                <div style="font-size: 0.9rem; color: #6C757D;">Total Players</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_sessions = df_filtered.shape[0]
            st.markdown(f"""
            <div class="metric-card-enhanced" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #43A047; margin-bottom: 0.25rem;">{total_sessions:,}</div>
                <div style="font-size: 0.9rem; color: #6C757D;">Total Sessions</div>
                {f'<div style="font-size: 0.75rem; color: #6C757D; margin-top: 0.25rem;">From {df.shape[0]:,} total</div>' if df_filtered.shape[0] < df.shape[0] else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if primary_metric and primary_metric in df_filtered.columns:
                avg_metric = df_filtered[primary_metric].mean()
                max_metric = df_filtered[primary_metric].max()
                st.markdown(f"""
                <div class="metric-card-enhanced" style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #667eea; margin-bottom: 0.25rem;">{avg_metric:.1f}</div>
                    <div style="font-size: 0.9rem; color: #6C757D;">Avg {primary_metric}</div>
                    <div style="font-size: 0.75rem; color: #6C757D; margin-top: 0.25rem;">Max: {max_metric:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if 'Date' in df_filtered.columns:
                    try:
                        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
                        recent_sessions = df_filtered[df_filtered['Date'] >= (datetime.now() - timedelta(days=7))].shape[0]
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center;">
                            <div style="font-size: 2rem; font-weight: 700; color: #212529; margin-bottom: 0.25rem;">{recent_sessions}</div>
                            <div style="font-size: 0.9rem; color: #6C757D;">Sessions (7 days)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    except:
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center;">
                            <div style="font-size: 1.2rem; font-weight: 700; color: #212529; margin-bottom: 0.25rem;">N/A</div>
                            <div style="font-size: 0.9rem; color: #6C757D;">Sessions (7 days)</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card-enhanced" style="text-align: center;">
                        <div style="font-size: 1.2rem; font-weight: 700; color: #212529; margin-bottom: 0.25rem;">N/A</div>
                        <div style="font-size: 0.9rem; color: #6C757D;">Sessions (7 days)</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.regression_model is not None and st.session_state.classification_model is not None:
                st.markdown(f"""
                <div class="metric-card-enhanced" style="text-align: center; border-color: #28A745;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #212529; margin-bottom: 0.25rem;">Ready</div>
                    <div style="font-size: 0.9rem; color: #6C757D;">Models Status</div>
                    <span class="badge badge-success" style="margin-top: 0.5rem; display: inline-block;">Trained</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card-enhanced" style="text-align: center; border-color: #FFC107;">
                    <div style="font-size: 1.2rem; font-weight: 700; color: #212529; margin-bottom: 0.25rem;">Not Trained</div>
                    <div style="font-size: 0.9rem; color: #6C757D;">Models Status</div>
                    <span class="badge badge-warning" style="margin-top: 0.5rem; display: inline-block;">Action Needed</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive Visualizations
        if primary_metric and primary_metric in df_filtered.columns:
            st.markdown("---")
            st.markdown("### Interactive Performance Analysis")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Metric distribution - Interactive Plotly
                fig_dist = px.histogram(
                    df_filtered, 
                    x=primary_metric, 
                    nbins=30,
                    title=f'{primary_metric} Distribution',
                    labels={primary_metric: primary_metric, 'count': 'Frequency'},
                    color_discrete_sequence=['#1E88E5']
                )
                fig_dist.update_layout(
                    hovermode='x unified',
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col_viz2:
                # Player comparison if selected - Interactive Plotly
                if show_comparison and 'Player Name' in df_filtered.columns and df_filtered['Player Name'].nunique() > 1:
                    player_avg = df_filtered.groupby('Player Name')[primary_metric].mean().sort_values(ascending=False).head(10)
                    player_avg_df = player_avg.reset_index()
                    
                    fig_comp = px.bar(
                        player_avg_df,
                        x=primary_metric,
                        y='Player Name',
                        orientation='h',
                        title=f'Top Players - Avg {primary_metric}',
                        labels={primary_metric: primary_metric},
                        color=primary_metric,
                        color_continuous_scale='Greens'
                    )
                    fig_comp.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        hovermode='y unified'
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("---")
        
        # Alert Box for High Risk Players
        if st.session_state.classification_model is not None:
            st.markdown("### 🚨 Injury Risk Overview")
            
            try:
                df_clean = limpiar_datos_classification(df)
                
                # Get recent data
                if 'Date' in df_clean.columns:
                    try:
                        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
                        max_date = df_clean['Date'].max()
                        two_weeks_ago = max_date - timedelta(days=14)
                        recent_data = df_clean[df_clean['Date'] >= two_weeks_ago]
                    except:
                        recent_data = df_clean.tail(min(50, len(df_clean)))
                else:
                    recent_data = df_clean.tail(min(50, len(df_clean)))
                
                # Calculate risk if we have data
                if len(recent_data) > 0 and 'Player Name' in recent_data.columns:
                    features = [f for f in ['Work Ratio', 'Energy (kcal)', 'Distance (miles)', 'Sprint Distance (yards)',
                        'Top Speed (mph)', 'Hr Load', 'Impacts'] if f in recent_data.columns]
                    
                    if len(features) > 0:
                        X_recent = recent_data[features]
                        if len(X_recent) > 0:
                            try:
                                predictions = st.session_state.classification_model.predict(X_recent)
                                recent_data['Predicted_Risk'] = predictions
                                
                                # Group by player (using mean and max)
                                player_risk = recent_data.groupby('Player Name')['Predicted_Risk'].agg(['mean', 'max']).reset_index()
                                # Risk classification: Low if avg_risk <= 0.40
                                risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
                                def classify_risk(row):
                                    mean_val = float(row['mean'])
                                    max_val = float(row['max'])
                                    # Low Risk: Average risk <= 0.40
                                    if mean_val <= 0.40:
                                        return 'Low'
                                    # Otherwise, use maximum risk to classify
                                    elif max_val >= 2.0:  # High risk class
                                        return 'High'
                                    elif max_val >= 1.0:  # Medium risk class
                                        return 'Medium'
                                    else:
                                        return 'Low'
                                player_risk['Risk_Category'] = player_risk.apply(classify_risk, axis=1)
                                
                                high_risk_count = len(player_risk[player_risk['Risk_Category'] == 'High'])
                                medium_risk_count = len(player_risk[player_risk['Risk_Category'] == 'Medium'])
                                low_risk_count = len(player_risk[player_risk['Risk_Category'] == 'Low'])
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="risk-card risk-card-high">
                                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚨</div>
                                        <h3 style='color:#DC3545; margin:0; font-size: 2.5rem; font-weight: 700;'>{high_risk_count}</h3>
                                        <p style='margin:0.5rem 0; font-weight: 600; color: #212529;'>High Risk Players</p>
                                        <span class="badge badge-danger">Attention Required</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="risk-card risk-card-medium">
                                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                                        <h3 style='color:#FFC107; margin:0; font-size: 2.5rem; font-weight: 700;'>{medium_risk_count}</h3>
                                        <p style='margin:0.5rem 0; font-weight: 600; color: #212529;'>Medium Risk</p>
                                        <span class="badge badge-warning">Monitor</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="risk-card risk-card-low">
                                        <div style="font-size: 3rem; margin-bottom: 0.5rem;">✅</div>
                                        <h3 style='color:#28A745; margin:0; font-size: 2.5rem; font-weight: 700;'>{low_risk_count}</h3>
                                        <p style='margin:0.5rem 0; font-weight: 600; color: #212529;'>Low Risk</p>
                                        <span class="badge badge-success">Healthy</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if high_risk_count > 0:
                                    high_risk_players = player_risk[player_risk['Risk_Category'] == 'High']['Player Name'].tolist()
                                    st.markdown(f"""
                                    <div class="danger-box">
                                    <h4>⚠️ URGENT: {high_risk_count} High Risk Players Detected</h4>
                                    <p><b>Players:</b> {', '.join(high_risk_players)}</p>
                                    <p>Please visit the <b>🛡️ Injury Prevention</b> section for detailed analysis and recommendations.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if st.button("🔍 View Detailed Analysis", type="primary"):
                                        st.session_state.page_from_dashboard = "Injury Prevention"
                                        st.rerun()
                                
                            except Exception as e:
                                st.warning(f"Could not calculate risk: {str(e)}")
            
            except Exception as e:
                st.warning(f"Could not load injury risk data: {str(e)}")
        
        # Quick Actions - Enhanced
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("View Player Loads", use_container_width=True, type="primary"):
                st.session_state.page_from_dashboard = "Player Load Analysis"
                st.rerun()
        
        with col2:
            if st.button("Team Lineup", use_container_width=True, type="primary"):
                st.session_state.page_from_dashboard = "Team Lineup Calculator"
                st.rerun()
        
        with col3:
            if st.button("Analytics", use_container_width=True, type="primary"):
                st.session_state.page_from_dashboard = "Performance Analytics"
                st.rerun()

elif page == "Data Audit":
    st.markdown('<h1 class="main-header">Data Audit & Quality Control</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.markdown(f"""
        <div class="success-box">
            <h4 style="margin: 0 0 0.5rem 0;">✅ Dataset Successfully Loaded</h4>
            <p style="margin: 0;"><strong>{df.shape[0]:,}</strong> rows • <strong>{df.shape[1]}</strong> columns</p>
        </div>
        """, unsafe_allow_html=True)

        # Validate required columns for Injury Risk calculations
        required_columns = {
            'Time in HR Load Zone 85% - 96% Max HR (secs)',
            'Hr Max (bpm)',
            'Impact Zones: 15 - 20 G (Impacts)',
            'Max Acceleration (yd/s/s)',
            'Power Plays',
            'Max Deceleration (yd/s/s)',
            'Impact Zones: > 20 G (Impacts)',
            'Power Score (w/kg)',
            'Distance Per Min (yd/min)',
            'Time In Red Zone (min)'
        }
        df_columns_set = set(df.columns)
        missing_required = sorted(list(required_columns - df_columns_set))
        if missing_required:
            st.error(
                "Missing required columns for Injury Risk calculations: " + ", ".join(missing_required)
            )
            st.info(
                "Please upload a CSV that includes these fields. The current file will still load, but Injury Risk cannot be computed until the missing columns are provided."
            )
        
        col1, col2, col3 = st.columns(3)
        total_players = df['Player Name'].nunique() if 'Player Name' in df.columns else 0
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-enhanced" style="text-align: center;">
                <div style="font-size: 2rem; color: #1E88E5; margin-bottom: 0.5rem;">👥</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #212529;">{total_players}</div>
                <div style="font-size: 0.85rem; color: #6C757D;">Total Players</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card-enhanced" style="text-align: center;">
                <div style="font-size: 2rem; color: #43A047; margin-bottom: 0.5rem;">📊</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #212529;">{df.shape[0]:,}</div>
                <div style="font-size: 0.85rem; color: #6C757D;">Total Sessions</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card-enhanced" style="text-align: center;">
                <div style="font-size: 2rem; color: #667eea; margin-bottom: 0.5rem;">📈</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #212529;">{df.shape[1]}</div>
                <div style="font-size: 0.85rem; color: #6C757D;">Features</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, height=400)
        
        st.markdown("### 📊 Data Quality Report")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <strong>📊 Numeric Variables:</strong> <span class="badge badge-info">{len(numeric_cols)}</span>
            </div>
            <div class="info-box">
                <strong>📝 Categorical Variables:</strong> <span class="badge badge-info">{len(cat_cols)}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Missing Values:</strong> <span class="badge badge-warning">{nulls.sum():,} total</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <strong>✅ No Missing Values</strong> <span class="badge badge-success">Perfect</span>
                </div>
                """, unsafe_allow_html=True)
        
        if 'Player Load' in df.columns:
            st.markdown("### 📈 Player Load Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df['Player Load'], kde=True, color='#1E88E5', ax=ax)
            ax.set_title('Player Load Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("### Outlier Detection & Cleaning")
        
        col_info_outlier, col_btn_outlier = st.columns([3, 1])
        with col_info_outlier:
            st.markdown("""
            **Outlier Removal Strategy:**
            - Removes rows with MANY zeros across critical metrics (data-quality cleanup)
            - Removes **very extreme outliers** (beyond 4.5 standard deviations or 4.5×IQR)
            - A row is removed only if flagged as outlier in **multiple critical metrics** (conservative approach)
            - Preserves most data while removing only the most exaggerated values
            """)
        with col_btn_outlier:
            if st.button("Remove Extreme Outliers", type="primary", use_container_width=True):
                with st.spinner("Detecting and removing extreme outliers..."):
                    # Detect outliers using z-score method (only very extreme values) and drop zeros in critical metrics
                    df_clean = df.copy()
                    original_rows = len(df_clean)
                    
                    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    # Focus only on critical performance metrics for outlier detection
                    critical_metrics = ['Player Load', 'Energy (kcal)', 'Distance (miles)', 
                                      'Top Speed (mph)', 'Hr Load', 'Sprint Distance (yards)',
                                      'Power Score (w/kg)', 'Impacts']
                    critical_cols = [col for col in numeric_cols if any(metric in col for metric in critical_metrics)]
                    
                    if not critical_cols:
                        critical_cols = numeric_cols[:5]  # Use first 5 if no critical metrics found
                    
                    outlier_scores = pd.Series([0] * len(df_clean))  # Track how many columns flag a row as outlier
                    zero_counts = pd.Series([0] * len(df_clean))     # Track how many critical metrics are zero per row
                    
                    for col in critical_cols:
                        if col in df_clean.columns:
                            try:
                                # Count zeros per row across critical metrics (we'll threshold later)
                                zero_counts += (df_clean[col].fillna(0) == 0.0).astype(int)
                                
                                # Method 1: Z-score - only VERY extreme outliers (beyond 4 standard deviations)
                                col_data = df_clean[col].fillna(df_clean[col].median())
                                if col_data.std() > 0:  # Avoid division by zero
                                    z_scores = np.abs(stats.zscore(col_data))
                                    extreme_outliers = z_scores > 4.5  # Very conservative - only extreme outliers
                                    outlier_scores += extreme_outliers.astype(int)
                                
                                # Method 2: IQR - only remove if beyond 4.5 * IQR (very conservative)
                                Q1 = df_clean[col].quantile(0.25)
                                Q3 = df_clean[col].quantile(0.75)
                                IQR = Q3 - Q1
                                if IQR > 0:  # Avoid division by zero
                                    lower_bound = Q1 - 4.5 * IQR  # Very conservative
                                    upper_bound = Q3 + 4.5 * IQR
                                    iqr_extreme = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).astype(int)
                                    outlier_scores += iqr_extreme
                            except:
                                continue
                    
                    # Only remove rows that are outliers in MULTIPLE critical metrics (at least 2)
                    # And optionally rows with MANY zeros across critical metrics
                    base_outlier_mask = (outlier_scores >= 2)
                    # Threshold: many zeros if at least half of the critical metrics are zero (min 3)
                    zero_threshold = max(3, int(np.ceil(len(critical_cols) * 0.5))) if len(critical_cols) > 0 else 3
                    many_zeros_mask = zero_counts >= zero_threshold
                    proposed_mask = base_outlier_mask | many_zeros_mask
                    
                    # Safety guard: if proposed removal would drop too many rows (>20%), fall back to base outliers only
                    removal_ratio = proposed_mask.mean() if len(df_clean) > 0 else 0
                    outlier_mask = base_outlier_mask if removal_ratio > 0.20 else proposed_mask
                    
                    # Remove only extreme outliers
                    rows_before = len(df_clean)
                    df_clean = df_clean[~outlier_mask].copy()
                    rows_removed = rows_before - len(df_clean)
                    
                    if rows_removed > 0:
                        st.session_state.df = df_clean  # Update main dataframe
                        st.session_state.df_clean = df_clean
                        
                        st.success(f"Data cleaned successfully! Removed {rows_removed} extreme outlier rows ({rows_removed/rows_before*100:.1f}% of data).")
                        
                        st.markdown("#### Before vs After")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Rows", rows_before)
                        with col2:
                            st.metric("Cleaned Rows", len(df_clean))
                        with col3:
                            st.metric("Rows Removed", rows_removed, delta=f"-{rows_removed/rows_before*100:.1f}%")
                    else:
                        st.info("No extreme outliers detected. Data is clean!")

elif page == "Model Training":
    st.markdown('<h1 class="main-header">Advanced Model Training</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        
        tab1, tab2 = st.tabs(["⚽ Player Load Prediction (Regression)", "🛡️ Injury Risk Classification"])
        
        with tab1:
            st.markdown("### Train Player Load Prediction Model")
            st.info("This model predicts Player Load based on performance metrics")
            
            if st.button("Train Regression Model", type="primary", key="train_reg"):
                with st.spinner("Training model... This may take a moment"):
                    model, metrics, features = train_regression_model_fast(df)
                    st.session_state.regression_model = model
                    
                    st.success("✅ Model trained successfully!")
                    
                    st.markdown("### 📊 Train vs Test Performance")
                    
                    col1, col2 = st.columns(2)
                    train_r2 = metrics['train']['R2']
                    test_r2 = metrics['test']['R2']
                    
                    with col1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2196F3;">
                            <h4 style="color: #2196F3; margin: 0 0 1rem 0;">🟢 Training Set</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #1E88E5;">{train_r2:.4f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">R² Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        # Visual progress for R²
                        st.progress(min(max(train_r2, 0.0), 1.0))
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #212529;">{metrics['train']['MAE']:.2f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">MAE</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #212529;">{metrics['train']['RMSE']:.2f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">RMSE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #43A047;">
                            <h4 style="color: #43A047; margin: 0 0 1rem 0;">🔵 Test Set</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 2rem; font-weight: 700; color: #43A047;">{test_r2:.4f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">R² Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        # Visual progress for R²
                        st.progress(min(max(test_r2, 0.0), 1.0))
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #212529;">{metrics['test']['MAE']:.2f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">MAE</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-card-enhanced" style="text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: #212529;">{metrics['test']['RMSE']:.2f}</div>
                            <div style="font-size: 0.85rem; color: #6C757D; margin-top: 0.25rem;">RMSE</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Check for overfitting
                    diff = abs(train_r2 - test_r2)
                    st.metric("Generalization gap (|Train-Test|)", f"{diff:.4f}", delta=f"{(test_r2-train_r2):+.4f}")
                    if diff > 0.15:
                        st.warning(f"⚠️ **Warning:** Potential overfitting! Train R² ({train_r2:.4f}) vs Test R² ({test_r2:.4f}) gap: {diff:.4f}")
                    elif diff > 0.10:
                        st.info(f"⚠️ **Moderate overfitting:** Gap: {diff:.4f}")
                    else:
                        st.success(f"✅ **No overfitting!** Gap: {diff:.4f}")
                    
                    # Balloons: celebrate either excellent test R² or very small gap with solid performance
                    if (metrics['test']['R2'] >= 0.90) or (diff <= 0.05 and metrics['test']['R2'] >= 0.80):
                        st.balloons()
                        st.markdown('<div class="success-box">🎯 <b>Great Performance!</b> Strong generalization and high R²</div>', unsafe_allow_html=True)
                    
                    # Visualize overfitting with Plotly
                    st.markdown("### 📊 Overfitting Visualization")
                    fig_overfitting = plot_overfitting_metrics(metrics)
                    if fig_overfitting:
                        st.plotly_chart(fig_overfitting, use_container_width=True)
                    
                    # Cross-validation metrics
                    if 'cv' in metrics and display_cv_metrics in globals():
                        st.markdown("---")
                        display_cv_metrics(metrics['cv'], model_type='regression')
                        
                        # Box plot for CV
                        cv_box = create_cv_box_plot(metrics['cv'], model_type='regression')
                        if cv_box:
                            st.plotly_chart(cv_box, use_container_width=True)
                    
                    # SHAP feature importance
                    if SHAP_AVAILABLE and 'generate_shap_analysis' in globals():
                        with st.expander("🔬 SHAP Analysis (Advanced Explainability)", expanded=False):
                            st.info("SHAP values explain individual predictions by showing feature contributions")
                            try:
                                shap_values, X_transformed = generate_shap_analysis(model, st.session_state.df[features].head(100), features)
                                if shap_values is not None:
                                    fig_shap = plot_shap_summary(shap_values, X_transformed, features)
                                    if fig_shap:
                                        st.plotly_chart(fig_shap, use_container_width=True)
                            except Exception as e:
                                st.warning(f"SHAP analysis failed: {e}")
                    
                    # Feature importance visualization
                    st.markdown("### 🔍 Feature Importance")
                    importance_df = plot_feature_importance(model, features)
                    if importance_df is not None and len(importance_df) > 0:
                        st.dataframe(importance_df, use_container_width=True)
                        
                        # Create bar chart
                        fig_importance = go.Figure()
                        fig_importance.add_trace(go.Bar(
                            x=importance_df['importance'],
                            y=importance_df['feature'],
                            orientation='h',
                            marker_color='#1E88E5'
                        ))
                        fig_importance.update_layout(
                            title='Top 15 Most Important Features',
                            xaxis_title='Importance',
                            yaxis_title='Feature',
                            height=500
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type.")
                    
                    # MLflow tracking status
                    if MLFLOW_AVAILABLE:
                        st.success("✅ Model logged to MLflow for versioning and tracking")
                    
                    st.markdown(f"**Features used:** {len(features)}")
                    with st.expander("View all features"):
                        st.write(features)
        
        with tab2:
            st.markdown("### Train Injury Risk Classification Model")
            st.info("This model classifies players into low/medium/high injury risk categories")
            
            st.markdown("""
            **Risk Classification Rules:**
            - **Low Risk:** Average predicted risk ≤ 0.40
            - **Medium Risk:** Average risk > 0.40 and maximum risk indicates medium
            - **High Risk:** Maximum risk indicates high
            """)
            
            if st.button("Train Classification Model", type="primary", key="train_class"):
                with st.spinner("Training model... This may take a moment"):
                    model, metrics, features = train_classification_model_fast(df)
                    st.session_state.classification_model = model
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Check for overfitting
                    train_acc = metrics['train']['Accuracy']
                    test_acc = metrics['test']['Accuracy']
                    
                    st.markdown("### 📊 Train vs Test Performance")
                    
                    # Create tabs for different metrics
                    metric_type = st.radio("Select Metric", ["Accuracy", "Precision", "Recall", "F1 Score"])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🟢 Training Set")
                        train_val = metrics['train'].get(metric_type, 0)
                        st.metric(metric_type, f"{train_val:.4f}")
                        st.metric("Accuracy", f"{metrics['train']['Accuracy']:.4f}")
                        st.metric("F1 Score", f"{metrics['train']['F1']:.4f}")
                    
                    with col2:
                        st.markdown("#### 🔵 Test Set")
                        test_val = metrics['test'].get(metric_type, 0)
                        st.metric(metric_type, f"{test_val:.4f}")
                        st.metric("Accuracy", f"{metrics['test']['Accuracy']:.4f}")
                        st.metric("F1 Score", f"{metrics['test']['F1']:.4f}")
                    
                    # Overfitting warning
                    acc_diff = abs(train_acc - test_acc)
                    if acc_diff > 0.15:
                        st.warning(f"⚠️ **Warning:** Potential overfitting! Train Acc ({train_acc:.4f}) vs Test Acc ({test_acc:.4f}) gap: {acc_diff:.4f}")
                    elif acc_diff > 0.10:
                        st.info(f"⚠️ **Moderate overfitting:** Gap: {acc_diff:.4f}")
                    else:
                        st.success(f"✅ **No overfitting!** Gap: {acc_diff:.4f}")
                    
                    if metrics['test']['Accuracy'] >= 0.90:
                        st.balloons()
                        st.markdown('<div class="success-box">🎯 <b>Excellent Performance!</b> Test Accuracy exceeds 0.90 threshold</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Features used:** {len(features)}")
                    with st.expander("View all features"):
                        st.write(features)

elif page == "Player Load Analysis":
    st.markdown('<h1 class="main-header">Player Load Analysis</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        
        if 'Player Name' in df.columns and 'Player Load' in df.columns:
            # Initialize excluded players in session state
            if 'excluded_players' not in st.session_state:
                st.session_state.excluded_players = []
            
            # Player exclusion panel
            with st.expander("⚙️ Player Management - Exclude Players from Analysis", expanded=False):
                st.markdown("#### 🗑️ Select players to exclude from analysis")
                st.info("💡 **Tip:** Exclude players who are no longer training, injured, or not part of current analysis")
                
                # Get list of all unique players
                all_players = sorted(df['Player Name'].unique())
                
                # Create two columns for better layout
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    # Multiselect for excluding players
                    players_to_exclude = st.multiselect(
                        "Select players to exclude:",
                        options=all_players,
                        default=st.session_state.excluded_players,
                        key="player_exclude_multiselect"
                    )
                    
                    # Update session state
                    if players_to_exclude != st.session_state.excluded_players:
                        st.session_state.excluded_players = players_to_exclude
                
                with col_right:
                    st.markdown("#### 📊 Current Status")
                    st.metric("Total Players", len(all_players))
                    st.metric("Excluded Players", len(st.session_state.excluded_players))
                    st.metric("Active Players", len(all_players) - len(st.session_state.excluded_players))
                
                # Show excluded players
                if st.session_state.excluded_players:
                    st.markdown("**Currently excluded players:**")
                    for player in st.session_state.excluded_players:
                        st.write(f"  • {player}")
                    
                    # Button to clear all exclusions
                    if st.button("🔄 Clear All Exclusions", type="secondary"):
                        st.session_state.excluded_players = []
                        st.rerun()
            
            # Filter dataframe to exclude selected players
            if st.session_state.excluded_players:
                df = df[~df['Player Name'].isin(st.session_state.excluded_players)]
                st.info(f"📋 Analysis showing {len(df['Player Name'].unique())} active players (excluding {len(st.session_state.excluded_players)} players)")
            
            st.markdown("---")
            # Separate training and matches
            if 'Session Title' in df.columns:
                df['Session Type'] = df['Session Title'].apply(
                    lambda x: 'Match' if 'match' in str(x).lower() or 'game' in str(x).lower() else 'Training'
                )
            else:
                df['Session Type'] = 'Training'
            
            st.markdown("### 🏃 Average Player Load by Player")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏋️ Training Sessions")
                training_data = df[df['Session Type'] == 'Training'].groupby('Player Name')['Player Load'].mean().sort_values(ascending=False)
                st.dataframe(training_data.head(10).reset_index().rename(columns={'Player Load': 'Avg Player Load'}), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                training_data.head(10).plot(kind='barh', ax=ax, color='#43A047')
                ax.set_xlabel('Average Player Load')
                ax.set_title('Top 10 Players - Training Load')
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### ⚽ Match Sessions")
                match_data = df[df['Session Type'] == 'Match'].groupby('Player Name')['Player Load'].mean().sort_values(ascending=False)
                
                if len(match_data) > 0:
                    st.dataframe(match_data.head(10).reset_index().rename(columns={'Player Load': 'Avg Player Load'}), use_container_width=True)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    match_data.head(10).plot(kind='barh', ax=ax, color='#1E88E5')
                    ax.set_xlabel('Average Player Load')
                    ax.set_title('Top 10 Players - Match Load')
                    st.pyplot(fig)
                else:
                    st.info("No match data available")
            
            st.markdown("### 💡 Professional Recommendations for Coach")
            
            top_training = training_data.head(3).index.tolist()
            top_match = match_data.head(3).index.tolist() if len(match_data) > 0 else []
            
            st.markdown(f"""
            <div class="success-box">
            <h4>🎯 High Performance Athletes (Training)</h4>
            <p><b>Top 3 Players:</b> {', '.join(top_training)}</p>
            <ul>
                <li>These players consistently show high work capacity during training</li>
                <li>Consider them as leaders for high-intensity drills</li>
                <li>Monitor closely for signs of overtraining despite high capacity</li>
                <li>Use their work rate as benchmarks for team standards</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if len(top_match) > 0:
                st.markdown(f"""
                <div class="success-box">
                <h4>⚽ Match Day Performers</h4>
                <p><b>Top 3 Players:</b> {', '.join(top_match)}</p>
                <ul>
                    <li>These players excel under competitive pressure</li>
                    <li>Prioritize their fitness for critical matches</li>
                    <li>Ensure adequate recovery between matches</li>
                    <li>Consider rotation if back-to-back games scheduled</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            bottom_training = training_data.tail(3).index.tolist()
            
            st.markdown(f"""
            <div class="warning-box">
            <h4>⚠️ Development Focus Areas</h4>
            <p><b>Bottom 3 Players:</b> {', '.join(bottom_training)}</p>
            <ul>
                <li>May benefit from individualized conditioning programs</li>
                <li>Investigate potential underlying fitness or motivation issues</li>
                <li>Consider gradual load progression protocols</li>
                <li>Schedule one-on-one performance reviews</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "Injury Prevention":
    st.markdown('<h1 class="main-header">Injury Risk Prevention System</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    elif st.session_state.classification_model is None:
        st.warning("⚠️ Please train the classification model first in Model Training section.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_classification(df)
        
        # Initialize excluded players in session state for injury prevention
        if 'excluded_players_injury' not in st.session_state:
            st.session_state.excluded_players_injury = []
        
        # Player exclusion panel
        with st.expander("⚙️ Player Management - Exclude Players from Analysis", expanded=False):
            st.markdown("#### 🗑️ Select players to exclude from risk analysis")
            st.info("💡 **Tip:** Exclude players who are no longer training, injured, or not part of current analysis")
            
            # Get list of all unique players
            all_players = sorted(df_clean['Player Name'].unique()) if 'Player Name' in df_clean.columns else []
            
            if len(all_players) > 0:
                # Create two columns for better layout
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    # Multiselect for excluding players
                    players_to_exclude = st.multiselect(
                        "Select players to exclude:",
                        options=all_players,
                        default=st.session_state.excluded_players_injury,
                        key="player_exclude_injury_multiselect"
                    )
                    
                    # Update session state
                    if players_to_exclude != st.session_state.excluded_players_injury:
                        st.session_state.excluded_players_injury = players_to_exclude
                
                with col_right:
                    st.markdown("#### 📊 Current Status")
                    st.metric("Total Players", len(all_players))
                    st.metric("Excluded Players", len(st.session_state.excluded_players_injury))
                    st.metric("Active Players", len(all_players) - len(st.session_state.excluded_players_injury))
                
                # Show excluded players
                if st.session_state.excluded_players_injury:
                    st.markdown("**Currently excluded players:**")
                    for player in st.session_state.excluded_players_injury:
                        st.write(f"  • {player}")
                    
                    # Button to clear all exclusions
                    if st.button("🔄 Clear All Exclusions", type="secondary"):
                        st.session_state.excluded_players_injury = []
                        st.rerun()
                
                # Filter out excluded players
                if st.session_state.excluded_players_injury:
                    df_clean = df_clean[~df_clean['Player Name'].isin(st.session_state.excluded_players_injury)]
                    st.info(f"📋 Analysis showing {len(df_clean['Player Name'].unique())} active players")
        
        # Get last 2 weeks of data
        if 'Date' in df_clean.columns:
            # Handle Excel date format (numbers like 45757)
            # First try to convert if it's numeric
            if df_clean['Date'].dtype in ['int64', 'float64']:
                try:
                    # Excel date format conversion
                    df_clean['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df_clean['Date'], unit='D')
                except:
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            else:
                # Try multiple date formats for text dates
                try:
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
                except:
                    # If first attempt fails, try specific format
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y%m%d', errors='coerce')
            
            # Remove rows where date parsing failed
            df_clean = df_clean.dropna(subset=['Date'])
            
            # Check if we have valid dates
            if len(df_clean) > 0 and not pd.isna(df_clean['Date'].max()):
                max_date = df_clean['Date'].max()
                two_weeks_ago = max_date - timedelta(days=14)
                recent_data = df_clean[df_clean['Date'] >= two_weeks_ago]
                
                # If no recent data but we have valid dates, use last 30% or min 50 rows
                if len(recent_data) == 0:
                    st.warning("⚠️ No data in last 2 weeks, using most recent 30% instead")
                    recent_data = df_clean.tail(max(50, int(len(df_clean) * 0.30)))
            else:
                # No valid dates, use last portion of data
                recent_data = df_clean.tail(int(len(df_clean) * 0.15) if len(df_clean) > 0 else 50)
        else:
            # No Date column, use last 30% of data
            recent_data = df_clean.tail(max(50, int(len(df_clean) * 0.30)))
        
        # Final check - if still no data, use all available data
        if len(recent_data) == 0:
            st.warning("⚠️ No recent data available, using all dataset")
            recent_data = df_clean.copy()
        
        st.info(f"📅 Analyzing {len(recent_data)} sessions (last 2 weeks)")
        
        # Calculate injury risk
        features = [
            'Work Ratio', 'Energy (kcal)', 'Distance (miles)', 'Sprint Distance (yards)',
            'Top Speed (mph)', 'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)',
            'Distance Per Min (yd/min)', 'Hr Load', 'Hr Max (bpm)', 'Time In Red Zone (min)',
            'Impacts', 'Impact Zones: > 20 G (Impacts)', 'Impact Zones: 15 - 20 G (Impacts)',
            'Power Plays', 'Power Score (w/kg)', 'Distance in Speed Zone 4 (miles)',
            'Distance in Speed Zone 5 (miles)', 'Time in HR Load Zone 85% - 96% Max HR (secs)'
        ]
        features = [f for f in features if f in recent_data.columns]
        
        # Check if we have enough features
        if len(features) == 0:
            st.error("❌ **No required features found in dataset!**")
            st.stop()
        
        X_recent = recent_data[features]
        
        # Make sure we have data to predict
        if len(X_recent) == 0:
            st.warning("⚠️ No data available for prediction. Please check your dataset.")
            st.stop()
        
        predictions = st.session_state.classification_model.predict(X_recent)
        
        recent_data['Predicted_Risk'] = predictions
        recent_data['Risk_Label'] = recent_data['Predicted_Risk'].map({0: 'Low', 1: 'Medium', 2: 'High'})
        
        if 'Player Name' in recent_data.columns:
            player_risk = recent_data.groupby('Player Name')['Predicted_Risk'].agg(['mean', 'max']).reset_index()
            # Risk classification: Low if avg_risk <= 0.40
            risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
            def classify_risk(row):
                mean_val = float(row['mean'])
                max_val = float(row['max'])
                # Low Risk: Average risk <= 0.40
                if mean_val <= 0.40:
                    return 'Low'
                # Otherwise, use maximum risk to classify
                elif max_val >= 2.0:  # High risk class
                    return 'High'
                elif max_val >= 1.0:  # Medium risk class
                    return 'Medium'
                else:
                    return 'Low'
            player_risk['Risk_Category'] = player_risk.apply(classify_risk, axis=1)
            player_risk = player_risk.sort_values('mean', ascending=False)
            
            st.markdown("### Current Injury Risk Status")
            
            col1, col2, col3 = st.columns(3)
            high_risk = len(player_risk[player_risk['Risk_Category'] == 'High'])
            medium_risk = len(player_risk[player_risk['Risk_Category'] == 'Medium'])
            low_risk = len(player_risk[player_risk['Risk_Category'] == 'Low'])
            
            with col1:
                st.markdown(f"""
                <div style='background-color:#F8D7DA; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:#DC3545; margin:0;'>{high_risk}</h2>
                    <p style='margin:0;'><b>High Risk</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color:#FFF3CD; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:#FFC107; margin:0;'>{medium_risk}</h2>
                    <p style='margin:0;'><b>Medium Risk</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='background-color:#D4EDDA; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:#28A745; margin:0;'>{low_risk}</h2>
                    <p style='margin:0;'><b>Low Risk</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📋 Player Risk Assessment")
            
            # Export to Excel button
            col_export, col_space = st.columns([1, 5])
            with col_export:
                if st.button("📥 Export to Excel", type="secondary"):
                    try:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"reports/injury_risk_{timestamp}.xlsx"
                        
                        # Create Excel writer
                        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                            # Main risk table
                            display_df = player_risk.copy()
                            display_df.columns = ['Player Name', 'Avg Risk Score', 'Max Risk Level', 'Risk Category']
                            display_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
                            
                            # Detailed predictions
                            if len(recent_data) > 0 and 'Predicted_Risk' in recent_data.columns:
                                detail_df = recent_data[['Player Name', 'Date', 'Predicted_Risk', 'Risk_Label']].copy()
                                detail_df.to_excel(writer, sheet_name='Detailed Sessions', index=False)
                            
                            # Summary sheet
                            summary_data = {
                                'Metric': ['Total Players', 'High Risk', 'Medium Risk', 'Low Risk'],
                                'Count': [len(player_risk), high_risk, medium_risk, low_risk]
                            }
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        st.success(f"✅ Exported to: {filename}")
                        st.info("📁 File saved in reports/ directory")
                        
                    except Exception as e:
                        st.error(f"Error exporting to Excel: {str(e)}")
            
            display_df = player_risk.copy()
            display_df.columns = ['Player Name', 'Avg Risk Score', 'Max Risk Level', 'Risk Category']
            st.dataframe(display_df, use_container_width=True)
            
            # High risk players
            high_risk_players = player_risk[player_risk['Risk_Category'] == 'High']
            
            if len(high_risk_players) > 0:
                st.markdown(f"""
                <div class="danger-box">
                <h4>🚨 URGENT: High Risk Players</h4>
                <p><b>Players requiring immediate attention:</b> {', '.join(high_risk_players['Player Name'].tolist())}</p>
                <h5>Immediate Actions Required:</h5>
                <ul>
                    <li><b>Reduce Training Load:</b> Decrease intensity by 30-40% for next 3-5 days</li>
                    <li><b>Medical Assessment:</b> Schedule immediate check-up with medical staff</li>
                    <li><b>Modified Training:</b> Focus on low-impact recovery sessions (swimming, yoga, mobility)</li>
                    <li><b>Sleep & Nutrition:</b> Verify adequate rest (8+ hours) and proper nutrition</li>
                    <li><b>Match Consideration:</b> Strongly consider rotation for upcoming matches</li>
                    <li><b>Daily Monitoring:</b> Track wellness questionnaires and morning heart rate variability</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            medium_risk_players = player_risk[player_risk['Risk_Category'] == 'Medium']
            
            if len(medium_risk_players) > 0:
                st.markdown(f"""
                <div class="warning-box">
                <h4>⚠️ CAUTION: Medium Risk Players</h4>
                <p><b>Players requiring monitoring:</b> {', '.join(medium_risk_players['Player Name'].tolist())}</p>
                <h5>Recommended Actions:</h5>
                <ul>
                    <li><b>Load Management:</b> Reduce intensity by 15-20% in next training sessions</li>
                    <li><b>Enhanced Recovery:</b> Add extra recovery day this week</li>
                    <li><b>Preventive Measures:</b> Increase stretching and mobility work</li>
                    <li><b>Monitoring:</b> Track fatigue levels daily through wellness surveys</li>
                    <li><b>Match Time:</b> Consider limiting minutes if playing consecutive matches</li>
                    <li><b>Physiotherapy:</b> Optional preventive sessions for muscle maintenance</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
            <h4>✅ General Injury Prevention Strategies</h4>
            <ul>
                <li><b>Periodization:</b> Implement proper training load periodization (hard days followed by easy days)</li>
                <li><b>Acute:Chronic Workload Ratio:</b> Maintain ratio between 0.8-1.3 to minimize injury risk</li>
                <li><b>Warm-up Protocol:</b> Ensure 15-20 minute dynamic warm-up before all sessions</li>
                <li><b>Cool-down:</b> 10-15 minutes of active recovery post-training</li>
                <li><b>Hydration:</b> Monitor hydration status, especially during high-intensity periods</li>
                <li><b>Sleep Monitoring:</b> Encourage 8+ hours of quality sleep for all players</li>
                <li><b>Communication:</b> Maintain open dialogue with players about their perceived fatigue</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # ==================== PERSONALIZED PLAYER RECOMMENDATIONS ====================
            st.markdown("---")
            st.markdown("### 🎯 Personalized Player Analysis & Recommendations")
            
            if 'Player Name' in recent_data.columns:
                # Select player for detailed analysis
                selected_player = st.selectbox(
                    "👤 Select a player for detailed analysis:",
                    options=['Choose a player...'] + sorted(recent_data['Player Name'].unique()),
                    key="injury_player_selector"
                )
                
                if selected_player != 'Choose a player...':
                    player_data = recent_data[recent_data['Player Name'] == selected_player]
                    
                    if len(player_data) > 0:
                        # Get player's risk level - use MAX risk to match table logic
                        if 'Predicted_Risk' in player_data.columns:
                            # Use max risk (worst case scenario) to match the table logic
                            max_risk = player_data['Predicted_Risk'].max()
                            player_risk_level = {0: 'Low', 1: 'Medium', 2: 'High'}.get(max_risk, 'Medium')
                        elif 'Risk_Label' in player_data.columns and len(player_data['Risk_Label']) > 0:
                            # If we have Risk_Label, take the maximum (worst) risk level
                            if player_data['Risk_Label'].eq('High').any():
                                player_risk_level = 'High'
                            elif player_data['Risk_Label'].eq('Medium').any():
                                player_risk_level = 'Medium'
                            else:
                                player_risk_level = 'Low'
                        else:
                            player_risk_level = 'Medium'
                        
                        # Generate intelligent recommendations
                        intel_recs = generate_intelligent_recommendations(selected_player, player_data, recent_data)
                        
                        # Calculate player metrics
                        avg_load = player_data['Player Load'].mean() if 'Player Load' in player_data.columns else 0
                        max_load = player_data['Player Load'].max() if 'Player Load' in player_data.columns else 0
                        avg_energy = player_data['Energy (kcal)'].mean() if 'Energy (kcal)' in player_data.columns else 0
                        avg_distance = player_data['Distance (miles)'].mean() if 'Distance (miles)' in player_data.columns else 0
                        max_speed = player_data['Top Speed (mph)'].max() if 'Top Speed (mph)' in player_data.columns else 0
                        total_impacts = player_data['Impacts'].sum() if 'Impacts' in player_data.columns else 0
                        avg_hr_load = player_data['Hr Load'].mean() if 'Hr Load' in player_data.columns else 0
                        
                        # Display player overview
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Average Player Load", f"{avg_load:.1f}")
                            st.metric("Max Player Load", f"{max_load:.1f}")
                        
                        with col2:
                            st.metric("Average Energy (kcal)", f"{avg_energy:.0f}")
                            st.metric("Total Impacts", f"{int(total_impacts)}")
                        
                        with col3:
                            st.metric("Max Speed (mph)", f"{max_speed:.1f}")
                            st.metric("Avg HR Load", f"{avg_hr_load:.1f}")
                        
                        # Display intelligent insights if available
                        if intel_recs.get('comparison_to_team'):
                            st.markdown("### 🔍 Smart Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'load_vs_team' in intel_recs['comparison_to_team']:
                                    st.info(f"📊 Load: {intel_recs['comparison_to_team']['load_vs_team']}")
                            with col2:
                                if 'load_percentile' in intel_recs['comparison_to_team']:
                                    st.info(f"📊 Percentile: Top {intel_recs['comparison_to_team']['load_percentile']}")
                        
                        # Display trends
                        if intel_recs.get('trends'):
                            for key, value in intel_recs['trends'].items():
                                if key == 'load_trend':
                                    if 'INCREASING' in value:
                                        st.warning(f"📈 {value}")
                                    elif 'DECREASING' in value:
                                        st.success(f"📉 {value}")
                                    else:
                                        st.info(f"➡️ {value}")
                        
                        # Display key insights
                        if intel_recs.get('key_insights'):
                            st.markdown("### 💡 Key Insights")
                            for insight in intel_recs['key_insights']:
                                st.markdown(f"- {insight}")
                        
                        # Risk-based recommendations
                        if player_risk_level == 'High':
                            st.markdown(f"""
                            <div class="danger-box">
                            <h4>🚨 {selected_player} - HIGH RISK PROTOCOL</h4>
                            <p><b>Current Risk Level:</b> HIGH - Immediate intervention required</p>
                            
                            <h5>📊 Key Metrics:</h5>
                            <ul>
                                <li>Average Load: <b>{avg_load:.1f}</b> (Consider reducing to <b>{avg_load * 0.6:.1f}</b>)</li>
                                <li>Maximum Load: <b>{max_load:.1f}</b> (Peak reached this period)</li>
                                <li>Total Impacts: <b>{int(total_impacts)}</b> (Monitor for cumulative fatigue)</li>
                            </ul>
                            
                            <h5>🎯 IMMEDIATE ACTIONS (Next 48-72 hours):</h5>
                            <ol>
                                <li><b>Load Reduction:</b> Cut training intensity by 40-50% immediately</li>
                                <li><b>Recovery Priority:</b> Implement 2 full recovery days (pool/cycling/yoga only)</li>
                                <li><b>Medical Review:</b> Schedule physiotherapy assessment within 24 hours</li>
                                <li><b>Match Status:</b> Recommend starting from bench or limiting to 30-45 minutes</li>
                                <li><b>Wellness Check:</b> Daily sleep quality (target 9+ hours) and hydration monitoring</li>
                            </ol>
                            
                            <h5>📅 Weekly Management:</h5>
                            <ul>
                                <li>No high-intensity drills until load decreases by 30%</li>
                                <li>Focus on technical/tactical work instead of physical load</li>
                                <li>Gradual return over 5-7 days with daily load monitoring</li>
                                <li>Consider GPS tracking during return-to-play protocol</li>
                            </ul>
                            
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display data-driven specific actions
                            if intel_recs.get('specific_actions'):
                                st.markdown("### 📋 Data-Driven Specific Actions")
                                for action in intel_recs['specific_actions']:
                                    st.markdown(f"- {action}")
                            
                            # Display personalized training plan
                            if intel_recs.get('personalized_plan'):
                                st.markdown("### 📈 Personalized Training Plan")
                                for item in intel_recs['personalized_plan']:
                                    st.markdown(f"- {item}")
                        
                        elif player_risk_level == 'Medium':
                            st.markdown(f"""
                            <div class="warning-box">
                            <h4>⚠️ {selected_player} - PREVENTIVE MANAGEMENT</h4>
                            <p><b>Current Risk Level:</b> MEDIUM - Proactive monitoring recommended</p>
                            
                            <h5>📊 Key Metrics:</h5>
                            <ul>
                                <li>Average Load: <b>{avg_load:.1f}</b> (Maintain or slightly reduce)</li>
                                <li>Energy Expenditure: <b>{avg_energy:.0f} kcal</b> (Normal range)</li>
                                <li>Average Distance: <b>{avg_distance:.2f} miles</b> per session</li>
                            </ul>
                            
                            <h5>🎯 PREVENTIVE ACTIONS:</h5>
                            <ol>
                                <li><b>Load Management:</b> Reduce intensity by 20% for next 2 sessions</li>
                                <li><b>Recovery:</b> Add 1 extra recovery day this week</li>
                                <li><b>Mobility Work:</b> Increase dynamic stretching and foam rolling (15 min daily)</li>
                                <li><b>Monitoring:</b> Track morning heart rate variability (target decrease <5 bpm)</li>
                                <li><b>Rotation:</b> Consider limiting consecutive match minutes</li>
                            </ol>
                            
                            <h5>📅 Training Modifications:</h5>
                            <ul>
                                <li>Replace 1 high-intensity session with technical/tactical work</li>
                                <li>Implement active recovery between hard training days</li>
                                <li>Monitor for early signs of fatigue or discomfort</li>
                                <li>Consider nutritional review (pre/post training fueling)</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display data-driven actions for medium risk
                            if intel_recs.get('specific_actions'):
                                st.markdown("### 📋 Data-Driven Actions")
                                for action in intel_recs['specific_actions']:
                                    st.markdown(f"- {action}")
                            
                            if intel_recs.get('personalized_plan'):
                                st.markdown("### 📈 Personalized Plan")
                                for item in intel_recs['personalized_plan']:
                                    st.markdown(f"- {item}")
                        
                        else:  # Low risk
                            st.markdown(f"""
                            <div class="success-box">
                            <h4>✅ {selected_player} - OPTIMAL STATUS</h4>
                            <p><b>Current Risk Level:</b> LOW - Player in ideal condition</p>
                            
                            <h5>📊 Key Metrics:</h5>
                            <ul>
                                <li>Average Load: <b>{avg_load:.1f}</b> (Optimal training zone)</li>
                                <li>Energy Expenditure: <b>{avg_energy:.0f} kcal</b> (Efficient energy use)</li>
                                <li>Max Speed: <b>{max_speed:.1f} mph</b> ({'Good' if max_speed > 15 else 'Can improve'})</li>
                            </ul>
                            
                            <h5>🎯 MAINTAIN CURRENT PROTOCOL:</h5>
                            <ol>
                                <li><b>Training Load:</b> Continue current intensity and volume</li>
                                <li><b>Recovery:</b> Maintain standard recovery protocols (8+ hrs sleep)</li>
                                <li><b>Progression:</b> Can gradually increase load by 5-10% weekly if desired</li>
                                <li><b>Performance:</b> Ready for high-intensity training and matches</li>
                                <li><b>Monitoring:</b> Weekly wellness checks sufficient</li>
                            </ol>
                            
                            <h5>📈 Development Opportunities:</h5>
                            <ul>
                                <li>Consider adding explosive/sprint work if max speed < 18 mph</li>
                                <li>Can experiment with new training stimuli</li>
                                <li>Optimal candidate for leadership/mentoring roles</li>
                                <li>Use as benchmark for team standards</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display intelligent recommendations for low risk players
                            if intel_recs.get('specific_actions'):
                                st.markdown("### 📋 Data-Driven Actions")
                                for action in intel_recs['specific_actions']:
                                    st.markdown(f"- {action}")
                            
                            if intel_recs.get('personalized_plan'):
                                st.markdown("### 📈 Personalized Development Plan")
                                for item in intel_recs['personalized_plan']:
                                    st.markdown(f"- {item}")

elif page == "Team Lineup Calculator":
    st.markdown('<h1 class="main-header">Optimal Team Lineup Calculator</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_regression(df)
        
        # Initialize excluded players for lineup
        if 'excluded_players_lineup' not in st.session_state:
            st.session_state.excluded_players_lineup = []
        
        st.markdown("### ⚙️ Configure Lineup Selection Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Selection Method")
            selection_method = st.selectbox(
                "Choose optimization method",
                ["Balanced Performance", "Maximum Energy Output", "Speed-Focused", "Endurance-Focused", "Custom Weights"]
            )
        
        with col2:
            st.markdown("#### 🔢 Team Size")
            team_size = st.slider("Number of players to select", min_value=5, max_value=20, value=11)
        
        # Player exclusion section
        with st.expander("🗑️ Exclude Players from Lineup", expanded=False):
            st.markdown("#### Select players to exclude from optimal lineup calculation")
        
        if 'Player Name' in df_clean.columns:
                all_players_lineup = sorted(df_clean['Player Name'].unique())
                
                col_exclude1, col_exclude2 = st.columns([2, 1])
                
                with col_exclude1:
                    players_to_exclude_lineup = st.multiselect(
                        "Select players to exclude:",
                        options=all_players_lineup,
                        default=st.session_state.excluded_players_lineup,
                        key="player_exclude_lineup_multiselect"
                    )
                    
                    # Update session state
                    if players_to_exclude_lineup != st.session_state.excluded_players_lineup:
                        st.session_state.excluded_players_lineup = players_to_exclude_lineup
                
                with col_exclude2:
                    st.markdown("#### 📊 Current Status")
                    st.metric("Total Players", len(all_players_lineup))
                    st.metric("Excluded", len(st.session_state.excluded_players_lineup))
                    st.metric("Available", len(all_players_lineup) - len(st.session_state.excluded_players_lineup))
                
                if st.session_state.excluded_players_lineup:
                    st.markdown("**Excluded players:**")
                    for player in st.session_state.excluded_players_lineup:
                        st.write(f"  • {player}")
                    
                    if st.button("🔄 Clear All Exclusions", type="secondary", key="clear_lineup_exclusions"):
                        st.session_state.excluded_players_lineup = []
                        st.rerun()
        
        st.markdown("---")
        
        if 'Player Name' in df_clean.columns:
            # Filter out excluded players
            df_for_lineup = df_clean.copy()
            if st.session_state.excluded_players_lineup:
                df_for_lineup = df_for_lineup[~df_for_lineup['Player Name'].isin(st.session_state.excluded_players_lineup)]
            
            # Calculate player metrics (only use columns that exist)
            metrics_to_aggregate = {
                'Player Load': 'mean',
                'Energy (kcal)': 'mean',
                'Top Speed (mph)': 'max',
                'Distance (miles)': 'mean',
                'Sprint Distance (yards)': 'mean',
                'Hr Load': 'mean',
                'Power Score (w/kg)': 'mean',
                'Impacts': 'mean'
            }
            
            # Filter to only include columns that exist in the data
            available_metrics = {k: v for k, v in metrics_to_aggregate.items() if k in df_for_lineup.columns}
            
            player_metrics = df_for_lineup.groupby('Player Name').agg(available_metrics).reset_index()
            
            # Normalize metrics (0-100 scale) with division by zero protection
            for col in player_metrics.columns[1:]:
                col_min = player_metrics[col].min()
                col_max = player_metrics[col].max()
                col_range = col_max - col_min
                
                # Handle division by zero - if all values are the same, set normalized to 50 (middle)
                if col_range == 0:
                    player_metrics[f'{col}_norm'] = 50.0
                else:
                    player_metrics[f'{col}_norm'] = ((player_metrics[col] - col_min) / col_range) * 100
            
            # Calculate composite score based on selection method
            if selection_method == "Balanced Performance":
                weights = {
                    'Player Load_norm': 0.25,
                    'Energy (kcal)_norm': 0.20,
                    'Top Speed (mph)_norm': 0.15,
                    'Distance (miles)_norm': 0.15,
                    'Sprint Distance (yards)_norm': 0.10,
                    'Power Score (w/kg)_norm': 0.15
                }
            elif selection_method == "Maximum Energy Output":
                weights = {
                    'Energy (kcal)_norm': 0.40,
                    'Player Load_norm': 0.30,
                    'Hr Load_norm': 0.20,
                    'Power Score (w/kg)_norm': 0.10
                }
            elif selection_method == "Speed-Focused":
                weights = {
                    'Top Speed (mph)_norm': 0.40,
                    'Sprint Distance (yards)_norm': 0.30,
                    'Distance (miles)_norm': 0.15,
                    'Power Score (w/kg)_norm': 0.15
                }
            elif selection_method == "Endurance-Focused":
                weights = {
                    'Distance (miles)_norm': 0.35,
                    'Player Load_norm': 0.25,
                    'Energy (kcal)_norm': 0.25,
                    'Hr Load_norm': 0.15
                }
            else:  # Custom Weights
                st.markdown("#### 🎛️ Custom Weight Configuration")
                col1, col2, col3 = st.columns(3)
                with col1:
                    w_load = st.slider("Player Load", 0.0, 1.0, 0.25, 0.05)
                    w_energy = st.slider("Energy", 0.0, 1.0, 0.20, 0.05)
                with col2:
                    w_speed = st.slider("Top Speed", 0.0, 1.0, 0.15, 0.05)
                    w_distance = st.slider("Distance", 0.0, 1.0, 0.15, 0.05)
                with col3:
                    w_sprint = st.slider("Sprint Distance", 0.0, 1.0, 0.10, 0.05)
                    w_power = st.slider("Power Score", 0.0, 1.0, 0.15, 0.05)
                
                total_weight = w_load + w_energy + w_speed + w_distance + w_sprint + w_power
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Normalizing to 1.0...")
                    w_load, w_energy, w_speed, w_distance, w_sprint, w_power = [w/total_weight for w in [w_load, w_energy, w_speed, w_distance, w_sprint, w_power]]
                
                weights = {
                    'Player Load_norm': w_load,
                    'Energy (kcal)_norm': w_energy,
                    'Top Speed (mph)_norm': w_speed,
                    'Distance (miles)_norm': w_distance,
                    'Sprint Distance (yards)_norm': w_sprint,
                    'Power Score (w/kg)_norm': w_power
                }
            
            # Calculate composite score (only use available columns)
            player_metrics['Composite_Score'] = 0
            available_weights = {}
            total_weight = 0
            
            for metric, weight in weights.items():
                if metric in player_metrics.columns:
                    # Replace NaN values with 0 for calculation
                    if player_metrics[metric].isna().any():
                        player_metrics[metric] = player_metrics[metric].fillna(0)
                    player_metrics['Composite_Score'] += player_metrics[metric] * weight
                    available_weights[metric] = weight
                    total_weight += weight
            
            # Normalize the composite score if weights don't sum to 1.0
            if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                player_metrics['Composite_Score'] = player_metrics['Composite_Score'] / total_weight
            
            # Handle NaN composite scores (shouldn't happen now, but just in case)
            player_metrics['Composite_Score'] = player_metrics['Composite_Score'].fillna(0)
            
            # Sort and select top players
            optimal_lineup = player_metrics.nlargest(team_size, 'Composite_Score')
            
            st.markdown("### 🏆 Optimal Lineup")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Only include columns that exist in the data
                display_cols = ['Player Name', 'Composite_Score']
                col_mapping = {
                    'Player Load': 'Avg Load',
                    'Energy (kcal)': 'Avg Energy',
                    'Top Speed (mph)': 'Max Speed',
                    'Distance (miles)': 'Avg Distance'
                }
                
                for orig_col, display_name in col_mapping.items():
                    if orig_col in optimal_lineup.columns:
                        display_cols.append(orig_col)
                
                display_lineup = optimal_lineup[display_cols].copy()
                
                # Rename columns to display names
                new_col_names = ['Player', 'Score']
                for orig_col in display_cols[2:]:
                    new_col_names.append(col_mapping.get(orig_col, orig_col))
                
                display_lineup.columns = new_col_names
                display_lineup['Rank'] = range(1, len(display_lineup) + 1)
                st.dataframe(display_lineup, use_container_width=True)
            
            with col2:
                st.markdown("#### 📊 Team Statistics")
                st.metric("Avg Composite Score", f"{optimal_lineup['Composite_Score'].mean():.1f}")
                
                # Only display metrics if they exist in the data
                if 'Player Load' in optimal_lineup.columns:
                    st.metric("Avg Player Load", f"{optimal_lineup['Player Load'].mean():.1f}")
                if 'Energy (kcal)' in optimal_lineup.columns:
                    st.metric("Avg Energy (kcal)", f"{optimal_lineup['Energy (kcal)'].mean():.1f}")
                if 'Top Speed (mph)' in optimal_lineup.columns:
                    st.metric("Max Speed (mph)", f"{optimal_lineup['Top Speed (mph)'].max():.1f}")
            
            # Visualization
            st.markdown("### 📊 Lineup Composition Analysis")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Score distribution
            axes[0].barh(optimal_lineup['Player Name'], optimal_lineup['Composite_Score'], color='#1E88E5')
            axes[0].set_xlabel('Composite Score')
            axes[0].set_title('Player Performance Scores', fontweight='bold')
            axes[0].invert_yaxis()
            
            # Radar chart for top 3 players
            from math import pi
            top_3 = optimal_lineup.head(3)
            
            # Build categories and values dynamically based on what exists
            category_info = [
                ('Load', 'Player Load_norm'),
                ('Energy', 'Energy (kcal)_norm'),
                ('Speed', 'Top Speed (mph)_norm'),
                ('Distance', 'Distance (miles)_norm'),
                ('Sprint', 'Sprint Distance (yards)_norm'),
                ('Power', 'Power Score (w/kg)_norm')
            ]
            
            # Filter to only include categories where data exists
            valid_categories = []
            for cat_label, col_name in category_info:
                if col_name in optimal_lineup.columns:
                    valid_categories.append((cat_label, col_name))
            
            if len(valid_categories) > 2:  # Need at least 3 categories for radar chart
                angles = [n / float(len(valid_categories)) * 2 * pi for n in range(len(valid_categories))]
                angles += angles[:1]
                
                ax = plt.subplot(122, polar=True)
                colors = ['#1E88E5', '#43A047', '#FFC107']
                
                for idx, (_, player) in enumerate(top_3.iterrows()):
                    values = [player[col_name] for _, col_name in valid_categories]
                    values += values[:1]
                    ax.plot(angles, values, 'o-', linewidth=2, label=player['Player Name'], color=colors[idx])
                    ax.fill(angles, values, alpha=0.15, color=colors[idx])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([cat[0] for cat in valid_categories])
                ax.set_ylim(0, 100)
                ax.set_title('Top 3 Players - Performance Profile', fontweight='bold', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                ax.grid(True)
            else:
                # Not enough data for radar chart, hide this subplot
                axes[1].axis('off')
                axes[1].text(0.5, 0.5, 'Not enough data\nfor radar chart', 
                           ha='center', va='center', transform=axes[1].transAxes, 
                           fontsize=12, style='italic')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate detailed team statistics
            team_composite_avg = optimal_lineup['Composite_Score'].mean()
            team_composite_std = optimal_lineup['Composite_Score'].std()
            
            # Calculate team capability ranges
            if 'Player Load' in optimal_lineup.columns:
                avg_load_range = f"{optimal_lineup['Player Load'].min():.0f}-{optimal_lineup['Player Load'].max():.0f}"
                avg_load_mean = optimal_lineup['Player Load'].mean()
            if 'Energy (kcal)' in optimal_lineup.columns:
                avg_energy_range = f"{int(optimal_lineup['Energy (kcal)'].min())}-{int(optimal_lineup['Energy (kcal)'].max())}"
                avg_energy_mean = optimal_lineup['Energy (kcal)'].mean()
            if 'Top Speed (mph)' in optimal_lineup.columns:
                speed_range = f"{optimal_lineup['Top Speed (mph)'].min():.1f}-{optimal_lineup['Top Speed (mph)'].max():.1f}"
                speed_mean = optimal_lineup['Top Speed (mph)'].mean()
            
            # Get top and bottom performers
            top_player = optimal_lineup.iloc[0]['Player Name']
            top_score = optimal_lineup.iloc[0]['Composite_Score']
            bottom_player = optimal_lineup.iloc[-1]['Player Name']
            bottom_score = optimal_lineup.iloc[-1]['Composite_Score']
            
            # Performance variance
            cv = team_composite_std / team_composite_avg if team_composite_avg > 0 else 0
            consistency_status = "Excellent" if cv < 0.1 else "Good" if cv < 0.2 else "Moderate" if cv < 0.3 else "Variable"
            
            # Determine consistency color
            if cv < 0.15:
                consistency_color = "#28A745"
            elif cv < 0.25:
                consistency_color = "#FFC107"
            else:
                consistency_color = "#FF9800"
            
            # Build top player metrics string
            top_metrics = ""
            if 'Player Load' in optimal_lineup.columns:
                top_metrics += f" | Load: {optimal_lineup.iloc[0]['Player Load']:.1f}"
            if 'Energy (kcal)' in optimal_lineup.columns:
                top_metrics += f" | Energy: {int(optimal_lineup.iloc[0]['Energy (kcal)'])} kcal"
            
            st.markdown(f"""
            <div class="success-box">
            <h4>💡 Detailed Lineup Analysis & Recommendations</h4>
            
            <h5>📊 Strategy Overview</h5>
            <p><b>Optimization Method:</b> <span style='color:#1E88E5'>{selection_method}</span></p>
            <p><b>Team Size:</b> {team_size} players selected</p>
            <p><b>Composite Score:</b> {team_composite_avg:.1f} avg (±{team_composite_std:.1f} standard deviation)</p>
            <p><b>Performance Consistency:</b> <span style='color:{consistency_color}'>{consistency_status}</span> (CV: {cv*100:.1f}%)</p>
            
            <h5>🏆 Top Performers</h5>
            <ul>
                <li><b>#1 Player:</b> {top_player} - Score: {top_score:.1f}{top_metrics}</li>
                <li><b>Performance Gap:</b> {top_score - bottom_score:.1f} points difference between #1 and #{team_size}</li>
                <li><b>Top 3 Average:</b> {optimal_lineup.head(3)['Composite_Score'].mean():.1f} vs Bottom 3: {optimal_lineup.tail(3)['Composite_Score'].mean():.1f}</li>
            </ul>
            
            <h5>⚙️ Team Capabilities</h5>
            <ul>""", unsafe_allow_html=True)
            
            # Add team capabilities dynamically
            if 'Player Load' in optimal_lineup.columns:
                st.markdown(f"<li><b>Load Range:</b> {avg_load_range} (avg: {avg_load_mean:.1f})</li>", unsafe_allow_html=True)
            if 'Energy (kcal)' in optimal_lineup.columns:
                st.markdown(f"<li><b>Energy Range:</b> {avg_energy_range} kcal (avg: {avg_energy_mean:.0f})</li>", unsafe_allow_html=True)
            if 'Top Speed (mph)' in optimal_lineup.columns:
                st.markdown(f"<li><b>Speed Range:</b> {speed_range} mph (avg: {speed_mean:.1f})</li>", unsafe_allow_html=True)
            
            # Determine balance check message
            balance_msg = "Watch for consistency variance - consider mixing high/low performers" if cv > 0.25 else "Well-balanced team composition"
            
            st.markdown(f"""
            
            <h5>🎯 Tactical Recommendations</h5>
            <ul>
                <li><b>Starting XI Impact:</b> Top-ranked players should start for maximum impact</li>
                <li><b>Substitution Strategy:</b> Players ranked 12-15 provide tactical flexibility (consider after 60-70 min)</li>
                <li><b>Rotation Plan:</b> Monitor fatigue: Alternate use of players with load >{avg_load_mean * 1.1:.0f} if available</li>
                <li><b>Balance Check:</b> {balance_msg}</li>
            </ul>
            
            <h5>⚠️ Critical Considerations</h5>
            <ul>
                <li><b>Recent Load:</b> Check injury prevention page for current load status of selected players</li>
                <li><b>Fatigue Management:</b> High-scoring players may be fatigued - verify current status</li>
                <li><b>Match Context:</b> Consider if {selection_method} strategy fits the opposition</li>
                <li><b>Player Availability:</b> Confirm all {team_size} players are match-ready</li>
            </ul>
            
            <h5>📈 Performance Projection</h5>
            """, unsafe_allow_html=True)
            
            # Determine performance message
            performance_msg = "High consistency suggests reliable output." if cv < 0.2 else "Moderate consistency - monitor individual variations."
            st.markdown(f"<p>Based on historical data, this lineup is expected to deliver {team_composite_avg:.0f} composite performance. {performance_msg}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Performance Analytics":
    st.markdown('<h1 class="main-header">Advanced Performance Analytics</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_regression(df)
        
        # AI Coach Assistant - Interactive Chat with Ollama
        # Always show assistant UI - it will handle connection status internally
        with st.expander("AI Coach Assistant - Understand Your Analytics", expanded=True):
                st.markdown("### Ask Your AI Assistant About Performance Analytics")
                
                # Player selection for context with search
                col_player, col_info = st.columns([2, 3])
                
                with col_player:
                    if 'Player Name' in df_clean.columns:
                        # Always default to "All Players" to have team data available
                        if st.session_state.selected_chat_player is None:
                            st.session_state.selected_chat_player = None  # Force to All Players
                        
                        chat_player = st.selectbox(
                            "Select a Player (Optional)",
                            ["All Players"] + sorted(list(df_clean['Player Name'].unique())),
                            key="chat_player_selector",
                            index=0,  # Always default to "All Players"
                            help="Select a specific player to get personalized insights. 'All Players' provides team-wide context."
                        )
                        
                        if chat_player != "All Players":
                            st.session_state.selected_chat_player = chat_player
                        else:
                            st.session_state.selected_chat_player = None
                    else:
                        st.session_state.selected_chat_player = None
                
                with col_info:
                    # Check if Ollama is available and working (LOCAL ONLY)
                    ollama_ready = False
                    base_url = get_ollama_base_url()
                    is_local = base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost")
                    
                    if not is_local:
                        # Cloud deployment - AI Assistant not supported
                        st.info("""
                        **🤖 AI Assistant (Local Only Feature)**
                        
                        The AI Coach Assistant is designed to work **only when running locally** on your machine.
                        
                        **To use the AI Assistant:**
                        1. Clone the repository: `git clone <repo-url>`
                        2. Install dependencies: `pip install -r requirements.txt`
                        3. Install Ollama: `pip install ollama`
                        4. Start Ollama: `ollama serve` (in a separate terminal)
                        5. Run the app: `streamlit run app.py`
                        6. Download model: `ollama pull llama3.2`
                        
                        **💡 The rest of the analytics work perfectly in the cloud without the AI Assistant!**
                        
                        All performance analytics, visualizations, and ML models are fully functional.
                        """)
                    elif not OLLAMA_AVAILABLE:
                        st.info("""
                        **AI Assistant (Optional Feature)**
                        
                        🤖 To enable the AI Coach Assistant locally:
                        1. Install: `pip install ollama`
                        2. Start server: `ollama serve`
                        3. Download model: `ollama pull llama3.2`
                        4. Refresh this page
                        
                        **💡 The analytics work perfectly without it!**
                        """)
                    else:
                        # Check if server is reachable (local only)
                        try:
                            ollama_ready = is_ollama_reachable()
                        except:
                            ollama_ready = False
                        
                        if not ollama_ready:
                            st.info(f"""
                            **AI Assistant (Local Setup Needed)**
                            
                            ⚠️ Cannot connect to Ollama at `{base_url}`
                            
                            **Setup steps:**
                            1. Start Ollama: Open terminal and run `ollama serve`
                            2. Download model: In another terminal, run `ollama pull llama3.2`
                            3. Refresh this page
                            
                            **💡 The analytics work perfectly without the AI Assistant!**
                            """)
                    
                    if ollama_ready and is_local:
                        # Status display with model check
                        model_status_info = check_model_status("llama3.2")
                        status_color = "#28a745" if model_status_info["available"] else "#ffc107" if model_status_info["status"] == "missing" else "#dc3545"
                        status_icon = "✅" if model_status_info["available"] else "⚠️" if model_status_info["status"] == "missing" else "❌"
                        
                        st.markdown(f"""
                        <div style="font-size: 0.75rem; color: {status_color}; padding: 0.5rem; background: #F8F9FA; border-radius: 4px; margin-bottom: 0.5rem;">
                            {status_icon} {model_status_info["message"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Buttons row
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            if st.button("🔄 Check Model Status", key="check_model_status_btn", help="Check if llama3.2 model is installed and ready", use_container_width=True):
                                model_status_info = check_model_status("llama3.2")
                                
                                if model_status_info["available"]:
                                    st.success(model_status_info["message"])
                                    if "size_mb" in model_status_info and model_status_info["size_mb"] > 0:
                                        st.info(f"Model size: {model_status_info['size_mb']:.1f} MB")
                                elif model_status_info["status"] == "missing":
                                    st.warning(model_status_info["message"])
                                    if model_status_info.get("download_available"):
                                        st.info("💡 The model will be downloaded automatically when you make your first request, or you can check the Render logs for download progress.")
                                        st.markdown("""
                                        <div style="font-size: 0.7rem; color: #6C757D; margin-top: 0.5rem;">
                                            📊 <strong>To check download progress:</strong><br>
                                            1. Go to <a href="https://dashboard.render.com" target="_blank">Render Dashboard</a><br>
                                            2. Open your <code>ollama-server</code> service<br>
                                            3. Click on "Logs" tab to see download progress<br>
                                            4. Look for messages like "Downloading llama3.2 model..."
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.error(model_status_info["message"])
                        
                        with col_btn2:
                            # Download button - only show if model is not available
                            if not model_status_info["available"]:
                                if st.button("⬇️ Download Model", key="download_model_btn", help="Download llama3.2 model with progress bar", type="primary", use_container_width=True):
                                    # Create progress container
                                    progress_container = st.container()
                                    
                                    # Download with progress
                                    success, message = download_model_with_progress("llama3.2", progress_container)
                                    
                                    if success:
                                        st.success(message)
                                        st.balloons()  # Celebrate!
                                        # Refresh page after a moment to show updated status
                                        st.rerun()
                                    else:
                                        st.warning(message)
                                        st.info("💡 The download may still be in progress in the background. Use 'Check Model Status' to verify.")
                            else:
                                st.info("✅ Model is ready!", help="Model is already installed")
                
                # Chat interface
                st.markdown("---")
                st.markdown("#### Conversation")
                
                # Initialize chat history if needed
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Display chat history with enhanced bubbles
                chat_container = st.container()
                with chat_container:
                    if len(st.session_state.chat_history) == 0:
                        st.markdown("""
                        <div class="info-box" style="text-align: center; padding: 2rem;">
                            <h4 style="color: #2196F3; margin-bottom: 1rem;">Hi! I'm your sports analytics assistant</h4>
                            <p style="color: #6C757D; margin-bottom: 1.5rem;">You can ask me about:</p>
                            <div style="text-align: left; max-width: 500px; margin: 0 auto;">
                                <p style="margin: 0.5rem 0;"><b>Specific player performance</b> - Analyze individual player metrics and trends</p>
                                <p style="margin: 0.5rem 0;"><b>Metric explanations</b> - Learn about Player Load, Energy, Speed Zones, etc.</p>
                                <p style="margin: 0.5rem 0;"><b>Player comparisons</b> - Compare performance across different players</p>
                                <p style="margin: 0.5rem 0;"><b>Trends and data analysis</b> - Understand patterns and performance trajectories</p>
                                <p style="margin: 0.5rem 0;"><b>Training recommendations</b> - Get personalized coaching insights</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        for i, msg in enumerate(st.session_state.chat_history):
                            if msg['role'] == 'user':
                                st.markdown(f"""
                                <div class="chat-bubble-user">
                                    <strong>You:</strong><br>{msg['content']}
                                </div>
                                """, unsafe_allow_html=True)
                            elif msg['role'] == 'assistant':
                                st.markdown(f"""
                                <div class="chat-bubble-assistant">
                                    <strong>Assistant:</strong><br>{msg['content']}
                                </div>
                                """, unsafe_allow_html=True)
                
                # Chat input - only enable if Ollama is ready and local
                if ollama_ready and is_local:
                    user_input = st.chat_input("Ask your question here...")
                else:
                    if not is_local:
                        st.chat_input("AI Assistant available only when running locally", disabled=True)
                    else:
                        st.chat_input("AI Assistant not available - Start Ollama: 'ollama serve'", disabled=True)
                    user_input = None
                
                if user_input and ollama_ready and is_local:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Always get team context first (All Players), then add specific player if selected
                    player_context = format_player_data_for_context(df_clean, None)  # Team data
                    
                    # Auto-detect player names mentioned in the user input
                    detected_players = []
                    if 'Player Name' in df_clean.columns:
                        available_players = sorted(list(df_clean['Player Name'].unique()))
                        detected_players = extract_player_names_from_text(user_input, available_players)
                    
                    # Build context with detected players and selected player
                    players_to_include = []
                    if st.session_state.selected_chat_player:
                        matched_name = find_player_in_dataframe(df_clean, st.session_state.selected_chat_player)
                        if matched_name:
                            players_to_include.append(matched_name)
                    
                    # Add detected players from text
                    for detected_player in detected_players:
                        matched_name = find_player_in_dataframe(df_clean, detected_player)
                        if matched_name and matched_name not in players_to_include:
                            players_to_include.append(matched_name)
                    
                    # Add player contexts
                    if players_to_include:
                        player_contexts_parts = []
                        for player_name in players_to_include[:5]:  # Limit to 5 players to avoid too much context
                            specific_player_context = format_player_data_for_context(df_clean, player_name)
                            if specific_player_context and "NOT FOUND" not in specific_player_context:
                                player_contexts_parts.append(f"\n\n---PLAYER: {player_name}---\n{specific_player_context[:800]}")
                        
                        if player_contexts_parts:
                            player_context = f"{player_context}\n\n{' '.join(player_contexts_parts)}"
                    elif st.session_state.selected_chat_player:
                        # Fallback to selected player if no auto-detection worked
                        specific_player_context = format_player_data_for_context(df_clean, st.session_state.selected_chat_player)
                        if specific_player_context:
                            # Check if player was not found
                            if "NOT FOUND" in specific_player_context:
                                player_context = specific_player_context  # Use the "not found" message
                            else:
                                player_context = f"{player_context}\n\n---SPECIFIC PLAYER---\n{specific_player_context[:500]}"
                        else:
                            # If no context returned, player doesn't exist
                            available_players = sorted(list(df_clean['Player Name'].unique())) if 'Player Name' in df_clean.columns else []
                            player_context = f"REQUESTED_PLAYER: {st.session_state.selected_chat_player} NOT FOUND in dataset.\nAVAILABLE_PLAYERS: {', '.join(available_players[:20])}{' (and more)' if len(available_players) > 20 else ''}"
                    
                    # Show loading
                    with st.spinner("🤔 Thinking..."):
                        # Get response from Ollama
                        response = get_ollama_response(
                            user_input, 
                            player_context=player_context,
                            chat_history=st.session_state.chat_history[:-1]  # Exclude the just-added user message
                        )
                        
                        # Add assistant response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Rerun to show new messages
                        st.rerun()
                
                # Chat actions with suggestions - only if Ollama is ready and local
                if ollama_ready and is_local and len(st.session_state.chat_history) == 0:
                    st.markdown("**Quick Questions:**")
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        if st.button("What is Player Load?", key="q1"):
                            user_input = "What is Player Load?"
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            player_context = format_player_data_for_context(df_clean, None)
                            if st.session_state.selected_chat_player:
                                specific_player_context = format_player_data_for_context(df_clean, st.session_state.selected_chat_player)
                                if specific_player_context:
                                    player_context = f"{player_context}\n\n---SPECIFIC PLAYER---\n{specific_player_context[:500]}"
                                with st.spinner("Thinking..."):
                                    response = get_ollama_response(user_input, player_context=player_context, chat_history=[])
                                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                                st.rerun()
                    with col_q2:
                        if st.button("Compare top players", key="q2"):
                            user_input = "Compare the top performing players"
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            player_context = format_player_data_for_context(df_clean, None)
                            with st.spinner("Thinking..."):
                                response = get_ollama_response(user_input, player_context=player_context, chat_history=[])
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                    with col_q3:
                        if st.button("Training recommendations", key="q3"):
                            user_input = "What are some training recommendations based on the data?"
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            player_context = format_player_data_for_context(df_clean, None)
                            with st.spinner("Thinking..."):
                                response = get_ollama_response(user_input, player_context=player_context, chat_history=[])
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            st.rerun()
                
                if len(st.session_state.chat_history) > 0:
                    col_clear, col_export, col_spacer = st.columns([1, 1, 3])
                    with col_clear:
                        if st.button("Clear", type="secondary", use_container_width=True):
                            st.session_state.chat_history = []
                            st.rerun()
                    with col_export:
                        # Export conversation
                        chat_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.chat_history])
                        st.download_button(
                            label="Export",
                            data=chat_text,
                            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="export_chat",
                            use_container_width=True
                        )
        
        st.markdown("---")
        st.markdown("### Comprehensive Performance Dashboard")
        
        # Key Performance Indicators
        st.markdown("#### Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_load = df_clean['Player Load'].mean()
            st.metric("Avg Player Load", f"{avg_load:.1f}", delta=f"{(avg_load - 250):.1f} vs baseline")
        
        with col2:
            avg_distance = df_clean['Distance (miles)'].mean()
            st.metric("Avg Distance (mi)", f"{avg_distance:.2f}")
        
        with col3:
            max_speed = df_clean['Top Speed (mph)'].max()
            st.metric("Peak Speed (mph)", f"{max_speed:.1f}")
        
        with col4:
            avg_energy = df_clean['Energy (kcal)'].mean()
            st.metric("Avg Energy (kcal)", f"{avg_energy:.0f}")
        
        # Correlation Analysis
        st.markdown("#### 🔗 Correlation Analysis")
        st.info("💡 **Coach Insight:** This heatmap shows which metrics are strongly related. Strong positive correlations (red) indicate metrics that increase together.")
        
        metrics_for_corr = [
            'Player Load', 'Energy (kcal)', 'Distance (miles)', 
            'Top Speed (mph)', 'Hr Load', 'Sprint Distance (yards)',
            'Distance Per Min (yd/min)', 'Power Score (w/kg)'
        ]
        metrics_for_corr = [m for m in metrics_for_corr if m in df_clean.columns]
        
        # Interactive correlation heatmap with Plotly
        corr_matrix = df_clean[metrics_for_corr].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(
            title='Performance Metrics Correlation Matrix (Interactive)',
            height=700,
            xaxis_title="",
            yaxis_title="",
            hovermode='closest'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("""
        **Key Findings:**
        - **Strong correlations (>0.7)** indicate metrics that move together (e.g., Distance and Energy)
        - **Weak correlations (<0.3)** suggest independent performance aspects
        - Use this to avoid redundant metrics in training load calculations
        """)
        
        # Distribution Analysis
        st.markdown("#### 📈 Performance Distribution Analysis")
        st.info("💡 **Coach Insight:** These distributions show how your team performs across key metrics. Look for outliers and consistency.")
        
        metrics_to_plot = ['Player Load', 'Energy (kcal)', 'Top Speed (mph)', 'Distance (miles)']
        metrics_to_plot = [m for m in metrics_to_plot if m in df_clean.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            sns.histplot(df_clean[metric], kde=True, ax=axes[idx], color='#1E88E5', bins=30)
            axes[idx].axvline(df_clean[metric].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            axes[idx].axvline(df_clean[metric].median(), color='green', linestyle='--', linewidth=2, label='Median')
            axes[idx].set_title(f'{metric} Distribution', fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation Guide:**
        - **Normal distribution:** Most players cluster around the mean (good consistency)
        - **Skewed distribution:** Some players significantly outperform or underperform
        - **Mean vs Median:** If significantly different, indicates outliers pulling the average
        """)
        
        # Time Series Analysis
        if 'Date' in df_clean.columns and 'Player Name' in df_clean.columns:
            st.markdown("#### 📅 Temporal Performance Trends")
            st.info("💡 **Coach Insight:** Track how player loads evolve over time. Identify fatigue accumulation or improvement trends.")
            
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%Y%m%d', errors='coerce')
            
            selected_player = st.selectbox("Select Player", df_clean['Player Name'].unique(), key='temporal_player_selector')
            
            player_data = df_clean[df_clean['Player Name'] == selected_player].sort_values('Date')
            
            # Add session type filter for temporal trends
            has_tags_temporal = 'Tags' in df_clean.columns
            if has_tags_temporal:
                session_type_temporal = st.selectbox("Session Type", ["All Sessions", "Training Only", "Games Only"], key='temporal_session_filter')
                
                if session_type_temporal == "Training Only":
                    player_data = player_data[player_data['Tags'].str.contains('training', case=False, na=False)]
                elif session_type_temporal == "Games Only":
                    player_data = player_data[player_data['Tags'].str.contains('game', case=False, na=False)]
            
            # Update titles with filter suffix
            if has_tags_temporal:
                if session_type_temporal == "Training Only":
                    filter_suffix_temporal = " (Training)"
                elif session_type_temporal == "Games Only":
                    filter_suffix_temporal = " (Games)"
                else:
                    filter_suffix_temporal = " (All Sessions)"
            else:
                filter_suffix_temporal = ""
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # Player Load over time
            axes[0].plot(player_data['Date'], player_data['Player Load'], marker='o', color='#1E88E5', linewidth=2)
            axes[0].axhline(player_data['Player Load'].mean(), color='red', linestyle='--', label='Average')
            axes[0].fill_between(player_data['Date'], player_data['Player Load'], alpha=0.3)
            axes[0].set_title(f'{selected_player} - Player Load Over Time{filter_suffix_temporal}', fontweight='bold')
            axes[0].set_ylabel('Player Load')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Energy expenditure over time
            axes[1].plot(player_data['Date'], player_data['Energy (kcal)'], marker='s', color='#43A047', linewidth=2)
            axes[1].axhline(player_data['Energy (kcal)'].mean(), color='red', linestyle='--', label='Average')
            axes[1].fill_between(player_data['Date'], player_data['Energy (kcal)'], alpha=0.3, color='#43A047')
            axes[1].set_title(f'{selected_player} - Energy Expenditure Over Time{filter_suffix_temporal}', fontweight='bold')
            axes[1].set_ylabel('Energy (kcal)')
            axes[1].set_xlabel('Date')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate trend
            recent_avg = player_data.tail(5)['Player Load'].mean()
            overall_avg = player_data['Player Load'].mean()
            trend = "increasing" if recent_avg > overall_avg else "decreasing"
            
            st.markdown(f"""
                **Trend Analysis for {selected_player}:**
                - Recent 5-session average: **{recent_avg:.1f}**
                - Overall average: **{overall_avg:.1f}**
                - Trend: **{trend.upper()}** ({((recent_avg - overall_avg) / overall_avg * 100):.1f}% change)
                - **Recommendation:** {"Monitor closely for overtraining" if trend == "increasing" and recent_avg > overall_avg * 1.15 else "Performance is stable"}
                """)
            
            # Speed Zones Analysis
            st.markdown("#### ⚡ Speed Zone Distribution")
            st.info("💡 **Coach Insight:** Understanding speed zones helps optimize training intensity and match preparation.")
            
            speed_zones = [
                'Distance in Speed Zone 1  (miles)',
                'Distance in Speed Zone 2  (miles)',
                'Distance in Speed Zone 3  (miles)',
                'Distance in Speed Zone 4  (miles)',
                'Distance in Speed Zone 5  (miles)'
            ]
            speed_zones = [sz for sz in speed_zones if sz in df_clean.columns]
            
            if speed_zones:
                # Check if Tags column exists for session type filtering
                has_tags = 'Tags' in df_clean.columns
                
                if has_tags:
                    session_type = st.selectbox("Session Type", ["All Sessions", "Training Only", "Games Only"], key='speed_session_filter')
                
                # Player selection for speed zones
                if 'Player Name' in df_clean.columns:
                    selected_player_speed = st.selectbox(
                        "Select Player for Speed Zone Analysis",
                        ['All Team Average'] + list(df_clean['Player Name'].unique()),
                        key='speed_player_selector'
                    )
                    
                    if selected_player_speed != 'All Team Average':
                        df_filtered_speed = df_clean[df_clean['Player Name'] == selected_player_speed].copy()
                    else:
                        df_filtered_speed = df_clean.copy()
                else:
                    df_filtered_speed = df_clean.copy()
                    selected_player_speed = None
                
                # Apply session type filter
                if has_tags:
                    if session_type == "Training Only":
                        df_filtered_speed = df_filtered_speed[df_filtered_speed['Tags'].str.contains('training', case=False, na=False)]
                        filter_suffix = " (Training)"
                    elif session_type == "Games Only":
                        df_filtered_speed = df_filtered_speed[df_filtered_speed['Tags'].str.contains('game', case=False, na=False)]
                        filter_suffix = " (Games)"
                    else:
                        filter_suffix = " (All Sessions)"
                else:
                    filter_suffix = ""
                
                # Calculate speed data
                if selected_player_speed != 'All Team Average':
                    speed_data = df_filtered_speed[speed_zones].mean()
                    title_suffix = f" - {selected_player_speed}{filter_suffix}"
                else:
                    speed_data = df_filtered_speed[speed_zones].mean()
                    title_suffix = f" - Team Average{filter_suffix}"
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
                bars = ax.bar(range(len(speed_data)), speed_data.values, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_xticks(range(len(speed_data)))
                ax.set_xticklabels(['Zone 1\n(Low)', 'Zone 2\n(Light)', 'Zone 3\n(Moderate)', 'Zone 4\n(High)', 'Zone 5\n(Sprint)'])
                ax.set_ylabel('Average Distance (miles)', fontweight='bold')
                ax.set_title(f'Speed Zone Distribution{title_suffix}', fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
                zone_5_pct = (speed_data.iloc[-1] / speed_data.sum() * 100) if len(speed_data) > 0 and speed_data.sum() > 0 else 0
                zone_5_insight = ""
                if zone_5_pct < 5:
                    zone_5_insight = "⚠️ Increase high-intensity work to build match fitness"
                elif zone_5_pct > 15:
                    zone_5_insight = "⚠️ Consider recovery protocols - too much high intensity"
                else:
                    zone_5_insight = "✅ Maintain current intensity distribution"
                
                st.markdown(f"""
                **Speed Zone Insights{title_suffix}:**
                - **Zone 5 (Sprint)** represents **{zone_5_pct:.1f}%** of total distance
                - **Optimal range:** 5-15% for match preparation
                - **Action:** {zone_5_insight}
                """)
            
            # Team Comparison
            if 'Player Name' in df_clean.columns:
                st.markdown("#### 🏃 Player Comparison Matrix")
                st.info("💡 **Coach Insight:** Quickly identify top performers and players needing development in each category.")
                
                player_summary = df_clean.groupby('Player Name').agg({
                    'Player Load': 'mean',
                    'Distance (miles)': 'mean',
                    'Top Speed (mph)': 'max',
                    'Energy (kcal)': 'mean',
                    'Sprint Distance (yards)': 'mean'
                }).reset_index()
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Normalize data for heatmap
                player_summary_norm = player_summary.copy()
                for col in player_summary.columns[1:]:
                    player_summary_norm[col] = (player_summary[col] - player_summary[col].min()) / (player_summary[col].max() - player_summary[col].min())
                
                sns.heatmap(
                    player_summary_norm.set_index('Player Name'),
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    center=0.5,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8, "label": "Normalized Performance (0-1)"},
                    ax=ax
                )
                ax.set_title('Player Performance Heatmap (Normalized)', fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('Performance Metrics', fontweight='bold')
                ax.set_ylabel('Player Name', fontweight='bold')
                
                st.pyplot(fig)
                
                st.markdown("""
                **Heatmap Guide:**
                - **Green (high values):** Player excels in this metric
                - **Yellow (moderate):** Average performance
                - **Red (low values):** Area for improvement
                - Use this to create individualized training programs
                """)
            
            # Final Recommendations
            st.markdown("### 📝 Comprehensive Coaching Recommendations")
            
            st.markdown("""
            <div class="success-box">
            <h4>🎯 Data-Driven Coaching Strategy</h4>
            
            <h5>1. Training Load Management</h5>
            <ul>
                <li>Monitor weekly load accumulation to prevent spikes > 20% week-over-week</li>
                <li>Implement 3:1 or 4:1 work-to-recovery ratio (3-4 hard days, 1 easy day)</li>
                <li>Use Player Load as primary metric for overall workload tracking</li>
            </ul>
            
            <h5>2. Individualization</h5>
            <ul>
                <li>Players in bottom 25% for specific metrics need targeted development plans</li>
                <li>High performers (top 10%) can handle advanced training protocols</li>
                <li>Mid-range players benefit from progressive overload strategies</li>
            </ul>
            
            <h5>3. Match Preparation</h5>
            <ul>
                <li>Peak speed work should occur 48-72 hours before matches</li>
                <li>Taper training load by 30-40% in 24 hours pre-match</li>
                <li>Focus on tactical work rather than physical loading close to games</li>
            </ul>
            
            <h5>4. Recovery Protocols</h5>
            <ul>
                <li>Players with HR Load > team mean + 1 SD need extra recovery day</li>
                <li>High Impact zones (>20G) correlate with muscle damage - monitor closely</li>
                <li>Red Zone time > 15 min indicates need for enhanced recovery strategies</li>
            </ul>
            
            <h5>5. Performance Benchmarks</h5>
            <ul>
                <li><b>Elite Player Load:</b> 350-450 per session</li>
                <li><b>Target Distance:</b> 4-7 miles per training session</li>
                <li><b>Sprint Work:</b> 200-400 yards high-speed running per session</li>
                <li><b>Energy Expenditure:</b> 600-900 kcal per hour of activity</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show a subtle, non-blocking info only (no warning/errors on cloud)
            st.markdown(
                "<div style=\"font-size: 0.85rem; color: #6C757D;\">AI Assistant is disabled on this deployment. To use it, run the app locally with Ollama running.</div>",
                unsafe_allow_html=True,
            )

elif page == "Load Prediction":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0; text-align: center; font-size: 2.5rem;'>🎯 Player Load Prediction</h1>
        <p style='color: rgba(255,255,255,0.9); text-align: center; margin-top: 0.5rem; font-size: 1.1rem;'>Predict next session load based on machine learning analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    elif st.session_state.regression_model is None:
        st.warning("⚠️ Please train the regression model first in Model Training section.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_regression(df)
        
        # Improved player and session selection with cards
        st.markdown("""
        <div style='background: #F8F9FA; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea; margin-bottom: 2rem;'>
            <h3 style='margin: 0 0 1rem 0; color: #333;'>👤 Select Player & Session Type</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if 'Player Name' in df_clean.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_player_pred = st.selectbox(
                    "👤 Select Player",
                    ['Choose a player...'] + sorted(df_clean['Player Name'].unique()),
                    key="pred_player_selector"
                )
            
            with col2:
                session_type = st.selectbox(
                    "📅 Session Type",
                    ['Training Session', 'Match']
                )
            
            if selected_player_pred != 'Choose a player...':
                # Get player's recent data
                player_data = df_clean[df_clean['Player Name'] == selected_player_pred]
                
                # Check if Tags column exists and filter by session type
                if 'Tags' in df_clean.columns:
                    # Filter based on selected session type
                    if session_type == 'Match':
                        player_data_filtered = player_data[player_data['Tags'].str.contains('game', case=False, na=False)]
                    else:
                        player_data_filtered = player_data[player_data['Tags'].str.contains('training', case=False, na=False)]
                    
                    # If no data for the specific session type, use all data
                    if len(player_data_filtered) == 0:
                        player_data_filtered = player_data
                else:
                    player_data_filtered = player_data
                
                if len(player_data_filtered) > 0:
                    # Enhanced header for prediction section
                    session_emoji = "⚽" if session_type == "Match" else "🏃"
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                        <h2 style='color: white; margin: 0; text-align: center;'>{session_emoji} Predicting Load for <strong>{selected_player_pred}</strong></h2>
                        <p style='color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0;'>Next Session: {session_type}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get the most recent session - prioritize 'all' split which has total session data
                    if 'Session Title' in player_data_filtered.columns and 'Split Name' in player_data_filtered.columns:
                        all_splits = player_data_filtered[player_data_filtered['Split Name'].str.lower() == 'all'].copy()
                        
                        if len(all_splits) > 0:
                            # Sort by Date descending to get the most recent session
                            if 'Date' in all_splits.columns:
                                all_splits = all_splits.sort_values('Date', ascending=False).reset_index(drop=True)
                            
                            # Get the most recent 'all' record
                            recent_session = all_splits.iloc[0].copy()
                            
                            # Calculate actual playing time by summing periods/halves for this session
                            session_title = recent_session['Session Title'] if 'Session Title' in recent_session.index else None
                            session_date = recent_session['Date'] if 'Date' in recent_session.index else None
                            
                            if session_title and session_date:
                                # Get all splits for this specific session from original df
                                session_splits = df_clean[
                                    (df_clean['Session Title'] == session_title) &
                                    (df_clean['Date'] == session_date) &
                                    (df_clean['Player Name'] == selected_player_pred)
                                ]
                                
                                # Filter for playing time splits (exclude 'all' and empty values)
                                # Include only actual playing periods (halves, periods, etc.)
                                playing_splits = session_splits[
                                    (~session_splits['Split Name'].str.lower().isin(['all', 'full', 'complete'])) &
                                    (session_splits['Split Name'].str.strip() != '')
                                ]
                                
                                # Also try to match common period patterns (1st, 2nd, first, second, period)
                                if len(playing_splits) > 0 and 'Duration' in playing_splits.columns:
                                    # Sum actual playing time from all valid periods
                                    total_playing_duration = playing_splits['Duration'].sum()
                                    
                                    # Different approach for training vs matches
                                    # Check if this is a match or training based on Tags
                                    is_match = False
                                    if 'Tags' in recent_session.index and recent_session['Tags']:
                                        is_match = 'game' in str(recent_session['Tags']).lower()
                                    
                                    if is_match:
                                        # For matches, use actual splits duration (already sums halves)
                                        # Matches have breaks but we want the active playing time
                                        recent_session['Duration'] = total_playing_duration
                                    else:
                                        # For training, check if there's substantial activity
                                        if 'Distance (miles)' in playing_splits.columns or 'Distance (yards)' in playing_splits.columns:
                                            distance_col = 'Distance (miles)' if 'Distance (miles)' in playing_splits.columns else 'Distance (yards)'
                                            total_distance = playing_splits[distance_col].sum()
                                            
                                            # Training with low distance might be warm-up/cool-down
                                            if total_distance > 0.1:
                                                # Good activity during training
                                                recent_session['Duration'] = total_playing_duration
                                            else:
                                                # Low activity suggests warm-up/cool-down only
                                                recent_session['Duration'] = total_playing_duration * 0.7
                                        else:
                                            # No distance data, use raw duration
                                            recent_session['Duration'] = total_playing_duration
                                elif len(session_splits) > 0 and 'Duration' in session_splits.columns:
                                    # Fallback: use all splits except 'all'
                                    all_splits = session_splits[
                                        ~session_splits['Split Name'].str.lower().isin(['all', 'full', 'complete'])
                                    ]
                                    if len(all_splits) > 0:
                                        recent_session['Duration'] = all_splits['Duration'].sum()
                        else:
                            # No 'all' splits, take the last record from filtered data
                            if len(player_data_filtered) > 0:
                                recent_session = player_data_filtered.iloc[-1]
                            else:
                                recent_session = None
                    else:
                        # No Split Name column, just get the last record sorted by Date
                        if 'Date' in player_data_filtered.columns:
                            sorted_data = player_data_filtered.sort_values('Date', ascending=False).reset_index(drop=True)
                            recent_session = sorted_data.iloc[0] if len(sorted_data) > 0 else None
                        else:
                            recent_session = player_data_filtered.iloc[-1]
                    
                    # Check if we have a valid recent_session
                    if recent_session is not None:
                        # Helper function to format duration
                        def format_duration(seconds):
                            """Convert seconds to human readable format"""
                            if pd.isna(seconds) or seconds is None:
                                return "N/A"
                            try:
                                seconds = int(float(seconds))
                                hours = seconds // 3600
                                minutes = (seconds % 3600) // 60
                                secs = seconds % 60
                                if hours > 0:
                                    return f"{hours}h {minutes}m {secs}s"
                                elif minutes > 0:
                                    return f"{minutes}m {secs}s"
                                else:
                                    return f"{secs}s"
                            except:
                                return "N/A"
                        
                        # Enhanced metrics display with cards
                        st.markdown("""
                        <div style='background: #F8F9FA; padding: 1rem; border-radius: 10px; margin: 1.5rem 0;'>
                            <h3 style='margin: 0 0 1rem 0; color: #333;'>📈 Recent Session Performance</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                            <div style='background: white; padding: 1.2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #1E88E5;'>
                                <h4 style='margin: 0 0 1rem 0; color: #1E88E5;'>⚡ Performance Metrics</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            player_load = safe_get_metric(recent_session.get('Player Load', 0), 0)
                            st.metric("📊 Player Load", f"{player_load:.1f}" if player_load > 0 else "N/A", 
                                     delta=f"Baseline: {player_load:.1f}" if player_load > 0 else None)
                            
                            energy = safe_get_metric(recent_session.get('Energy (kcal)', 0), 0)
                            st.metric("🔥 Energy (kcal)", f"{energy:.0f}" if energy > 0 else "N/A")
                            
                            if 'Duration' in recent_session.index:
                                duration = format_duration(recent_session['Duration'])
                                st.metric("⏱️ Duration", duration)
                            else:
                                st.metric("⏱️ Duration", "N/A")
                        
                        with col2:
                            st.markdown("""
                            <div style='background: white; padding: 1.2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #43A047;'>
                                <h4 style='margin: 0 0 1rem 0; color: #43A047;'>🏃 Movement Metrics</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            distance = safe_get_metric(recent_session.get('Distance (miles)', 0), 0)
                            st.metric("📏 Distance", f"{distance:.2f} miles" if distance > 0 else "N/A")
                            
                            top_speed = safe_get_metric(recent_session.get('Top Speed (mph)', 0), 0)
                            st.metric("🚀 Top Speed", f"{top_speed:.1f} mph" if top_speed > 0 else "N/A")
                        
                        with col3:
                            st.markdown("""
                            <div style='background: white; padding: 1.2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #E53935;'>
                                <h4 style='margin: 0 0 1rem 0; color: #E53935;'>❤️ Heart Rate & Impact</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            hr_load = safe_get_metric(recent_session.get('Hr Load', 0), 0)
                            st.metric("💓 HR Load", f"{hr_load:.1f}" if hr_load > 0 else "N/A")
                            
                            hr_max = safe_get_metric(recent_session.get('Hr Max (bpm)', 0), 0)
                            st.metric("❤️ HR Max", f"{hr_max:.0f} bpm" if hr_max > 0 else "N/A")
                            
                            impacts = safe_get_metric(recent_session.get('Impacts', 0), 0)
                            st.metric("💥 Impacts", f"{int(impacts)}" if impacts > 0 else "N/A")
                        
                        # Predict next session load with enhanced header
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>
                            <h2 style='color: white; margin: 0; text-align: center;'>🔮 Next Session Prediction</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prepare input features
                        features_for_pred = [
                            'Work Ratio', 'Energy (kcal)', 'Distance (miles)', 'Sprint Distance (yards)',
                            'Top Speed (mph)', 'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)',
                            'Distance Per Min (yd/min)', 'Hr Load', 'Hr Max (bpm)', 'Time In Red Zone (min)',
                            'Impacts', 'Impact Zones: > 20 G (Impacts)', 'Impact Zones: 15 - 20 G (Impacts)',
                            'Power Plays', 'Power Score (w/kg)'
                        ]
                        
                        features_for_pred = [f for f in features_for_pred if f in recent_session.index]
                        
                        if len(features_for_pred) > 0:
                            # Use recent session as baseline - create DataFrame for the model
                            X_input = pd.DataFrame({f: [recent_session[f]] for f in features_for_pred})
                            
                            try:
                                predicted_load = st.session_state.regression_model.predict(X_input)[0]
                                
                                # Adjust prediction based on session type being predicted vs. data type used
                                if session_type == 'Match':
                                    if 'Tags' in recent_session.index and recent_session['Tags'] and 'game' in str(recent_session['Tags']).lower():
                                        load_category = "High Intensity"
                                        st.info("📌 **Match Prediction:** Based on player's last match performance")
                                    else:
                                        predicted_load = predicted_load * 1.35
                                        load_category = "High Intensity"
                                        st.info("📌 **Match Prediction:** Based on last training with 35% match intensity adjustment")
                                else:
                                    load_category = "Moderate Intensity"
                                    st.info("📌 **Training Prediction:** Based on player's recent training session patterns")
                                
                                # Categorize load level
                                if predicted_load < 150:
                                    load_status = "🟢 LOW"
                                    load_risk = "Low risk - can increase intensity if desired"
                                elif predicted_load < 350:
                                    load_status = "🟡 MODERATE"
                                    load_risk = "Optimal range - good training intensity"
                                elif predicted_load < 500:
                                    load_status = "🟠 HIGH"
                                    load_risk = "High load - ensure adequate recovery after"
                                else:
                                    load_status = "🔴 VERY HIGH"
                                    load_risk = "Very high load - high injury risk, consider reducing intensity"
                                
                                col_pred1, col_pred2 = st.columns([2, 1])
                                
                                with col_pred1:
                                    # Determine background color based on load status
                                    if "LOW" in load_status:
                                        bg_color = "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)"
                                    elif "MODERATE" in load_status:
                                        bg_color = "linear-gradient(135deg, #fbb040 0%, #f9ed32 100%)"
                                    elif "HIGH" in load_status:
                                        bg_color = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                                    else:  # VERY HIGH
                                        bg_color = "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)"
                                    
                                    st.markdown(f"""
                                    <div style='background: {bg_color}; padding: 2.5rem; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                                        <div style='font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.95;'>🎯 Predicted Player Load</div>
                                        <h1 style='margin: 0.5rem 0; font-size: 5rem; font-weight: bold; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{predicted_load:.1f}</h1>
                                        <div style='background: rgba(255,255,255,0.2); padding: 0.8rem; border-radius: 10px; margin-top: 1rem;'>
                                            <p style='margin: 0; font-size: 1.3rem; font-weight: bold;'>{load_status}</p>
                                            <p style='margin: 0.3rem 0 0 0; font-size: 1rem; opacity: 0.9;'>{load_category}</p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_pred2:
                                    st.markdown("""
                                    <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-top: 4px solid #667eea;'>
                                        <h3 style='margin: 0 0 1rem 0; color: #333;'>📊 Risk Assessment</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown(f"""
                                    <div style='background: #F8F9FA; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #667eea;'>
                                        <p style='margin: 0; font-size: 1rem; color: #333;'><strong>{load_risk}</strong></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Calculate load change
                                    if 'Player Load' in recent_session.index:
                                        last_load = recent_session['Player Load']
                                        load_change = ((predicted_load - last_load) / last_load) * 100 if last_load > 0 else 0
                                        
                                        if abs(load_change) < 5:
                                            change_emoji = "➡️"
                                            change_status = "Stable"
                                        elif load_change > 20:
                                            change_emoji = "📈"
                                            change_status = "Significant Increase"
                                        elif load_change > 5:
                                            change_emoji = "📊"
                                            change_status = "Moderate Increase"
                                        elif load_change < -20:
                                            change_emoji = "📉"
                                            change_status = "Significant Decrease"
                                        else:
                                            change_emoji = "📉"
                                            change_status = "Moderate Decrease"
                                        
                                        st.metric("Load Change", f"{change_emoji} {abs(load_change):.1f}%")
                                        
                                        if abs(load_change) > 20:
                                            st.warning(f"⚠️ {change_status} from last session ({load_change:+.1f}%)")
                            
                            except Exception as e:
                                st.error(f"❌ Error making prediction: {str(e)}")
                        else:
                            st.error("❌ Not enough features available for prediction")
                else:
                    st.warning(f"No data available for {selected_player_pred}")
        else:
            st.warning("⚠️ 'Player Name' column not found in dataset")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### ℹ️ About
**Elite Sports Performance Analytics**

Advanced analytics platform for optimizing player performance, preventing injuries, and maximizing team potential.

**Version:** 1.0.0  
**Developer:** Alvaro Martin-Pena  
**Powered by:** Machine Learning & Sports Science

⚠️ **Ethical Note:** This system is designed to support coaching decisions, not replace professional judgment. Always combine data insights with coaching experience and medical expertise.
""")