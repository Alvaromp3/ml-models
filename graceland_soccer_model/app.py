import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

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
from imblearn.over_sampling import SMOTE

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

# Set page config
st.set_page_config(
    page_title="Elite Sports Performance Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E88E5 0%, #43A047 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #F8D7DA;
        border-left: 5px solid #DC3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
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
    
    # Professional injury risk calculation with medical-grade thresholds
    if 'Player Load' in df_clean.columns:
        # Calculate percentiles for clinically-based risk assessment
        load_95th = df_clean['Player Load'].quantile(0.95)  # Only top 5% - truly elevated load
        load_80th = df_clean['Player Load'].quantile(0.80)  # Upper moderate
        load_median = df_clean['Player Load'].median()
        load_20th = df_clean['Player Load'].quantile(0.20)  # Lower threshold
        
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
                    # MEDIUM RISK: Moderate elevation or one risk factor
                    df_clean['Player Load'] > load_20th,
                    'medium',
                    # LOW RISK: Low load across all factors
                    'low'
                )
            )
        elif has_energy:
            energy_85th = df_clean['Energy (kcal)'].quantile(0.85)
            df_clean['injury_risk'] = np.where(
                (df_clean['Player Load'] > load_95th) |
                ((df_clean['Player Load'] > load_80th) & (df_clean['Energy (kcal)'] > energy_85th)),
                'high',
                np.where(df_clean['Player Load'] > load_20th, 'medium', 'low')
            )
        else:
            # Simple risk based only on load
            df_clean['injury_risk'] = np.where(
                df_clean['Player Load'] > load_95th,
                'high',
                np.where(df_clean['Player Load'] > load_20th, 'medium', 'low')
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

def train_regression_model_fast(df):
    """Fast regression model training with train/test split"""
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
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), features)])
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=1.0,
        min_samples_split=2,
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
    
    return pipe, metrics, features

def train_classification_model_fast(df):
    """Fast classification model training with high accuracy"""
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
    
    return pipe, metrics, features

# Sidebar Navigation
st.sidebar.markdown("# ⚽ Elite Sports Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Data Audit", "🤖 Model Training", "📈 Player Load Analysis", 
     "🛡️ Injury Prevention", "👥 Team Lineup Calculator", "📉 Performance Analytics", "🎯 Load Prediction"]
)

# Main Content
if page == "📊 Data Audit":
    st.markdown('<h1 class="main-header">📊 Data Audit & Quality Control</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows and {df.shape[1]} columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Players", df['Player Name'].nunique() if 'Player Name' in df.columns else 'N/A')
        with col2:
            st.metric("Total Sessions", df.shape[0])
        with col3:
            st.metric("Features", df.shape[1])
        
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📊 Data Quality Report")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numeric Variables:** {len(numeric_cols)}")
            st.info(f"**Categorical Variables:** {len(cat_cols)}")
        
        with col2:
            nulls = df.isnull().sum()
            if nulls.sum() > 0:
                st.warning(f"**Missing Values:** {nulls.sum()} total")
            else:
                st.success("**No Missing Values** ✓")
        
        if 'Player Load' in df.columns:
            st.markdown("### 📈 Player Load Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df['Player Load'], kde=True, color='#1E88E5', ax=ax)
            ax.set_title('Player Load Distribution', fontsize=14, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("### 🔧 Outlier Detection & Cleaning")
        
        if st.button("🧹 Clean Data & Remove Outliers", type="primary"):
            with st.spinner("Cleaning data..."):
                df_clean = limpiar_datos_regression(df)
                st.session_state.df_clean = df_clean
                st.success("✅ Data cleaned successfully!")
                
                st.markdown("#### Before vs After")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", df.shape[0])
                with col2:
                    st.metric("Cleaned Rows", df_clean.shape[0])

elif page == "🤖 Model Training":
    st.markdown('<h1 class="main-header">🤖 Advanced Model Training</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        
        tab1, tab2 = st.tabs(["⚽ Player Load Prediction (Regression)", "🛡️ Injury Risk Classification"])
        
        with tab1:
            st.markdown("### Train Player Load Prediction Model")
            st.info("This model predicts Player Load based on performance metrics")
            
            if st.button("🚀 Train Regression Model", type="primary", key="train_reg"):
                with st.spinner("Training model... This may take a moment"):
                    model, metrics, features = train_regression_model_fast(df)
                    st.session_state.regression_model = model
                    
                    st.success("✅ Model trained successfully!")
                    
                    st.markdown("### 📊 Train vs Test Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🟢 Training Set")
                        train_r2 = metrics['train']['R2']
                        st.metric("R² Score", f"{train_r2:.4f}")
                        st.metric("MAE", f"{metrics['train']['MAE']:.2f}")
                        st.metric("RMSE", f"{metrics['train']['RMSE']:.2f}")
                    
                    with col2:
                        st.markdown("#### 🔵 Test Set")
                        test_r2 = metrics['test']['R2']
                        st.metric("R² Score", f"{test_r2:.4f}")
                        st.metric("MAE", f"{metrics['test']['MAE']:.2f}")
                        st.metric("RMSE", f"{metrics['test']['RMSE']:.2f}")
                    
                    # Check for overfitting
                    diff = abs(train_r2 - test_r2)
                    if diff > 0.15:
                        st.warning(f"⚠️ **Warning:** Potential overfitting! Train R² ({train_r2:.4f}) vs Test R² ({test_r2:.4f}) gap: {diff:.4f}")
                    elif diff > 0.10:
                        st.info(f"⚠️ **Moderate overfitting:** Gap: {diff:.4f}")
                    else:
                        st.success(f"✅ **No overfitting!** Gap: {diff:.4f}")
                    
                    if metrics['test']['R2'] >= 0.90:
                        st.balloons()
                        st.markdown('<div class="success-box">🎯 <b>Excellent Performance!</b> R² score exceeds 0.90 threshold</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Features used:** {len(features)}")
                    with st.expander("View all features"):
                        st.write(features)
        
        with tab2:
            st.markdown("### Train Injury Risk Classification Model")
            st.info("This model classifies players into low/medium/high injury risk categories")
            
            if st.button("🚀 Train Classification Model", type="primary", key="train_class"):
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

elif page == "📈 Player Load Analysis":
    st.markdown('<h1 class="main-header">📈 Player Load Analysis</h1>', unsafe_allow_html=True)
    
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

elif page == "🛡️ Injury Prevention":
    st.markdown('<h1 class="main-header">🛡️ Injury Risk Prevention System</h1>', unsafe_allow_html=True)
    
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
            player_risk['Risk_Category'] = player_risk['max'].map({0: 'Low', 1: 'Medium', 2: 'High'})
            player_risk = player_risk.sort_values('mean', ascending=False)
            
            st.markdown("### 🚨 Current Injury Risk Status")
            
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

elif page == "👥 Team Lineup Calculator":
    st.markdown("""
    <h1 style='color:#1E88E5; font-size:2.5rem; font-weight:bold; text-align:center; padding:1rem;'>
        👥 Optimal Team Lineup Calculator
    </h1>
    """, unsafe_allow_html=True)
    
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

elif page == "📉 Performance Analytics":
    st.markdown('<h1 class="main-header">📉 Advanced Performance Analytics</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_regression(df)
        
        # AI Coach Assistant - Explain Analytics
        with st.expander("🤖 AI Coach Assistant - Understand Your Analytics", expanded=True):
            st.markdown("### 💬 Your AI Assistant Explains Each Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            # Player selection for personalized explanations
            selected_player_for_explain = None
            if 'Player Name' in df_clean.columns:
                col_select_player, col_select_analysis = st.columns([1, 2])
                
                with col_select_player:
                    explain_player = st.selectbox(
                        "👤 Select Player (Optional)",
                        ["All Players"] + list(df_clean['Player Name'].unique()),
                        key="explain_player_selector"
                    )
                    
                    if explain_player != "All Players":
                        selected_player_for_explain = explain_player
                        st.info(f"📊 Explaining for: **{explain_player}**")
                
                with col_select_analysis:
                    analysis_to_explain = st.selectbox(
                        "🎯 What do you want to understand?",
                        [
                            "Select an analysis...",
                            "📊 Key Performance Indicators (KPIs)",
                            "🔗 Correlation Analysis",
                            "📈 Performance Distribution",
                            "📅 Temporal Performance Trends",
                            "⚡ Speed Zone Distribution",
                            "🏃 Player Comparison Matrix",
                            "👥 All Analytics Overview"
                        ],
                        key="analysis_selector"
                    )
            else:
                analysis_to_explain = st.selectbox(
                    "🎯 What do you want to understand?",
                    [
                        "Select an analysis...",
                        "📊 Key Performance Indicators (KPIs)",
                        "🔗 Correlation Analysis",
                        "📈 Performance Distribution",
                        "📅 Temporal Performance Trends",
                        "⚡ Speed Zone Distribution",
                        "🏃 Player Comparison Matrix",
                        "👥 All Analytics Overview"
                    ]
                )
            
            # AI Explanations
            if analysis_to_explain != "Select an analysis...":
                if analysis_to_explain == "📊 Key Performance Indicators (KPIs)":
                    st.markdown("""
                    <div class="info-box">
                    <h4>📊 What Are KPIs?</h4>
                    <p><b>Think of KPIs as your team's vital signs.</b></p>
                    <ul>
                        <li><b>Player Load:</b> Total training stress per session. High scores (350+) mean high intensity work.</li>
                        <li><b>Distance:</b> How many miles covered. Shows overall work capacity.</li>
                        <li><b>Top Speed:</b> Maximum sprint speed. Elite players hit 18-20 mph.</li>
                        <li><b>Energy:</b> Calories burned. Indicates metabolic workload.</li>
                    </ul>
                    <p><b>Why it matters:</b> These numbers tell you who's working hardest, who needs more intensity, and who might be overtraining.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "🔗 Correlation Analysis":
                    st.markdown("""
                    <div class="info-box">
                    <h4>🔗 Understanding Correlations</h4>
                    <p><b>This shows which metrics move together.</b></p>
                    <ul>
                        <li><b>Red/Hot areas (+0.7 to +1.0):</b> Strong positive relationship - when one goes up, the other does too (e.g., Distance and Energy burn together)</li>
                        <li><b>Blue areas (-0.7 to -1.0):</b> Strong negative - they move opposite (rare in performance data)</li>
                        <li><b>Yellow areas (+0.3 to -0.3):</b> Weak relationship - they're independent</li>
                    </ul>
                    <p><b>Practical use:</b> If Player Load and Distance are highly correlated (red), then increasing running volume automatically increases load. This helps you predict training responses.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "📈 Performance Distribution":
                    st.markdown("""
                    <div class="info-box">
                    <h4>📈 What Distribution Plots Show</h4>
                    <p><b>This is like a report card for your entire team.</b></p>
                    <ul>
                        <li><b>Normal bell curve:</b> Most players cluster around average - healthy team consistency</li>
                        <li><b>Peaks (tails):</b> Some players far above or below average - mix of high performers and development players</li>
                        <li><b>Mean vs Median:</b> If mean (red line) is much higher than median (green), you have a few superstars pulling the average up</li>
                    </ul>
                    <p><b>What to look for:</b> Watch for players way off to the right (overperforming - good!) or way off to the left (underperforming - needs attention).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "📅 Temporal Performance Trends":
                    if selected_player_for_explain:
                        # Get player-specific data
                        player_data_explain = df_clean[df_clean['Player Name'] == selected_player_for_explain]
                        
                        if len(player_data_explain) > 0:
                            avg_load_player = player_data_explain['Player Load'].mean() if 'Player Load' in player_data_explain.columns else 0
                            avg_energy_player = player_data_explain['Energy (kcal)'].mean() if 'Energy (kcal)' in player_data_explain.columns else 0
                            avg_speed_player = player_data_explain['Top Speed (mph)'].mean() if 'Top Speed (mph)' in player_data_explain.columns else 0
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <h4>📅 Reading {selected_player_for_explain}'s Trends Over Time</h4>
                            <p><b>This is your crystal ball for fatigue and fitness.</b></p>
                            
                            <h5>📊 {selected_player_for_explain}'s Average Metrics:</h5>
                            <ul>
                                <li><b>Player Load:</b> {avg_load_player:.1f} ({"High intensity" if avg_load_player > 250 else "Moderate"})</li>
                                <li><b>Energy:</b> {avg_energy_player:.0f} kcal per session</li>
                                <li><b>Top Speed:</b> {avg_speed_player:.1f} mph</li>
                            </ul>
                            
                            <ul>
                                <li><b>Upward trend in Load:</b> {selected_player_for_explain} is working harder - either getting fitter OR heading toward injury</li>
                                <li><b>Downward trend:</b> Either recovering well or undertraining</li>
                                <li><b>Spikes (sudden peaks):</b> Heavy sessions or matches - needs recovery days after</li>
                                <li><b>Stable line:</b> Consistent training - good for maintaining fitness</li>
                            </ul>
                            <p><b>Key insight:</b> If Load is rising but Energy is falling, that's fatigue - {selected_player_for_explain} can't sustain intensity. Warning sign!</p>
                            <p><b>Action items:</b> Rising trends need monitoring, falling Energy suggests recovery needed.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"No data available for {selected_player_for_explain}")
                    else:
                        st.markdown("""
                        <div class="info-box">
                        <h4>📅 Reading Player Trends Over Time</h4>
                        <p><b>This is your crystal ball for fatigue and fitness.</b></p>
                        <ul>
                            <li><b>Upward trend in Load:</b> Player is working harder - either getting fitter OR heading toward injury</li>
                            <li><b>Downward trend:</b> Either recovering well or undertraining</li>
                            <li><b>Spikes (sudden peaks):</b> Heavy sessions or matches - needs recovery days after</li>
                            <li><b>Stable line:</b> Consistent training - good for maintaining fitness</li>
                        </ul>
                        <p><b>Key insight:</b> If Load is rising but Energy is falling, that's fatigue - player can't sustain intensity. Warning sign!</p>
                        <p><b>Action items:</b> Rising trends need monitoring, falling Energy suggests recovery needed.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "⚡ Speed Zone Distribution":
                    st.markdown("""
                    <div class="info-box">
                    <h4>⚡ Speed Zones Explained</h4>
                    <p><b>Think of this as your training intensity recipe.</b></p>
                    <ul>
                        <li><b>Zone 1 (Green):</b> Walking/recovery. High = low-intensity recovery work</li>
                        <li><b>Zone 2 (Light green):</b> Jogging. Building endurance</li>
                        <li><b>Zone 3 (Yellow):</b> Running. Threshold work - builds aerobic capacity</li>
                        <li><b>Zone 4 (Orange):</b> Fast running. High-intensity runs</li>
                        <li><b>Zone 5 (Red):</b> Sprinting. Max speed/power work - anaerobic zone</li>
                    </ul>
                    <p><b>Elite standards:</b> 5-15% should be Zone 5 (sprints) for match-ready fitness. Less than 5% = need more high-intensity work.</p>
                    <p><b>Too much Zone 5 (>15%):</b> Risk of fatigue and injury. Too little = not ready for match intensity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "🏃 Player Comparison Matrix":
                    st.markdown("""
                    <div class="info-box">
                    <h4>🏃 Heatmap - Quick Player Performance Check</h4>
                    <p><b>This is your one-page summary for every player.</b></p>
                    <ul>
                        <li><b>Green cells:</b> Player excels in this area - strength to leverage</li>
                        <li><b>Yellow cells:</b> Average performance - room for growth</li>
                        <li><b>Red cells:</b> Below team average - needs targeted training</li>
                    </ul>
                    <p><b>How to use it:</b></p>
                    <ol>
                        <li>Find players with lots of red - they need individualized programs</li>
                        <li>Green-heavy players are your stars - manage them carefully to avoid overtraining</li>
                        <li>Use this to create position-specific training (defenders need different patterns than forwards)</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif analysis_to_explain == "👥 All Analytics Overview":
                    avg_load_val = df_clean['Player Load'].mean() if 'Player Load' in df_clean.columns else 0
                    team_count = df_clean['Player Name'].nunique() if 'Player Name' in df_clean.columns else 0
                    
                    st.markdown(f"""
                    <div class="success-box">
                    <h4>📊 Your Complete Analytics Suite Overview</h4>
                    <p>You have access to <b>7 powerful analysis tools</b> to optimize your team's performance:</p>
                    
                    <h5>📊 Dashboard Features:</h5>
                    <ul>
                        <li><b>KPIs:</b> Quick team snapshot - {team_count} players with {avg_load_val:.0f} avg load</li>
                        <li><b>Correlation:</b> Understand which training aspects connect (e.g., does speed work increase load?)</li>
                        <li><b>Distribution:</b> See who's over/under-performing compared to team averages</li>
                        <li><b>Time Series:</b> Track individual players over time to catch fatigue early</li>
                        <li><b>Speed Zones:</b> Optimize training intensity mix (% sprint vs recovery)</li>
                        <li><b>Heart Rate:</b> Monitor cardiovascular fitness and match-readiness</li>
                        <li><b>Player Comparison:</b> One-page view of every player's strengths/weaknesses</li>
                        <li><b>Recommendations:</b> Data-driven coaching strategies</li>
                    </ul>
                    
                    <h5>🎯 How to Use This:</h5>
                    <ol>
                        <li><b>Start here:</b> Read this explanation to understand what each graph means</li>
                        <li><b>Check KPIs:</b> Get quick overview of team health</li>
                        <li><b>Drill down:</b> Use player-specific analyses (Time Series, Comparisons) for individual attention</li>
                        <li><b>Take action:</b> Use Recommendations section to create training plans</li>
                    </ol>
                    
                    <p><b>💡 Pro Tip:</b> Bookmark this page and check it weekly to track improvement trends!</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Comprehensive Performance Dashboard")
        
        # Key Performance Indicators
        st.markdown("#### 🎯 Key Performance Indicators (KPIs)")
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
        
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = df_clean[metrics_for_corr].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Performance Metrics Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        
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

elif page == "🎯 Load Prediction":
    st.markdown('<h1 class="main-header">🎯 Predict Next Session Player Load</h1>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data in the Data Audit section first.")
    elif st.session_state.regression_model is None:
        st.warning("⚠️ Please train the regression model first in Model Training section.")
    else:
        df = st.session_state.df
        df_clean = limpiar_datos_regression(df)
        
        st.markdown("### 👤 Select Player & Session Type")
        
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
                    st.markdown("---")
                    st.markdown(f"### 📊 Predicting Load for {selected_player_pred} - Next {session_type}")
                    
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
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### 📈 Recent Session Metrics")
                            if 'Player Load' in recent_session.index:
                                st.metric("Last Player Load", f"{recent_session['Player Load']:.1f}")
                            if 'Energy (kcal)' in recent_session.index:
                                st.metric("Last Energy (kcal)", f"{recent_session['Energy (kcal)']:.0f}")
                            if 'Duration' in recent_session.index:
                                duration = format_duration(recent_session['Duration'])
                                st.metric("Session Duration", duration)
                        
                        with col2:
                            if 'Distance (miles)' in recent_session.index:
                                st.metric("Last Distance", f"{recent_session['Distance (miles)']:.2f} miles")
                            if 'Top Speed (mph)' in recent_session.index:
                                st.metric("Last Top Speed", f"{recent_session['Top Speed (mph)']:.1f} mph")
                        
                        with col3:
                            if 'Hr Load' in recent_session.index:
                                st.metric("Last HR Load", f"{recent_session['Hr Load']:.1f}")
                            if 'Hr Max (bpm)' in recent_session.index:
                                st.metric("Heart Rate Max", f"{recent_session['Hr Max (bpm)']:.0f} bpm")
                            if 'Impacts' in recent_session.index:
                                st.metric("Last Impacts", f"{int(recent_session['Impacts'])}")
                        
                        st.markdown("---")
                        
                        # Predict next session load
                        st.markdown("### 🔮 Next Session Prediction")
                        
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
                                    st.markdown(f"""
                                    <div style='background-color:#1E88E5; padding:2rem; border-radius:15px; color:white; text-align:center;'>
                                        <h2 style='margin:0; color:white;'>🎯 Predicted Player Load</h2>
                                        <h1 style='margin:0.5rem 0; font-size:4rem; color:white;'>{predicted_load:.1f}</h1>
                                        <p style='margin:0; font-size:1.2rem;'><b>Status:</b> {load_status}</p>
                                        <p style='margin:0; font-size:0.9rem;'>{load_category}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col_pred2:
                                    st.markdown("#### 📊 Risk Assessment")
                                    st.markdown(f"**{load_risk}**")
                                    
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
**Powered by:** Machine Learning & Sports Science

⚠️ **Ethical Note:** This system is designed to support coaching decisions, not replace professional judgment. Always combine data insights with coaching experience and medical expertise.
""")