"""
Advanced ML Extensions for Elite Sports Performance Analytics
============================================================
This module adds professional-grade ML features:
- Cross-validation metrics
- SHAP values for explainability
- MLflow integration for model tracking
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import logging
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings

# Advanced imports with fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")


def compute_cross_validation_metrics(pipeline, X, y, cv_folds=5, model_type='regression'):
    """
    Compute cross-validation metrics for robust model evaluation
    
    Args:
        pipeline: Trained sklearn pipeline
        X: Features
        y: Target
        cv_folds: Number of CV folds (default: 5)
        model_type: 'regression' or 'classification'
    
    Returns:
        dict: CV metrics (mean, std) for each metric
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    if model_type == 'regression':
        scoring_metrics = {
            'neg_mean_squared_error': 'MSE',
            'neg_mean_absolute_error': 'MAE',
            'r2': 'R2'
        }
        
        cv_results = {}
        for metric_name, display_name in scoring_metrics.items():
            scores = cross_val_score(pipeline, X, y, cv=kfold, scoring=metric_name, n_jobs=-1)
            
            if 'neg' in metric_name:
                scores = -scores  # Convert to positive
            
            cv_results[display_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return cv_results
    
    else:  # classification
        cv_results = {}
        metrics = {
            'accuracy': cross_val_score(pipeline, X, y, cv=kfold, scoring='accuracy', n_jobs=-1),
            'precision_weighted': cross_val_score(pipeline, X, y, cv=kfold, scoring='precision_weighted', n_jobs=-1),
            'recall_weighted': cross_val_score(pipeline, X, y, cv=kfold, scoring='recall_weighted', n_jobs=-1),
            'f1_weighted': cross_val_score(pipeline, X, y, cv=kfold, scoring='f1_weighted', n_jobs=-1)
        }
        
        for metric_name, scores in metrics.items():
            cv_results[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        return cv_results


def generate_shap_analysis(pipeline, X_sample, feature_names):
    """
    Generate SHAP values for model explainability
    
    Args:
        pipeline: Trained sklearn pipeline
        X_sample: Sample of features (max 100 rows for performance)
        feature_names: List of feature names
    
    Returns:
        shap.Explanation object and summary plot
    """
    if not SHAP_AVAILABLE:
        return None, None
    
    try:
        # Limit sample size for performance
        if len(X_sample) > 100:
            X_sample = X_sample.sample(100, random_state=42)
        
        # Extract the final model from pipeline
        model = pipeline.named_steps['model']
        
        # Get preprocessed features
        preprocessor = pipeline.named_steps.get('preprocessor', None)
        if preprocessor:
            X_transformed = preprocessor.transform(X_sample)
        else:
            X_transformed = X_sample
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        return shap_values, X_transformed
    
    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}")
        return None, None


def plot_shap_summary(shap_values, X_transformed, feature_names):
    """
    Create SHAP summary plot
    
    Args:
        shap_values: SHAP values from generate_shap_analysis
        X_transformed: Transformed features
        feature_names: List of feature names
    
    Returns:
        plotly figure
    """
    if shap_values is None or not SHAP_AVAILABLE:
        return None
    
    try:
        # For classification models, average across classes
        if isinstance(shap_values, list):
            shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # Compute mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create dataframe
        df_shap = pd.DataFrame({
            'Feature': feature_names[:len(mean_shap)],
            'Importance': mean_shap
        }).sort_values('Importance', ascending=True)
        
        # Create plotly bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_shap['Importance'],
            y=df_shap['Feature'],
            orientation='h',
            marker=dict(
                color=df_shap['Importance'],
                colorscale='Viridis',
                showscale=True
            ),
            text=[f"{val:.4f}" for val in df_shap['Importance']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            height=400 + len(df_shap) * 20,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logging.error(f"SHAP plot failed: {e}")
        return None


def log_to_mlflow(pipeline, model_name, metrics, params=None, X_sample=None, y_sample=None):
    """
    Log model to MLflow for tracking and versioning
    
    Args:
        pipeline: Trained sklearn pipeline
        model_name: Name for the model
        metrics: Dictionary of metrics
        params: Dictionary of hyperparameters
        X_sample: Sample features for signature
        y_sample: Sample target for signature
    """
    if not MLFLOW_AVAILABLE:
        logging.warning("MLflow not available. Skipping MLflow logging.")
        return
    
    try:
        mlflow.set_experiment("graceland_soccer_models")
        
        with mlflow.start_run():
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log metrics (flatten nested dict)
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_metric(f"{key}_{sub_key}", sub_value)
                else:
                    mlflow.log_metric(key, value)
            
            # Log model
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                signature=None  # Optional: create signature from X_sample, y_sample
            )
            
            logging.info(f"Model {model_name} logged to MLflow successfully")
            
    except Exception as e:
        logging.error(f"MLflow logging failed: {e}")


def display_cv_metrics(cv_metrics, model_type='regression'):
    """
    Display cross-validation metrics in Streamlit
    
    Args:
        cv_metrics: Dictionary from compute_cross_validation_metrics
        model_type: 'regression' or 'classification'
    """
    if model_type == 'regression':
        st.markdown("### 📊 Cross-Validation Metrics (Robustness Check)")
        st.info("5-fold CV provides robust performance estimates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r2 = cv_metrics['R2']
            st.metric("R² Score", 
                     f"{r2['mean']:.4f}",
                     delta=f"±{r2['std']:.4f}")
        
        with col2:
            mae = cv_metrics['MAE']
            st.metric("MAE", 
                     f"{mae['mean']:.2f}",
                     delta=f"±{mae['std']:.2f}")
        
        with col3:
            rmse = cv_metrics['MSE']  # MSE from CV
            st.metric("RMSE", 
                     f"{np.sqrt(rmse['mean']):.2f}",
                     delta=f"±{np.sqrt(rmse['std']):.2f}")
        
        # Display ranges
        st.markdown("#### 📈 Performance Ranges")
        ranges_df = pd.DataFrame({
            'Metric': ['R² Score', 'MAE', 'RMSE'],
            'Min': [cv_metrics['R2']['min'], cv_metrics['MAE']['min'], 
                   np.sqrt(cv_metrics['MSE']['min'])],
            'Max': [cv_metrics['R2']['max'], cv_metrics['MAE']['max'], 
                   np.sqrt(cv_metrics['MSE']['max'])],
            'Range': [
                cv_metrics['R2']['max'] - cv_metrics['R2']['min'],
                cv_metrics['MAE']['max'] - cv_metrics['MAE']['min'],
                np.sqrt(cv_metrics['MSE']['max']) - np.sqrt(cv_metrics['MSE']['min'])
            ]
        })
        st.dataframe(ranges_df, use_container_width=True)
    
    else:  # classification
        st.markdown("### 📊 Cross-Validation Metrics (Robustness Check)")
        st.info("5-fold CV provides robust performance estimates")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            acc = cv_metrics['accuracy']
            st.metric("Accuracy", 
                     f"{acc['mean']:.4f}",
                     delta=f"±{acc['std']:.4f}")
        
        with col2:
            prec = cv_metrics['precision_weighted']
            st.metric("Precision", 
                     f"{prec['mean']:.4f}",
                     delta=f"±{prec['std']:.4f}")
        
        with col3:
            rec = cv_metrics['recall_weighted']
            st.metric("Recall", 
                     f"{rec['mean']:.4f}",
                     delta=f"±{rec['std']:.4f}")
        
        with col4:
            f1 = cv_metrics['f1_weighted']
            st.metric("F1 Score", 
                     f"{f1['mean']:.4f}",
                     delta=f"±{f1['std']:.4f}")


def create_cv_box_plot(cv_metrics, model_type='regression'):
    """
    Create box plot for CV metrics visualization
    
    Args:
        cv_metrics: Dictionary from compute_cross_validation_metrics
        model_type: 'regression' or 'classification'
    
    Returns:
        plotly figure
    """
    try:
        if model_type == 'regression':
            metrics = ['R2', 'MAE', 'RMSE']
            colors = ['#2ecc71', '#e74c3c', '#3498db']
        else:
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22']
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            mean = cv_metrics[metric]['mean']
            std = cv_metrics[metric]['std']
            
            # Create synthetic distribution for visualization
            values = np.random.normal(mean, std, 5)
            
            fig.add_trace(go.Box(
                y=values,
                name=metric.replace('_', ' ').title(),
                boxmean='sd',
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Cross-Validation Metrics Distribution",
            yaxis_title="Metric Value",
            xaxis_title="Metric",
            height=400,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logging.error(f"CV box plot failed: {e}")
        return None



