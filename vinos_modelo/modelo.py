import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

pipe = None

def limpiar_datos(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    # Fill NaN values with mean
    numeric_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'pH', 'sulphates']
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Remove outliers using IQR
    columnas_aplicar_IQR = ['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'sulphates']
    
    for col in columnas_aplicar_IQR:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= limite_inferior) & (df_clean[col] <= limite_superior)]
    
    return df_clean

def entrenar_modelo():
    global pipe
    
    dataset_path = os.getenv('WINE_DATASET_PATH', 'winequalityN.csv')
    df = pd.read_csv(dataset_path)
    df_clean = limpiar_datos(df)
    
    X = df_clean.drop(columns=['quality'])
    y = (df_clean['quality'] >= 6).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                       'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    cat_features = ['type']
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('model', XGBClassifier(
            n_estimators=250,
            learning_rate=0.06,
            max_depth=3,
            min_child_weight=10,
            subsample=0.55,
            colsample_bytree=0.55,
            gamma=0.8,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    
    # Metrics
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
        'train_precision': float(precision_score(y_train, y_pred_train, zero_division=0)),
        'test_precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'train_recall': float(recall_score(y_train, y_pred_train, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'train_f1': float(f1_score(y_train, y_pred_train, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred_test, zero_division=0))
    }
    
    return metrics

def predecir(data):
    global pipe
    if pipe is None:
        return None, "Modelo no entrenado"
    
    try:
        df_input = pd.DataFrame([data])
        prediction = pipe.predict(df_input)[0]
        probability = pipe.predict_proba(df_input)[0]
        
        quality = "Alta (≥6)" if prediction == 1 else "Baja (<6)"
        confidence = max(probability) * 100
        
        return quality, f"Confianza: {confidence:.1f}%"
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"

# ======================
# REGRESIÓN (R² / MAE / RMSE)
# ======================
def entrenar_modelo_regresion():
    dataset_path = os.getenv('WINE_DATASET_PATH', 'winequalityN.csv')
    df = pd.read_csv(dataset_path)
    df_clean = limpiar_datos(df)
    
    X = df_clean.drop(columns=['quality'])
    y = df_clean['quality'].astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    numeric_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                       'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    cat_features = ['type']
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=8,
        subsample=0.6,
        colsample_bytree=0.6,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1
    )
    
    pipe_reg = Pipeline([
        ('preprocessing', preprocessing),
        ('model', model)
    ])
    
    pipe_reg.fit(X_train, y_train)
    y_pred_train = pipe_reg.predict(X_train)
    y_pred_test = pipe_reg.predict(X_test)
    
    metrics = {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test)),
        'train_rmse': float(mean_squared_error(y_train, y_pred_train, squared=False)),
        'test_rmse': float(mean_squared_error(y_test, y_pred_test, squared=False)),
    }
    
    return metrics, pipe_reg

def predecir_regresion(pipe_reg, data):
    try:
        df_input = pd.DataFrame([data])
        pred = float(pipe_reg.predict(df_input)[0])
        return pred, None
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"