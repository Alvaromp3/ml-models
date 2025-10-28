import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pipe = None

def limpiar_datos(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    numeric_col = df_clean.select_dtypes(include=['float64','int64']).columns
    
    for col in numeric_col:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1 
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        df_clean = df_clean[(df_clean[col] >= limite_inferior) & 
                           (df_clean[col] <= limite_superior)]
    
    print(f"Clean dataset: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
    return df_clean

def entrenar_modelo(df):
    global pipe
    
    df_clean = limpiar_datos(df)
    
    X = df_clean.drop(columns=['precio_eur'])
    y = df_clean['precio_eur']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    numeric_features = ['edad','kilometraje_km','potencia_cv','consumo_l_100km','num_duenos','puertas']
    cat_features = ['marca','modelo','combustible','transmision','estado','region','color','carroceria']
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('col', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    pipe = Pipeline([
        ('preprocessing', preprocessing),
        ('model', RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_leaf=2, random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return r2, mae, rmse

def predecir(edad, kilometraje_km, potencia_cv, consumo_l_100km, num_duenos, puertas,
            marca, modelo, combustible, transmision, estado, region, color, carroceria):
    
    if pipe is None:
        return "Error: Model must be trained first"
    
    datos_entrada = pd.DataFrame({
        'edad': [edad],
        'kilometraje_km': [kilometraje_km],
        'potencia_cv': [potencia_cv],
        'consumo_l_100km': [consumo_l_100km],
        'num_duenos': [num_duenos],
        'puertas': [puertas],
        'marca': [marca],
        'modelo': [modelo],
        'combustible': [combustible],
        'transmision': [transmision],
        'estado': [estado],
        'region': [region],
        'color': [color],
        'carroceria': [carroceria]
    })
    
    try:
        prediccion = pipe.predict(datos_entrada)
        return round(prediccion[0], 2)
    except Exception as e:
        return f"Prediction error: {str(e)}"