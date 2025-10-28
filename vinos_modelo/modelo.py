import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    
    df = pd.read_csv('winequalityN.csv')
    df_clean = limpiar_datos(df)
    
    X = df_clean.drop(columns=['quality'])
    y = (df_clean['quality'] >= 6).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        ('model', DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_split=10, min_samples_leaf=5))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

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