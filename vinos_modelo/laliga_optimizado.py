# ======================
# 📊 CLASIFICACIÓN DE RESULTADOS EN LALIGA - OPTIMIZADO
# ======================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def limpiar_datos(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    
    if 'attendance' in df_clean.columns:
        df_clean['attendance'] = df_clean['attendance'].fillna(df_clean['attendance'].mean())
    
    columnas_IQR = ['xg', 'xga', 'poss', 'dist']
    for col in columnas_IQR:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= limite_inferior) & (df_clean[col] <= limite_superior)]
    
    return df_clean

def entrenar_modelo_optimizado(df):
    df_clean = limpiar_datos(df)
    
    X = df_clean[['poss','sh','sot','xg','xga','dist','fk','pk','pkatt',
                  'formation','opp formation','season','venue','team','opponent']]
    y = df_clean['result']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    numeric_features = ['poss','sh','sot','xg','xga','dist','fk','pk','pkatt']
    cat_features = ['venue','team','opponent','season','formation','opp formation']
    
    preprocessing = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )
    
    # MODELOS A PROBAR
    modelos = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    mejor_accuracy = 0
    mejor_modelo = None
    mejor_nombre = ""
    
    print("🔍 Probando modelos...")
    
    for nombre, modelo in modelos.items():
        pipe = Pipeline([
            ('preprocessing', preprocessing),
            ('model', modelo)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{nombre}: {accuracy:.4f}")
        
        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = pipe
            mejor_nombre = nombre
    
    # EVALUACIÓN DEL MEJOR MODELO
    y_pred_final = mejor_modelo.predict(X_test)
    precision = precision_score(y_test, y_pred_final, average="weighted")
    recall = recall_score(y_test, y_pred_final, average="weighted")
    f1 = f1_score(y_test, y_pred_final, average="weighted")
    
    print(f"\n🏆 MEJOR MODELO: {mejor_nombre}")
    print(f"📈 Accuracy: {mejor_accuracy:.4f}")
    print(f"📈 Precision: {precision:.4f}")
    print(f"📈 Recall: {recall:.4f}")
    print(f"📈 F1-score: {f1:.4f}")
    print("\n📊 Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred_final))
    
    return mejor_modelo, mejor_accuracy

# FUNCIÓN PARA USAR CON DATOS REALES
def entrenar_con_datos(archivo_csv):
    df = pd.read_csv(archivo_csv)
    return entrenar_modelo_optimizado(df)