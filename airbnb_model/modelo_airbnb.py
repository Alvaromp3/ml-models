import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

DATASET_PATH = os.getenv('AIRBNB_DATASET_PATH', 'airbnb_synthetic.csv')

class ModeloAirbnb:
    def __init__(self):
        self.pipeline = None
        self.label_encoders = {}
        self.metrics = {}
        
    def limpiar_datos(self, df):
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        
        numeric_col = df_clean.select_dtypes(include=['float64','int64']).columns
        
        for col in numeric_col:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[col] >= limite_inferior) & (df_clean[col] <= limite_superior)]
            
        return df_clean
    
    def entrenar_modelo(self):
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
        
        df = pd.read_csv(DATASET_PATH)
        df_clean = self.limpiar_datos(df)
        
        # Codificar variables categóricas
        for col in ['property_type', 'room_type']:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])
            self.label_encoders[col] = le
        
        X = df_clean.drop(columns=['price'])
        y = df_clean['price']
        
        numeric_features = ['latitude','longitude','bedrooms','bathrooms','number_of_reviews','availability_365']
        cat_features = ['property_type','room_type']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        preprocessing = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ]
        )
        
        from xgboost import XGBRegressor
        
        # Use XGBoost for reliable high R² performance
        # Optimized hyperparameters for R² > 0.90
        self.pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=10,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.metrics = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
        
        return self.metrics
    
    def predecir(self, datos):
        # Codificar variables categóricas usando los encoders entrenados
        datos_encoded = datos.copy()
        for col in ['property_type', 'room_type']:
            if col in datos_encoded:
                datos_encoded[col] = self.label_encoders[col].transform([datos_encoded[col]])[0]
        
        # Crear DataFrame con el orden correcto de columnas
        df_pred = pd.DataFrame([datos_encoded])
        prediccion = self.pipeline.predict(df_pred)
        return prediccion[0]
    
    def guardar_modelo(self, filename='modelo_airbnb.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'pipeline': self.pipeline, 'encoders': self.label_encoders}, f)
    
    def cargar_modelo(self, filename='modelo_airbnb.pkl'):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.pipeline = data['pipeline']
                self.label_encoders = data['encoders']
            return True
        except:
            return False

if __name__ == "__main__":
    modelo = ModeloAirbnb()
    r2 = modelo.entrenar_modelo()
    print(f"R2 Score: {r2}")
    modelo.guardar_modelo()