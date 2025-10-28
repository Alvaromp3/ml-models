# 🏠 Predictor de Precios Airbnb

Una aplicación web que permite predecir precios de propiedades Airbnb usando un modelo KNN Regressor.

## 🚀 Cómo usar

### Opción 1: Ejecutar directamente
```bash
python3 run_app.py
```

### Opción 2: Paso a paso
```bash
# 1. Entrenar el modelo
python3 modelo_airbnb.py

# 2. Ejecutar la aplicación web
python3 app.py
```

## 📱 Uso de la aplicación

1. Abre tu navegador en `http://localhost:5000`
2. Completa el formulario con los datos de la propiedad:
   - **ID**: Identificador único
   - **Latitud/Longitud**: Coordenadas de ubicación
   - **Tipo de Propiedad**: Apartment, House, Condominium, Loft
   - **Tipo de Habitación**: Entire home/apt, Private room, Shared room
   - **Habitaciones**: Número de dormitorios
   - **Baños**: Número de baños
   - **Reseñas**: Cantidad de reseñas
   - **Disponibilidad**: Días disponibles al año
3. Haz clic en "Predecir Precio"
4. Obtén la predicción del precio

## 📊 Características del modelo

- **Algoritmo**: K-Nearest Neighbors Regressor
- **Preprocesamiento**: StandardScaler para variables numéricas
- **Variables**: 9 características de entrada
- **Precisión**: R² Score ≈ 0.36

## 📁 Archivos

- `app.py`: Aplicación Flask principal
- `modelo_airbnb.py`: Clase del modelo de ML
- `templates/index.html`: Interfaz web
- `airbnb_synthetic.csv`: Dataset de entrenamiento
- `run_app.py`: Script de ejecución simplificado