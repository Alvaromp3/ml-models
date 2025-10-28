# Predictor de Calidad de Vino

Aplicación Flask para predecir la calidad del vino usando machine learning.

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Ejecutar la aplicación:
```bash
python app.py
```

2. Abrir navegador en: http://localhost:5000

3. Entrenar el modelo haciendo clic en "Entrenar Modelo"

4. Introducir características del vino y predecir su calidad

## Características del modelo

- **Algoritmo**: Decision Tree Classifier
- **Preprocesamiento**: StandardScaler para variables numéricas, OneHotEncoder para categóricas
- **Limpieza**: Eliminación de duplicados, imputación de NaN con media, eliminación de outliers con IQR
- **Target**: Calidad Alta (≥6) vs Baja (<6)

## Archivos

- `app.py`: Aplicación Flask principal
- `modelo.py`: Lógica del modelo de ML
- `templates/index.html`: Interfaz web
- `winequalityN.csv`: Dataset de vinos
- `requirements.txt`: Dependencias