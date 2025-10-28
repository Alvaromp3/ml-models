# 🚗 Predicción de Precios de Coches Usados en España

Este proyecto utiliza machine learning para predecir el precio de coches usados en España basándose en diversas características del vehículo.

## 📋 Descripción

El modelo utiliza un algoritmo Random Forest Regressor para predecir precios de coches usados considerando factores como:

- **Características técnicas**: Edad, kilometraje, potencia, consumo
- **Características físicas**: Número de puertas, color, carrocería
- **Información comercial**: Marca, modelo, número de dueños, estado
- **Ubicación**: Región de venta

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar o descargar el proyecto**
```bash
cd prediccion_precio_coches_espana
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecutar la aplicación web**
```bash
python app.py
```

4. **Abrir en el navegador**
Ve a `http://localhost:5000`

## 📊 Uso de la Aplicación

### 1. Entrenar Modelo
- Haz clic en "🎯 Entrenar Modelo" para entrenar con el dataset incluido
- El sistema mostrará las métricas de rendimiento del modelo

### 2. Realizar Predicciones
- Completa el formulario con las características del coche
- Haz clic en "🔮 Predecir Precio" para obtener la estimación

## 📁 Estructura del Proyecto

```
prediccion_precio_coches_espana/
│
├── app.py                 # Aplicación web con Flask
├── templates/
│   └── index.html        # Interfaz web HTML
├── modelo.py             # Funciones de ML (limpieza, entrenamiento, predicción)
├── requirements.txt      # Dependencias del proyecto
├── README.md            # Documentación
└── coches_espana_usados_1500_edad.csv  # Dataset (a añadir)
```

## 🔧 Características del Modelo

### Variables de Entrada

**Numéricas:**
- Edad (años)
- Kilometraje (km)
- Potencia (CV)
- Consumo (L/100km)
- Número de dueños
- Número de puertas

**Categóricas:**
- Marca
- Modelo
- Combustible
- Transmisión
- Estado
- Región
- Color
- Carrocería

### Algoritmo Utilizado

- **Random Forest Regressor**
  - 300 estimadores
  - Profundidad máxima: 20
  - Mínimo de muestras por hoja: 2

### Preprocesamiento

- **Limpieza de datos**: Eliminación de duplicados y outliers
- **Escalado**: StandardScaler para variables numéricas
- **Codificación**: OneHotEncoder para variables categóricas

## 📈 Métricas de Rendimiento

El modelo proporciona las siguientes métricas:

- **R² Score**: Coeficiente de determinación (precisión del modelo)
- **MAE**: Error Medio Absoluto en euros
- **RMSE**: Error Cuadrático Medio en euros

## 🛠️ Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **Flask**: Framework web ligero
- **Scikit-learn**: Biblioteca de machine learning
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **HTML/CSS/JavaScript**: Interfaz web responsiva

## 📝 Notas Importantes

1. **Dataset incluido**: El archivo CSV con datos de coches ya está incluido en el proyecto
2. **Calidad de datos**: El rendimiento del modelo depende de la calidad y cantidad de datos de entrenamiento
3. **Actualización**: Se recomienda reentrenar el modelo periódicamente con datos actualizados

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Abre un issue para discutir los cambios propuestos
2. Realiza un fork del proyecto
3. Crea una rama para tu feature
4. Realiza tus cambios y pruebas
5. Envía un pull request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o sugerencias sobre el proyecto, puedes abrir un issue en el repositorio.

---

**¡Disfruta prediciendo precios de coches! 🚗💰**