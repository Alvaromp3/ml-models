# Importamos las librerías necesarias para Flask
from flask import Flask, render_template, request, jsonify
# Importamos nuestras funciones del modelo de machine learning
from modelo import entrenar_modelo, predecir
import os

# Creamos la aplicación Flask
app = Flask(__name__)

# Ruta principal - muestra la página web
@app.route('/')
def index():
    # Renderiza el archivo HTML desde la carpeta templates
    return render_template('index.html')

# Ruta para entrenar el modelo - solo acepta peticiones POST
@app.route('/entrenar', methods=['POST'])
def entrenar():
    try:
        # Llama a la función que entrena el modelo y devuelve la precisión
        accuracy = entrenar_modelo()
        # Devuelve respuesta JSON con éxito y precisión del modelo
        return jsonify({'success': True, 'accuracy': f'{accuracy:.3f}'})
    except Exception as e:
        # Si hay error, devuelve respuesta JSON con el mensaje de error
        return jsonify({'success': False, 'error': str(e)})

# Ruta para hacer predicciones - solo acepta peticiones POST
@app.route('/predecir', methods=['POST']) ## idica que funcion se ejecutara de modelo.py
def hacer_prediccion():
    try:
        # Extraemos todos los datos del formulario HTML y los convertimos a números
        data = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'total sulfur dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol']),
            'type': request.form['type']  # Este es texto, no número
        }
        
        # Llamamos a la función predecir con los datos del usuario
        quality, confidence = predecir(data)
        
        # Si la predicción falló (quality es None), devolvemos error
        if quality is None:
            return jsonify({'success': False, 'error': confidence})
        
        # Si todo salió bien, devolvemos la calidad predicha y la confianza
        return jsonify({'success': True, 'quality': quality, 'confidence': confidence})
    
    except Exception as e:
        # Capturamos cualquier error (ej: datos mal formateados) y lo devolvemos
        return jsonify({'success': False, 'error': str(e)})

# Solo ejecuta la app si este archivo se ejecuta directamente (no si se importa)
if __name__ == '__main__':
    # Inicia el servidor Flask en modo debug (muestra errores detallados)
    app.run(debug=True)