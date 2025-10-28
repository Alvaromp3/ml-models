from flask import Flask, render_template, request, jsonify
from modelo_airbnb import ModeloAirbnb
import pandas as pd

app = Flask(__name__)

# Inicializar y entrenar el modelo
modelo = ModeloAirbnb()

# Intentar cargar modelo existente, si no existe, entrenar uno nuevo
if not modelo.cargar_modelo():
    print("Entrenando nuevo modelo...")
    r2 = modelo.entrenar_modelo()
    modelo.guardar_modelo()
    print(f"Modelo entrenado con R2: {r2}")
else:
    print("Modelo cargado exitosamente")

# Obtener valores únicos para los dropdowns
df = pd.read_csv('airbnb_synthetic.csv')
property_types = sorted(df['property_type'].unique())
room_types = sorted(df['room_type'].unique())

@app.route('/')
def index():
    return render_template('index.html', 
                         property_types=property_types, 
                         room_types=room_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        datos = {
            'latitude': float(request.form['latitude']),
            'longitude': float(request.form['longitude']),
            'property_type': request.form['property_type'],
            'room_type': request.form['room_type'],
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'number_of_reviews': int(request.form['number_of_reviews']),
            'availability_365': int(request.form['availability_365'])
        }
        
        prediccion = modelo.predecir(datos)
        
        return jsonify({
            'success': True,
            'prediccion': float(round(prediccion, 2))
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(debug=True, port=port)