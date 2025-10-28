from flask import Flask, render_template, request, jsonify
import pandas as pd
from modelo import entrenar_modelo, predecir
import os

app = Flask(__name__)

DATASET_PATH = os.getenv('CARS_DATASET_PATH', 'coches_espana_usados_1500_edad.csv')

modelo_entrenado = False
metricas = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/entrenar', methods=['POST'])
def entrenar():
    global modelo_entrenado, metricas
    
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({
                'success': False,
                'message': f'Dataset not found: {DATASET_PATH}'
            })
        
        df = pd.read_csv(DATASET_PATH)
        
        r2, mae, rmse = entrenar_modelo(df)
        
        modelo_entrenado = True
        metricas = {
            'r2': round(r2, 3),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2)
        }
        
        return jsonify({
            'success': True,
            'message': 'Modelo entrenado exitosamente',
            'metricas': metricas
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error al entrenar: {str(e)}'
        })

@app.route('/predecir', methods=['POST'])
def hacer_prediccion():
    if not modelo_entrenado:
        return jsonify({
            'success': False,
            'message': 'Debe entrenar el modelo primero'
        })
    
    try:
        data = request.json
        
        resultado = predecir(
            data['edad'], data['kilometraje_km'], data['potencia_cv'],
            data['consumo_l_100km'], data['num_duenos'], data['puertas'],
            data['marca'], data['modelo'], data['combustible'],
            data['transmision'], data['estado'], data['region'],
            data['color'], data['carroceria']
        )
        
        if isinstance(resultado, str):
            return jsonify({'success': False, 'message': resultado})
        
        return jsonify({
            'success': True,
            'precio': resultado
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error en predicción: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)