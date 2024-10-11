from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

# Configurar el logging
logging.basicConfig(
    filename='application.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Cargar el modelo entrenado
pipeline = joblib.load('modelo_ods.pkl')

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para predicciones desde la aplicación web
@app.route('/predict_web', methods=['POST'])
def predict_web():
    try:
        logging.info("Solicitud recibida en /predict_web")
        data = request.get_json(force=True)
        instances = data['instances']
        predictions = pipeline.predict(instances)
        probabilities = pipeline.predict_proba(instances).max(axis=1)
        results = [{'prediction': str(pred), 'probability': float(prob)} for pred, prob in zip(predictions, probabilities)]
        logging.info(f"Predicciones realizadas: {results}")
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error en /predict_web: {e}")
        return jsonify({'error': 'Ocurrió un error al procesar la predicción.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
