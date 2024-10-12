from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import classification_report
from werkzeug.utils import secure_filename
import os

# Configurar el logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handlers
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formato para los logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Agregar los handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configuración para la carga de archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta de carga
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cargar el modelo y los datos
model_data = joblib.load('model_and_data.pkl')
pipeline = model_data['pipeline']
X_train = model_data['X_train']
y_train = model_data['y_train']

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para predicciones desde la aplicación web
@app.route('/predict_web', methods=['POST'])
def predict_web():
    try:
        logger.info("Solicitud recibida en /predict_web")
        data = request.get_json(force=True)
        instances = data['instances']
        
        # Cargar el modelo actualizado
        model_data = joblib.load('model_and_data.pkl')
        pipeline = model_data['pipeline']
        
        predictions = pipeline.predict(instances)
        probabilities = pipeline.predict_proba(instances).max(axis=1)
        results = [{'prediction': str(pred), 'probability': float(prob)} for pred, prob in zip(predictions, probabilities)]
        logger.info(f"Predicciones realizadas: {results}")
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error en /predict_web: {e}")
        return jsonify({'error': 'Ocurrió un error al procesar la predicción.'}), 500

# Endpoint para reentrenamiento con archivo CSV
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        logger.info("Solicitud recibida en /retrain")
        if 'csv_file' not in request.files:
            logger.error("No se encontró el archivo en la solicitud.")
            return jsonify({'error': 'No se encontró el archivo en la solicitud.'}), 400
        file = request.files['csv_file']
        if file.filename == '':
            logger.error("No se seleccionó ningún archivo.")
            return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Archivo recibido y guardado en {filepath}")
            
            new_data = pd.read_csv(filepath)
            
            if 'Textos_espanol' not in new_data.columns or 'sdg' not in new_data.columns:
                logger.error("El archivo CSV no tiene las columnas 'Textos_espanol' y 'sdg'.")
                return jsonify({'error': "El archivo CSV debe contener las columnas 'Textos_espanol' y 'sdg'."}), 400
            
            # Extraer las instancias y etiquetas
            X_new = new_data['Textos_espanol']
            y_new = new_data['sdg']
            
            # Combinar datos originales y nuevos
            X_combined = pd.concat([X_train, X_new], ignore_index=True)
            y_combined = pd.concat([y_train, y_new], ignore_index=True)
            
            # Reentrenar el pipeline
            pipeline.fit(X_combined, y_combined)
            
            # Actualizar el modelo y los datos en el diccionario
            model_data['pipeline'] = pipeline
            model_data['X_train'] = X_combined
            model_data['y_train'] = y_combined
            
            # Guardar el modelo y los datos actualizados
            joblib.dump(model_data, 'model_and_data.pkl')
            
            # Calcular métricas de rendimiento en los nuevos datos
            predictions = pipeline.predict(X_new)
            report = classification_report(y_new, predictions, output_dict=True)
            logger.info(f"Modelo reentrenado. Métricas: {report}")
            os.remove(filepath)
            
            return jsonify(report)
        else:
            logger.error("Tipo de archivo no permitido.")
            return jsonify({'error': 'Tipo de archivo no permitido. Solo se permiten archivos CSV.'}), 400
        
    except Exception as e:
        logger.error(f"Error en /retrain: {e}")
        return jsonify({'error': 'Ocurrió un error durante el reentrenamiento.'}), 500

if __name__ == '__main__':
    app.run(debug=True)