import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

from preprocessing import preprocess_data
from randomforest import train_and_save_model, load_and_predict_bulk
from isolationforest import train_isolation_forest, detect_anomalies
from combined_model import train_combined_model, evaluate_combined_model
from autoencoder_backend import train_autoencoder, predict_autoencoder

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        logging.info('Received request for /preprocess')
        result = preprocess_data()
        logging.info('Preprocessing completed successfully')
        logging.debug(f'Preprocessing result: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error in /preprocess: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        model_name = data.get('name', 'rf_model')
        logging.info(f'Received data for Random Forest training: {data}')
        result = train_and_save_model(model_name)
        logging.info(f'Random Forest training complete: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error in /train/randomforest: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        logging.info('Training Isolation Forest...')
        result = train_isolation_forest()
        logging.info(f'Isolation Forest training complete: {result}')
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error in /train/isolationforest: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/predict/randomforest/all', methods=['GET'])
def predict_rf_all():
    try:
        model_path = request.args.get('model', 'rf_model.pkl')
        logging.info(f"Loading Random Forest model from {model_path}")
        result = load_and_predict_bulk(model_path=model_path)
        logging.debug("Random Forest predictions done.")
        return jsonify(result)
    except Exception as e:
        logging.exception("Error in /predict/randomforest/all")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/isolationforest/all', methods=['GET'])
def predict_iso_all():
    try:
        logging.info("Predicting with Isolation Forest...")
        result = detect_anomalies()
        logging.debug("Isolation Forest anomaly detection done.")
        return jsonify(result)
    except Exception as e:
        logging.exception("Error in /predict/isolationforest/all")
        return jsonify({'error': str(e)}), 500


@app.route('/train/combined', methods=['POST'])
def train_combined():
    try:
        data = request.get_json(force=True)
        model_name = data.get('name', 'rf_model.pkl')

        logging.info(f'[TRAIN COMBINED] Training started with parameters: {data}')
        
        result = train_combined_model(model_name)

        logging.info(f'[TRAIN COMBINED] Training completed successfully: {result}')
        return jsonify({'status': 'success', 'details': result}), 200
    except Exception as e:
        logging.error(f'[TRAIN COMBINED] Training failed: {e}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/evaluate/combined', methods=['GET'])
def evaluate_combined():
    try:
        logging.info('[EVALUATE COMBINED] Evaluation started...')
        
        result = evaluate_combined_model()

        logging.info(f'[EVALUATE COMBINED] Evaluation completed successfully: {result}')
        return jsonify({'status': 'success', 'metrics': result}), 200
    except Exception as e:
        logging.error(f'[EVALUATE COMBINED] Evaluation failed: {e}', exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route("/train/autoencoder", methods=["POST"])
def train_autoencoder_route():
    try:
        logging.info("Training Autoencoder...")
        result = train_autoencoder()
        logging.info("Autoencoder training successful.")
        return jsonify({"message": "Autoencoder trained successfully", **result})
    except Exception as e:
        logging.error(f"Error training Autoencoder: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/autoencoder/all", methods=["GET"])
def predict_autoencoder_route():
    try:
        logging.info("Predicting with Autoencoder...")
        df, predictions, mse_threshold = predict_autoencoder()

        result_df = df.copy()
        result_df['Autoencoder_Anomaly'] = predictions
        result_df['Anomaly_Label'] = result_df['Autoencoder_Anomaly'].map({0: 'Normal', 1: 'Anomaly (Possible Fraud)'})
        result_df['Actual_Label'] = result_df['Class'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})

        anomaly_count = sum(predictions)
        total = len(predictions)
        stats = {
            'total': total,
            'anomalies_detected': int(anomaly_count),
            'normal': int(total - anomaly_count),
            'anomaly_rate': round((anomaly_count / total) * 100, 2),
            'mse_threshold': mse_threshold
        }

        logging.info("Autoencoder prediction successful.")
        return jsonify({
            'predictions': result_df.head(100).to_dict(orient='records'),
            'stats': stats
        })
    except Exception as e:
        logging.error(f"Autoencoder prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
