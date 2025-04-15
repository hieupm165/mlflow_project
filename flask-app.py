from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the best model
def load_best_model():
    try:
        # Read best run ID from file
        with open("best_run_id.txt", "r") as f:
            best_run_id = f.read().strip()
        
        print(f"Loading model from run: {best_run_id}")
        
        # Load model directly from file
        model = joblib.load("best_model.pkl")
        # Load scaler
        scaler = joblib.load("scaler.pkl")
        
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, scaler = load_best_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0].tolist()
        return jsonify({
            'prediction': int(prediction),
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        # Read best run ID from file
        with open("best_run_id.txt", "r") as f:
            best_run_id = f.read().strip()
        
        # Get run information from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(best_run_id)
        
        return jsonify({
            'run_id': best_run_id,
            'parameters': run.data.params,
            'metrics': run.data.metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Model Ver1.2</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 800px; margin: 0 auto; }
                .result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; }
                button { padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                input { padding: 8px; margin: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Classification Model Prediction</h1>
                
                <h2>Model Information</h2>
                <button onclick="getModelInfo()">Get Model Info</button>
                <div id="modelInfo" class="result"></div>
                
                <h2>Make a Prediction</h2>
                <p>Enter feature values (comma-separated):</p>
                <input type="text" id="features" placeholder="e.g., 0.1, 0.2, 0.3, ...">
                <button onclick="predict()">Predict</button>
                
                <div id="prediction" class="result"></div>
            </div>
            
            <script>
                async function getModelInfo() {
                    const response = await fetch('/model_info');
                    const data = await response.json();
                    
                    const infoDiv = document.getElementById('modelInfo');
                    infoDiv.innerHTML = '<h3>Run ID: ' + data.run_id + '</h3>';
                    
                    infoDiv.innerHTML += '<h4>Parameters:</h4><ul>';
                    for (const [key, value] of Object.entries(data.parameters)) {
                        infoDiv.innerHTML += '<li>' + key + ': ' + value + '</li>';
                    }
                    infoDiv.innerHTML += '</ul>';
                    
                    infoDiv.innerHTML += '<h4>Metrics:</h4><ul>';
                    for (const [key, value] of Object.entries(data.metrics)) {
                        infoDiv.innerHTML += '<li>' + key + ': ' + value.toFixed(4) + '</li>';
                    }
                    infoDiv.innerHTML += '</ul>';
                }
                
                async function predict() {
                    const featuresInput = document.getElementById('features').value;
                    const features = featuresInput.split(',').map(x => parseFloat(x.trim()));
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({features})
                    });
                    
                    const data = await response.json();
                    
                    const predDiv = document.getElementById('prediction');
                    predDiv.innerHTML = '<h3>Prediction: ' + data.prediction + '</h3>';
                    predDiv.innerHTML += '<h4>Probability:</h4>';
                    predDiv.innerHTML += '<p>Class 0: ' + (data.probability[0] * 100).toFixed(2) + '%</p>';
                    predDiv.innerHTML += '<p>Class 1: ' + (data.probability[1] * 100).toFixed(2) + '%</p>';
                }
            </script>
        </body>
        </html>
        ''')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
