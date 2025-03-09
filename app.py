from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("house_price_model.pkl")  # Ensure this file exists

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "House Price Prediction API is running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        features = np.array([data['features']])  # Expecting {"features": [val1, val2, val3, ...]}

        # Make prediction
        prediction = model.predict(features)

        # Return result as JSON
        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
