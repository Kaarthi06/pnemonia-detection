from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import numpy as np
import cv2
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# Load trained model
model = tf.keras.models.load_model("models/pneumonia_detection_model.keras")

def preprocess_image(image):
    """Preprocess uploaded image for model prediction."""
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (150, 150))  # Resize to match model input size
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    return image

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to handle image upload and return prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    img = preprocess_image(file)
    prediction = model.predict(img)

    result = "Pneumonia" if prediction > 0.5 else "Normal"
    return jsonify({"prediction": result})

# Run Flask app locally
if __name__ == '__main__':
    app.run(debug=True)
