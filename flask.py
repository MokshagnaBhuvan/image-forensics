from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the pre-trained autoencoder model
model = load_model("autoencoder_model_robust.h5")

# Image preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)  # Resize image
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define a function to compute reconstruction loss (to identify adversarial images)
def compute_reconstruction_loss(original, reconstructed):
    # Compute Mean Squared Error (MSE) between the original and reconstructed images
    return np.mean(np.square(original - reconstructed))

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    # Read the image from the request
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    
    # Predict the reconstructed image
    reconstructed_img = model.predict(preprocessed_img)
    
    # Calculate reconstruction loss
    loss = compute_reconstruction_loss(preprocessed_img, reconstructed_img)
    
    # Define a threshold for detecting adversarial images (tune this threshold as needed)
    threshold = 0.01  # Adjust this threshold based on model performance
    is_adversarial = loss > threshold
    confidence = min(100, (loss * 100))  # Confidence score (percentage)
    
    # Respond with the result
    return jsonify({
        "is_adversarial": is_adversarial,
        "confidence": round(confidence, 2)  # Return confidence as percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
