import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained Autoencoder model
model = tf.keras.models.load_model(r"D:\new\autoencoder_model.h5")

# Directory for adversarial images
adversarial_dir = r"D:\new\adversarial_test"

# Load adversarial image paths from Excel file
adversarial_labels_excel_path = r"D:\new\adversarial_labels.xlsx"
adversarial_df = pd.read_excel(adversarial_labels_excel_path)

# Function to load and preprocess images
def load_image(image_path, image_size=(128, 128)):
    image = np.array(Image.open(image_path).resize(image_size)) / 255.0  # Normalize to [0, 1]
    return image

# Prepare adversarial images for testing
adversarial_images = []
for _, row in adversarial_df.iterrows():
    image_path = row['image_path']
    adversarial_images.append(load_image(image_path))

adversarial_images = np.array(adversarial_images)

# Use the autoencoder to reconstruct the adversarial images
reconstructed_images = model.predict(adversarial_images)

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_images, reconstructed_images):
    return np.mean(np.square(original_images - reconstructed_images), axis=(1, 2, 3))

# Calculate MSE for adversarial images
mse_loss = calculate_mse(adversarial_images, reconstructed_images)

# Set a threshold for correct/incorrect prediction based on MSE
threshold = 0.04  # Example threshold value (tune this based on your dataset and performance)

# Determine if each adversarial image was correctly predicted or not
predictions = [0 if mse < threshold else 1 for mse in mse_loss]  # 0 for Correct, 1 for Incorrect
true_labels = [row['class'] for _, row in adversarial_df.iterrows()]  # Assuming class 1 for adversarial images

# Calculate accuracy
correct_predictions = np.sum(np.array(predictions) == np.array(true_labels))
total_images = len(predictions)
accuracy = correct_predictions / total_images * 100

# Display results
for i, prediction in enumerate(predictions):
    print(f"Adversarial Image {i+1}: MSE = {mse_loss[i]:.4f} - Prediction: {'Correct' if prediction == 0 else 'Incorrect'}")

# Print the overall accuracy
print(f"\nTotal Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Model Accuracy on Adversarial Images: {accuracy:.2f}%")

# Generate and display confusion matrix
cm = confusion_matrix(true_labels, predictions, labels=[0, 1])  # 0 = Correct, 1 = Incorrect
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Correct', 'Incorrect'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
