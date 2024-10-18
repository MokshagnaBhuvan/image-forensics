import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained Autoencoder model
model = tf.keras.models.load_model(r"D:\new\autoencoder_model.h5")

# Directories for original and adversarial images
original_dir = r"D:\new\test_dataset"
adversarial_dir = r"D:\new\adversarial_test"

# Load original image paths from Excel file
original_labels_excel_path = r"D:\new\original_labels.xlsx"
original_df = pd.read_excel(original_labels_excel_path)

# Load adversarial image paths from Excel file
adversarial_labels_excel_path = r"D:\new\adversarial_labels.xlsx"
adversarial_df = pd.read_excel(adversarial_labels_excel_path)

# Function to load and preprocess images
def load_image(image_path, image_size=(128, 128)):
    image = np.array(Image.open(image_path).resize(image_size)) / 255.0  # Normalize to [0, 1]
    return image

# Prepare original and adversarial images for testing
original_images = []
for _, row in original_df.iterrows():
    image_path = row['image_path']
    original_images.append(load_image(image_path))

original_images = np.array(original_images)

adversarial_images = []
for _, row in adversarial_df.iterrows():
    image_path = row['image_path']
    adversarial_images.append(load_image(image_path))

adversarial_images = np.array(adversarial_images)

# Combine original and adversarial images
combined_images = np.concatenate((original_images, adversarial_images), axis=0)
combined_labels = [0] * len(original_images) + [1] * len(adversarial_images)  # 0 for original, 1 for adversarial

# Use the autoencoder to reconstruct the combined images
reconstructed_images = model.predict(combined_images)

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_images, reconstructed_images):
    return np.mean(np.square(original_images - reconstructed_images), axis=(1, 2, 3))

# Calculate MSE for combined images
mse_loss = calculate_mse(combined_images, reconstructed_images)

# Set a threshold for correct/incorrect prediction based on MSE
threshold = 0.04  # Example threshold value (you might want to tune this based on your dataset)

# Determine if each image was correctly predicted or not
predictions = [0 if mse < threshold else 1 for mse in mse_loss]  # 0 for Correct, 1 for Incorrect

# Calculate accuracy
correct_predictions = np.sum(np.array(predictions) == np.array(combined_labels))
total_images = len(predictions)
accuracy = correct_predictions / total_images * 100

# Display results
for i, prediction in enumerate(predictions):
    print(f"Image {i+1}: MSE = {mse_loss[i]:.4f} - Prediction: {'Correct' if prediction == 0 else 'Incorrect'}")

# Print the overall accuracy
print(f"\nTotal Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Model Accuracy on Combined Images: {accuracy:.2f}%")

# Generate and display confusion matrix
cm = confusion_matrix(combined_labels, predictions, labels=[0, 1])  # 0 = Original, 1 = Adversarial
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Original', 'Adversarial'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
