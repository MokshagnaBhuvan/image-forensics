import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained Autoencoder model
model = tf.keras.models.load_model(r"C:\Users\moksh\projects\image forensics\image-forensics\autoencoder_model_full.h5")

# Directories for original and adversarial images
original_dir = r"C:\Users\moksh\projects\image forensics\image-forensics\test_dataset"  # Directory containing original images
adversarial_dir = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_test"  # Directory containing adversarial images

# Load original image paths from Excel file
original_labels_excel_path = r"C:\Users\moksh\projects\image forensics\image-forensics\test_labels.xlsx"
original_df = pd.read_excel(original_labels_excel_path)

# Load adversarial image paths from Excel file
adversarial_labels_excel_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_labels(test).xlsx"
adversarial_df = pd.read_excel(adversarial_labels_excel_path)

# Function to load and preprocess images
def load_image(image_dir, image_name, image_size=(128, 128)):
    image_path = os.path.join(image_dir, image_name)  # Construct full image path
    try:
        # Try loading the image and resizing it
        image = np.array(Image.open(image_path).resize(image_size)) / 255.0  # Normalize to [0, 1]
        return image
    except FileNotFoundError as e:
        # If image is not found, print the error and return None
        print(f"File not found: {image_path}")
        return None

# Prepare original and adversarial images for testing
original_images = []
for _, row in original_df.iterrows():
    image_name = row['image_path']  # Assuming 'image_path' column only contains image name
    image = load_image(original_dir, image_name)  # Prepend directory path
    if image is not None:
        original_images.append(image)

original_images = np.array(original_images)

adversarial_images = []
for _, row in adversarial_df.iterrows():
    image_name = row['image_path']  # Assuming 'image_path' column only contains image name
    image = load_image(adversarial_dir, image_name)  # Prepend directory path
    if image is not None:
        adversarial_images.append(image)

adversarial_images = np.array(adversarial_images)

# Check if there are any images loaded
if len(original_images) == 0 or len(adversarial_images) == 0:
    raise Exception("No images loaded. Please check the image paths and availability.")

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
