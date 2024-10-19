import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Parameters
original_test_data_path = r"C:\Users\moksh\projects\image forensics\image-forensics\test_dataset"  # Path to your original test dataset folder
adversarial_test_data_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_test" # Path to your adversarial test dataset folder
adversarial_labels_excel_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_labels(test).xlsx"  # Path to labels Excel file for adversarial images

# Load and preprocess images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            img = img / 255.0  # Normalize to [0,1]
            images.append(img)
    return np.array(images)

# Load original and adversarial images for testing
original_test_images = load_images_from_folder(original_test_data_path)
adversarial_test_images = load_images_from_folder(adversarial_test_data_path)

# Load adversarial labels
adversarial_df = pd.read_excel(adversarial_labels_excel_path, engine="openpyxl")
adversarial_labels = adversarial_df['class'].values  # Assuming the class column contains 1s for adversarial images

# Combine test datasets
combined_test_images = np.concatenate((original_test_images, adversarial_test_images), axis=0)
combined_labels = np.concatenate((np.zeros(len(original_test_images)), np.ones(len(adversarial_test_images))), axis=0)  # 0 for original, 1 for adversarial

# Load the trained autoencoder model
model = tf.keras.models.load_model(r"C:\Users\moksh\projects\image forensics\image-forensics\autoencoder_model_robust2.h5")

# Use the autoencoder to reconstruct the test images
reconstructed_images = model.predict(combined_test_images)

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original_images, reconstructed_images):
    return np.mean(np.square(original_images - reconstructed_images), axis=(1, 2, 3))

# Calculate MSE for test images
mse_loss = calculate_mse(combined_test_images, reconstructed_images)

# Set a threshold for correct/incorrect prediction based on MSE
threshold = 0.04  # Tune this threshold based on your dataset and performance

# Determine if each test image was correctly predicted or not
predictions = [0 if mse < threshold else 1 for mse in mse_loss]  # 0 for Correct (original), 1 for Incorrect (adversarial)

# Calculate performance metrics
correct_predictions = np.sum(np.array(predictions) == np.array(combined_labels))
total_images = len(predictions)
accuracy = correct_predictions / total_images * 100

# Calculate precision, recall, and F1 score
precision = precision_score(combined_labels, predictions)
recall = recall_score(combined_labels, predictions)
f1 = f1_score(combined_labels, predictions)

# Calculate AUC (Area Under the Curve)
# To calculate AUC, we need the predicted probabilities, not just the hard predictions.
# Autoencoders typically return reconstruction errors (MSE). For the sake of AUC, we can use these MSE values as a continuous probability score.
# You could normalize the MSE scores to use as the prediction probability.
# If using MSE, lower MSE indicates higher likelihood that the image is original.

# Calculate AUC (treating lower MSE as higher "original" confidence)
auc = roc_auc_score(combined_labels, 1 - (mse_loss / mse_loss.max()))  # Normalize MSE to [0, 1]

# Print results
print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Model Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")

# Generate and display confusion matrix
cm = confusion_matrix(combined_labels, predictions, labels=[0, 1])  # 0 = Original, 1 = Adversarial
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Original', 'Adversarial'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
