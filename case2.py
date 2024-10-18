import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
adversarial_dir = r"D:\new\adversarial_test_images"  # Path to adversarial images
adversarial_excel_path = r"D:\new\adversarial_labels.xlsx"  # Path to the Excel file with labels
model_path = "autoencoder_model.h5"  # Path to your trained model

# Load the pre-trained model
model = load_model(model_path)

# Load the adversarial labels
adversarial_labels_df = pd.read_excel(adversarial_excel_path)
adversarial_labels = adversarial_labels_df['label'].values

# Load and preprocess adversarial images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to the model input size
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

# Load adversarial images
adversarial_images = load_images_from_folder(adversarial_dir)

# Predict using the model
predictions = model.predict(adversarial_images)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Calculate metrics using true labels from the Excel sheet
accuracy = accuracy_score(adversarial_labels, predicted_classes)
precision = precision_score(adversarial_labels, predicted_classes)
recall = recall_score(adversarial_labels, predicted_classes)
f1 = f1_score(adversarial_labels, predicted_classes)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
cm = confusion_matrix(adversarial_labels, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
