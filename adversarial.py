import os
import shutil
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Parameters
test_dir = r"D:\new\test_dataset"  # Path to the test images
adversarial_dir = r"D:\new\adversarial_test_images"  # Path to save adversarial images
adversarial_excel_path = r"D:\new\adversarial_labels.xlsx"  # Path to save the adversarial Excel file

# Create directory for adversarial images
os.makedirs(adversarial_dir, exist_ok=True)

# Load your pre-trained model for adversarial image generation
model = load_model("autoencoder_model.h5")  # Change this to your classification model

# Load and preprocess test images for adversarial generation
def load_images_from_folder(folder, image_size):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize to the model input size
            img = img / 255.0  # Normalize to [0,1]
            images.append((img, filename))
    return images

# Generate adversarial images using FGSM
def generate_adversarial_images(model, images, epsilon=0.01):
    adversarial_images = []
    for img, filename in images:
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        with tf.GradientTape() as tape:
            tape.watch(img)
            prediction = model(img)
            loss = tf.keras.losses.categorical_crossentropy(tf.keras.utils.to_categorical(np.argmax(prediction, axis=1), num_classes=10), prediction)  # Adjust num_classes
        gradient = tape.gradient(loss, img)
        perturbation = epsilon * tf.sign(gradient)
        adversarial_img = tf.clip_by_value(img + perturbation, 0, 1)  # Ensure values are still in [0, 1]
        adversarial_images.append((adversarial_img[0], filename))  # Remove batch dimension
    return adversarial_images

# Load test images for adversarial generation
test_images = load_images_from_folder(test_dir, (128, 128))

# Generate adversarial images
adversarial_images = generate_adversarial_images(model, test_images)

# Save adversarial images and prepare labels for the Excel sheet
adversarial_labels = []
for adv_img, filename in adversarial_images:
    adv_img = (adv_img * 255).astype(np.uint8)  # Convert back to [0, 255]
    adv_filename = f"adversarial_{filename}"
    cv2.imwrite(os.path.join(adversarial_dir, adv_filename), adv_img)
    adversarial_labels.append({'filename': adv_filename, 'label': 1})  # 1 for adversarial

# Create DataFrame and save to Excel
adversarial_df = pd.DataFrame(adversarial_labels)
adversarial_df.to_excel(adversarial_excel_path, index=False)

print(f"Adversarial images generated and saved to {adversarial_dir}.")
print(f"Adversarial labels saved to {adversarial_excel_path}.")
