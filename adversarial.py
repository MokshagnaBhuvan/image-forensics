import tensorflow as tf
import numpy as np
import os
import pandas as pd
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model(r"C:\Users\moksh\projects\image forensics\image-forensics\autoencoder_model.h5")

# Directory for test images
test_dir = r"C:\Users\moksh\projects\image forensics\image-forensics\test_dataset"

# Directory for adversarial images
adversarial_dir = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_test"
os.makedirs(adversarial_dir, exist_ok=True)

# Excel file path to save adversarial image paths and labels
adversarial_labels_excel_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_labels(test).xlsx"

# FGSM function to generate adversarial examples
def generate_adversarial_image(model, image, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        
        # Here, assuming a binary classification model, we target the opposite class
        target = tf.zeros_like(prediction)  # Targeting class 0 for the adversarial attack
        
        # If your model has more classes, adjust this target based on your needs
        loss = tf.keras.losses.binary_crossentropy(target, prediction, from_logits=False)
    
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_image = image_tensor + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0.0, 1.0)  # Keep pixel values in [0, 1]
    return adversarial_image.numpy()[0]  # Remove batch dimension

# Prepare a list to hold image paths and labels
adversarial_data = []

# Generate adversarial images from test set
for file in os.listdir(test_dir):
    if file.endswith('.png'):
        image_path = os.path.join(test_dir, file)
        
        # Resize the image to the size the model expects (128x128)
        image = np.array(Image.open(image_path).resize((128, 128))) / 255.0
        
        # Generate adversarial image
        adversarial_image = generate_adversarial_image(model, image)
        
        # Save the adversarial image
        save_path = os.path.join(adversarial_dir, f"adv_{file}")
        Image.fromarray((adversarial_image * 255).astype(np.uint8)).save(save_path)
        
        # Append image path and label to the list
        adversarial_data.append({'image_path': save_path, 'class': 1})

# Create a DataFrame from the list
adversarial_df = pd.DataFrame(adversarial_data)

# Save the DataFrame to an Excel file
adversarial_df.to_excel(adversarial_labels_excel_path, index=False)

print("Adversarial images generated, saved, and their labels recorded in Excel.")
