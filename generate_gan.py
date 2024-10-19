import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained generator model
generator = load_model('generator_model.h5')

NOISE_DIM = 100  # or whatever value you used in your GAN training script

# Load original test images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))  # Adjust target size as needed
        img = img_to_array(img) / 255.0  # Normalize images
        images.append(img)
    return np.array(images)

# Load original test images
original_test_path = r"C:\Users\moksh\projects\image forensics\image-forensics\test_dataset"
original_images = load_images_from_folder(original_test_path)

# Generate adversarial images using the GAN
num_images = original_images.shape[0]  # Get the number of original images
noise = np.random.normal(0, 1, (num_images, NOISE_DIM))  # Generate noise
generated_images = generator.predict(noise)

# Rescale generated images to save them correctly
generated_images = (generated_images * 255).astype(np.uint8)

# Specify the directory to save the generated images
output_folder = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_test_gan"

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save the generated adversarial images
for i in range(num_images):
    img_path = os.path.join(output_folder, f'adversarial_image_{i}.png')
    tf.keras.preprocessing.image.save_img(img_path, generated_images[i])

print(f'Generated adversarial images saved to {output_folder}')
