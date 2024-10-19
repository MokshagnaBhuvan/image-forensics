import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained discriminator model
discriminator = load_model('discriminator_model.h5')

# Load images from a specified folder
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

# Load GAN-generated adversarial images
generated_images_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_test_gan"
generated_images = load_images_from_folder(generated_images_path)

# Combine real and fake images
x_test = np.concatenate((original_images, generated_images), axis=0)

# Create labels for the images
real_labels = np.ones((original_images.shape[0], 1))  # Label for real images
fake_labels = np.zeros((generated_images.shape[0], 1))  # Label for fake images
y_test = np.concatenate((real_labels, fake_labels), axis=0)

# Evaluate the discriminator on the test set
loss, accuracy = discriminator.evaluate(x_test, y_test, verbose=1)

print(f'Discriminator Loss: {loss}')
print(f'Discriminator Accuracy: {accuracy}')
