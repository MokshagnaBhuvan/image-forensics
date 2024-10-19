import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set parameters
IMAGE_SHAPE = (28, 28, 1)  # Adjust as needed
NOISE_DIM = 100
EPOCHS = 100
BATCH_SIZE = 128
SAVE_MODEL_PATH = 'gan_model.h5'

# Load original images from the specified directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))  # Resize images to 28x28
        img = img_to_array(img) / 255.0  # Normalize images
        images.append(img)
    return np.array(images)

# Load labels from Excel file
def load_labels_from_excel(file_path):
    labels_df = pd.read_excel(file_path)
    return labels_df.values.flatten()  # Flatten to get a 1D array of labels

# Load data
train_dataset_path = r"C:\Users\moksh\projects\image forensics\image-forensics\train_dataset"
x_train = load_images_from_folder(train_dataset_path)

# Load labels (if needed for further training)
labels_path = r"C:\Users\moksh\projects\image forensics\image-forensics\train_labels.xlsx"
labels = load_labels_from_excel(labels_path)

# Build the Generator
def build_generator():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 128, activation='relu', input_dim=NOISE_DIM))
    model.add(layers.Reshape((7, 7, 128)))  # Start with 7x7
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', activation='relu'))  # Up to 14x14
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='relu'))  # Up to 28x28
    model.add(layers.Conv2DTranspose(1, kernel_size=5, padding='same', activation='sigmoid'))  # Final 28x28 output
    return model


# Build the Discriminator
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu', input_shape=IMAGE_SHAPE))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create GAN model
z = layers.Input(shape=(NOISE_DIM,))
img = generator(z)
discriminator.trainable = False
validity = discriminator(img)
gan = models.Model(z, validity)

# Compile GAN
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Train the GAN
for epoch in range(EPOCHS):
    # Generate fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    fake_images = generator.predict(noise)

    # Select a random batch of real images
    idx = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
    real_images = x_train[idx]

    # Labels for real and fake images
    real_labels = np.ones((BATCH_SIZE, 1))
    fake_labels = np.zeros((BATCH_SIZE, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_DIM))
    g_loss = gan.train_on_batch(noise, real_labels)

    # Print losses and save model at intervals
    if epoch % 1000 == 0:
        print(f'Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}')
        gan.save(SAVE_MODEL_PATH)

# Save the models
generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')
