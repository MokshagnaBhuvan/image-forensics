import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd

# Parameters
image_size = (128, 128)  # Image size for resizing
batch_size = 32  # Batch size for training
epochs = 50  # Number of epochs for training
original_data_path = r"C:\Users\moksh\projects\image forensics\image-forensics\train_dataset"  # Path to your original dataset folder
adversarial_data_path = r"C:\Users\moksh\projects\image forensics\image-forensics\adversarial_train"  # Path to your adversarial dataset folder

# Load and preprocess images from a folder
def load_images_from_folder(folder, image_size):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize to 128x128
            img = img / 255.0  # Normalize to [0,1]
            images.append(img)
    return np.array(images)

# Load original and adversarial images
original_images = load_images_from_folder(original_data_path, image_size)
adversarial_images = load_images_from_folder(adversarial_data_path, image_size)

# Combine original and adversarial images for training
combined_images = np.concatenate((original_images, adversarial_images), axis=0)

# Split the data into train and validation sets
train_size = int(0.8 * len(combined_images))
train_images = combined_images[:train_size]
val_images = combined_images[train_size:]

# Build 2D Convolutional Autoencoder
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)  # Dropout layer
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)  # Dropout layer
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)  # Dropout layer
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Build and train the model
autoencoder = build_autoencoder(input_shape=(128, 128, 3))

# Data augmentation
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=0.2, rotation_range=20)

# Learning rate reduction
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# Train the autoencoder
autoencoder.fit(
    datagen.flow(train_images, train_images, batch_size=batch_size),
    epochs=epochs,
    validation_data=(val_images, val_images),
    callbacks=[lr_reduction]
)

# Save the model
autoencoder.save("autoencoder_model_robust1.h5")
