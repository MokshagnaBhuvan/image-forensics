import os
import shutil
import numpy as np
import pandas as pd

# Parameters
data_path = r"D:\new\preprocessed_dataset"  # Path to your dataset folder
train_dir = r"D:\new\train_dataset"  # Path to save training images
test_dir = r"D:\new\test_dataset"  # Path to save testing images
train_excel_path = r"D:\new\train_labels.xlsx"  # Path to save the train Excel file
test_excel_path = r"D:\new\test_labels.xlsx"  # Path to save the test Excel file
test_size = 0.2  # Proportion of the dataset to use for testing

# Create directories for train and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load images
image_files = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
np.random.shuffle(image_files)  # Shuffle the dataset

# Split into train and test sets
num_images = len(image_files)
train_size = int((1 - test_size) * num_images)

train_files = image_files[:train_size]
test_files = image_files[train_size:]

# Prepare labels for the Excel sheets
train_labels = [{'filename': file, 'label': 0} for file in train_files]  # 0 for original
test_labels = [{'filename': file, 'label': 1} for file in test_files]    # 1 for original

# Move original images to train and test directories
for file in train_files:
    shutil.copy(os.path.join(data_path, file), os.path.join(train_dir, file))

for file in test_files:
    shutil.copy(os.path.join(data_path, file), os.path.join(test_dir, file))

# Create DataFrames for train and test labels
train_df = pd.DataFrame(train_labels)
test_df = pd.DataFrame(test_labels)

# Save DataFrames to Excel
train_df.to_excel(train_excel_path, index=False)
test_df.to_excel(test_excel_path, index=False)

print(f"Saved {len(train_files)} training images and {len(test_files)} testing images.")
print(f"Train labels saved to {train_excel_path}.")
print(f"Test labels saved to {test_excel_path}.")
