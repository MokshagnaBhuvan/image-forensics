import os
import numpy as np
from PIL import Image
import pandas as pd

# Directories
original_images_dir = r"D:\4k"  # Path to original images
preprocessed_dataset_dir = r"C:\Users\moksh\projects\image forensics\image-forensics\preprocessed datasets"  # Path to save preprocessed images

# Create directory for preprocessed images if it doesn't exist
if not os.path.exists(preprocessed_dataset_dir):
    os.makedirs(preprocessed_dataset_dir)

# List to store preprocessed image paths and class labels (0 for original images)
image_data = []

# Loop through each image in the original images directory
for idx, file in enumerate(os.listdir(original_images_dir)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Support different image formats
        image_path = os.path.join(original_images_dir, file)

        # Open the image and preprocess it
        image = Image.open(image_path)
        image = image.resize((150, 150))  # Resize to 150x150
        image = np.array(image) / 255.0   # Normalize the image

        # Save preprocessed image to the preprocessed dataset folder
        preprocessed_image_name = f"preprocessed_{idx}.png"
        preprocessed_image_path = os.path.join(preprocessed_dataset_dir, preprocessed_image_name)
        Image.fromarray((image * 255).astype(np.uint8)).save(preprocessed_image_path)
        
        # Append the preprocessed image path and its class label (0 for original)
        image_data.append([preprocessed_image_path, 0])

# Convert the image data (file paths and labels) into a DataFrame
df = pd.DataFrame(image_data, columns=['image_path', 'class'])

# Save the DataFrame to an Excel file
excel_file_path = r"D:\DL\image_labels.xlsx"
df.to_excel(excel_file_path, index=False)

# Output message indicating successful completion
print(f"Preprocessing complete. Preprocessed dataset saved to: {preprocessed_dataset_dir}")
print(f"Image paths and class labels saved to: {excel_file_path}")
