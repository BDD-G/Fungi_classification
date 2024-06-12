import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from pathlib import Path
import cv2
from skimage.feature import local_binary_pattern
from os.path import join
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from IPython.display import clear_output

img_path = '/work3/s220243/Thesis'
base_path = '/zhome/ac/d/174101/thesis'

df = pd.read_excel(f"{base_path}/data/imageAnalysis_information.xlsx")
df  = pd.DataFrame(df.values[1:], columns=df.iloc[0])
df.head()

# Strip the blank spaces from names
df['species'] = df['species'].str.strip()
df['genus'] = df['genus'].str.strip()

species_genus_df = pd.DataFrame({"IBT_number": df['IBT number'],
                          "Target" : df["genus"]+"-"+df["species"]})
species_genus_df.dropna()

ibt_numbers=[]
img_paths=[]

# Get a list of all the images paths
img = Path(f"{img_path}/images")
paths = list(img.glob('**/*.jpeg'))

# Create list of all IBTs and paths to images for the given IBT
for path in paths:
  match = re.search(r'IBT \d+',str(path))
  if match:
    ibt_numbers.append(match.group())
    img_paths.append(str(path))
  else:
    ibt_numbers.append("ACU1")
    img_paths.append(str(path))

# Create DF from a lists
paths_df = pd.DataFrame({"IBT_number": ibt_numbers,
                     "path": img_paths})

# Merge target_df and paths_df
paths_df = paths_df.groupby('IBT_number')['path'].apply(list).reset_index()
merged_df = pd.merge(species_genus_df, paths_df, on='IBT_number', how='inner')
target_paths_df = merged_df.explode('path')
target_paths_df

# Drop all images that are prior to day 2
target_paths_df['Image_number'] = target_paths_df['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
target_paths_df['Image_number'] = target_paths_df['Image_number'].str.replace(r'\D', '', regex=True)
target_paths_df['Image_number'] = target_paths_df['Image_number'].astype(int)
target_df = target_paths_df[(target_paths_df['Image_number'] >= 48) & (target_paths_df['Image_number'] <= 168)]
val_df = target_paths_df[(target_paths_df['Image_number'] > 168) & (target_paths_df['Image_number'] <= 192)]

def apply_gradient(image_path, output_directory, scale_factor=2, apply_threshold=False, threshold_value=100):
    # Ensure the image is in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute the Sobel gradient in the X direction
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    
    # Compute the Sobel gradient in the Y direction
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the combined gradient magnitude
    combined_grad = np.sqrt(sobelx**2 + sobely**2)
    
    # Scale the gradients to enhance visibility
    combined_grad *= scale_factor

    # Resize
    resized_image = resize(combined_grad, (224,224), anti_aliasing=True)
    
    # Normalize the combined gradient to the range [0, 255]
    resized_image = np.uint8(np.clip(resized_image, 0, 255))
    
    # Optionally apply a threshold to enhance edges
    if apply_threshold:
        _, resized_image = cv2.threshold(resized_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Get the file name
    file_name = image_path.split('/')[-1]
    
    # Save the processed image
    output_path = join(output_directory, file_name)
    cv2.imwrite(output_path, resized_image)


base = Path(f"{img_path}/data_split_sobel")
base.mkdir(exist_ok=True)

# Create train-test split folders
train_dst = base / "train"
test_dst = base / "test"
val_dst = base / "validation"

shutil.rmtree(train_dst)
shutil.rmtree(test_dst)
shutil.rmtree(val_dst)
print('Directories removed')

train_dst.mkdir(exist_ok=True)
test_dst.mkdir(exist_ok=True)
val_dst.mkdir(exist_ok=True)

# Iterate through each row in the DataFrame
def copy_sobel(df, dst_folder):
    counter = 0
    for index, row in df.iterrows():
        image_path = row['path']

        target_dst = dst_folder / str(row['Target'])
        target_dst.mkdir(exist_ok=True)

        # Apply LBP and save the processed image
        apply_gradient(image_path, target_dst, scale_factor=2, apply_threshold=False, threshold_value=100)
        # Progress counter
        counter += 1
        print(f'Image number: {counter} out of {len(df['Image_number'])}')

train_df, test_df = train_test_split(target_df, test_size=0.3, random_state=42)
print('copying...')

# Copy images to train directory
copy_sobel(train_df, train_dst)
print('train images copy finished')

# Copy images to test directory
copy_sobel(test_df, test_dst)
print('test images copy finished')

# Copy images to validation directory
copy_sobel(val_df, val_dst)
print('validation images copy finished')