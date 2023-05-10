#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:32:56 2023

@author: harryzhu
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import graycomatrix
import pickle
from PIL import Image


def high_pass_filter(image_path):
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is 200x200
    if image.shape != (200, 200):
        raise ValueError("The input image must have a size of 200x200.")

    # Create a Gaussian low-pass filter
    low_pass_filter = cv2.GaussianBlur(image, (3, 3), 0)

    # Subtract the low-pass filtered image from the original image
    high_pass_filtered_image = cv2.subtract(image, low_pass_filter)

    # Normalize the pixel values to the range [0, 255]
    normalized_image = cv2.normalize(high_pass_filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return normalized_image

def unsharp_mask(image_path, sigma=1.5, amount=1.5): #don't use
    # Read the image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is 200x200
    if image.shape != (200, 200):
        raise ValueError("The input image must have a size of 200x200.")

    # Create a Gaussian low-pass filter
    low_pass_filter = cv2.GaussianBlur(image, (0, 0), sigma)

    # Calculate the high-pass filtered image by subtracting the low-pass filtered image from the original image
    high_pass_filtered_image = cv2.subtract(image, low_pass_filter)

    # Apply the unsharp mask by adding a weighted high-pass filtered image to the original image
    sharpened_image = cv2.addWeighted(image, 1 + amount, high_pass_filtered_image, amount, 0)

    return sharpened_image

#a function that let you apply a specific preprocessing image to a folder 
#that contains test and train image in separate folders
def apply_preprocessing_to_subfolders(preprocess_function, src_folder, dst_folder=None):
    
    # List all subfolders in the source folder
    subfolders = ['test', 'train']

    # Create the destination folder if not provided
    if dst_folder is None:
        dst_folder = os.path.join(src_folder, preprocess_function.__name__)
    
    os.makedirs(dst_folder, exist_ok=True)

    # Iterate through the subfolders and apply the preprocessing function to each image
    for subfolder in subfolders:
        src_subfolder = os.path.join(src_folder, subfolder)
        dst_subfolder = os.path.join(dst_folder, f"{preprocess_function.__name__}_{subfolder}")
        os.makedirs(dst_subfolder, exist_ok=True)

        file_list = os.listdir(src_subfolder)
        total_files = len(file_list)

        print(f"Processing {subfolder} folder...")
        for file in tqdm(file_list, total=total_files, unit='file'):
            file_path = os.path.join(src_subfolder, file)

            # Check if the file is an image by verifying the file extension
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                preprocessed_image = preprocess_function(file_path)

                # Save the preprocessed image to the destination subfolder
                preprocessed_image_path = os.path.join(dst_subfolder, file)
                cv2.imwrite(preprocessed_image_path, preprocessed_image)

    print(f"Finished applying {preprocess_function.__name__} to all images in the subfolders.")
    
#apply a color transformation, converting the image to the HSV color space
def to_hsv(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image

#apply a color transformation, converting the image to the YCBCR color space
def to_ycbcr(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the YCrCb color space (OpenCV uses the name YCrCb instead of YCbCr)
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    return ycbcr_image

def compute_rgb_glcm(image_path, distance=1, levels=256):
    # Read the image
    image = cv2.imread(image_path)

    # Ensure the image is 200x200
    if image.shape[:2] != (200, 200):
        raise ValueError("The input image must have a size of 200x200.")

    # Convert the image from BGR (OpenCV format) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize a 3x256x256 tensor for the co-occurrence matrices
    glcm_tensor = np.zeros((3, levels, levels), dtype=np.float32)

    # Compute the co-occurrence matrices for each channel
    for channel in range(3):
        glcm = graycomatrix(image[:, :, channel], [distance], [0], levels, symmetric=True, normed=True)
        glcm_tensor[channel, :, :] = glcm[:, :, 0, 0]

    return glcm_tensor

#a function that let you apply cooccurance preprocessing image to a folder 
#that contains test and train image in separate folders
def cooc_matrix(source_folder, distance=1, levels=256):
    # Create a dictionary to store the results for all subfolders
    result_dict = {}

    # List all subfolders in the source folder
    subfolders = ['test', 'train']

    # Loop through all subfolders
    for subfolder in subfolders:
        # Get the path to the subfolder
        subfolder_path = os.path.join(source_folder, subfolder)

        # Create an empty list for the subfolder
        sub_list = []

        # Get the list of image files in the subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

        # Create a progress bar for the subfolder
        pbar = tqdm(total=len(image_files), desc=f"Processing {subfolder}...")

        # Loop through all the image files in the subfolder
        for filename in image_files:
            # Get the full path to the image file
            image_path = os.path.join(subfolder_path, filename)

            try:
                # Compute the GLCM tensor for the image
                glcm_tensor = compute_rgb_glcm(image_path, distance=distance, levels=levels)

                # Add the GLCM tensor to the sub-list
                sub_list.append(glcm_tensor)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

        # Add the sub-list to the result dictionary with the subfolder name as key
        result_dict[subfolder] = sub_list

    # Modify the list at the 'test' key in the dictionary
    tensor_list = result_dict['test']
    tensor_list.append(tensor_list.pop(1999))
    tensor_list.append(tensor_list.pop(1999))

    # Save the dictionary to a file in the current directory
    file_path = os.path.join(os.getcwd(), f"{cooc_matrix.__name__}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(result_dict, f)

    # Return the dictionary of results
    return result_dict

#show a picture of cooccurrence matrix
def save_cooccurrence_matrix_as_image(matrix, output_path):
    # Normalize the co-occurrence matrix to the range [0, 255]
    normalized_matrix = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save the normalized matrix as an image
    cv2.imwrite(output_path, normalized_matrix)
#%% check
#processed_data = '/Users/harryzhu/Desktop/small_processed_data'
#apply_preprocessing_to_subfolders(high_pass_filter, processed_data)
#apply_preprocessing_to_subfolders(to_hsv, processed_data)
#apply_preprocessing_to_subfolders(to_ycbcr, processed_data)

#hi = cooc_matrix(processed_data)
#%% 

# Read the image
#image_path = '/Users/harryzhu/Desktop/processed_data/test/10391.jpg'
#image = cv2.imread(image_path)

# Get the dimensions (height, width, and channels) of the image
#height, width, channels = image.shape

#print(f"Image size: {width}x{height}")

#%% demo photo