#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:26:10 2023

@author: harryzhu
"""

import os
import random
import shutil
import csv
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from shutil import copyfile
import glob
import re
import json

# Directory settings
project_dataset_dir = '/Users/harryzhu/Desktop/project_dataset'
input_real_dir = os.path.join(project_dataset_dir, 'real')
input_fake_dir = os.path.join(project_dataset_dir, 'fake')
stylegan3_dir = os.path.join(input_fake_dir, 'stylegan3')
output_dir = 'autoML data'
output_data_dir = os.path.join(output_dir, 'data')
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Create output directories if they don't exist
os.makedirs(output_data_dir, exist_ok=True)

# CSV file
csv_file = os.path.join(output_dir, 'image_data.csv')

# Image counter
image_counter = 0

def save_images(images, label, dataset_name):
    global image_counter
    with open(csv_file, 'a', newline='') as csvfile: 
        fieldnames = ['id', 'label', 'dataset_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for img_path in tqdm(images, desc=f"Processing {dataset_name} images"):
            img = Image.open(img_path)

            output_img_path = os.path.join(output_data_dir, f"{image_counter}.jpg")
            img.save(output_img_path)

            writer.writerow({'id': image_counter, 'label': label, 'dataset_name': dataset_name})
            image_counter += 1

def process_images(input_dir, label, total_images, excluded_folders=None):
    if excluded_folders:
        input_subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d not in excluded_folders]
    else:
        input_subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    images_per_subdir = total_images // len(input_subdirs)

    for subdir in input_subdirs:
        subdir_path = os.path.join(input_dir, subdir)
        images = list(Path(subdir_path).rglob('*.[jJ][pP]*[gG]'))
        random.shuffle(images)

        selected_images = images[:images_per_subdir]
        save_images(selected_images, label, subdir)
        
def split_data(output_dir, csv_file, train_ratio=0.8, test_ratio=0.2):
    assert train_ratio + test_ratio == 1, "Train and test ratios must sum to 1"

    # Create train and test directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Group by dataset_name and split into train and test
    for dataset_name, group in df.groupby('dataset_name'):
        print(f"Processing dataset: {dataset_name}")
        
        train_df, test_df = train_test_split(group, train_size=train_ratio, shuffle=True)

        # Save images in corresponding directories without changing their names
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Train - {dataset_name}"):
            img_name = f"{row['id']}.jpg"
            src = os.path.join(output_data_dir, img_name)
            dst = os.path.join(train_dir, img_name)
            copyfile(src, dst)
            df.loc[idx, 'folder'] = 'train'

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Test - {dataset_name}"):
            img_name = f"{row['id']}.jpg"
            src = os.path.join(output_data_dir, img_name)
            dst = os.path.join(test_dir, img_name)
            copyfile(src, dst)
            df.loc[idx, 'folder'] = 'test'

    # Update the CSV file with the new folder column
    df.to_csv(csv_file, index=False)
    print("Data splitting completed.")

def print_dataset_name_distribution(csv_file):
    df = pd.read_csv(csv_file)

    for folder_name in ['train', 'val', 'test']:
        folder_df = df[df['folder'] == folder_name]
        print(f"\n{folder_name.capitalize()} folder:")
        total_images = len(folder_df)

        for dataset_name, group in folder_df.groupby('dataset_name'):
            count = len(group)
            percentage = (count / total_images) * 100
            print(f"{dataset_name}: {count} images ({percentage:.2f}%)")

#not used
def process_stylegan3(input_fake_dir, stylegan3_dir, test_dir, csv_file):
    stylegan3_images = list(Path(stylegan3_dir).rglob('*.[jJ][pP]*[gG]'))
    random.shuffle(stylegan3_images)

    df = pd.read_csv(csv_file)
    test_df = df[df['folder'] == 'test']
    fake_test_df = test_df[test_df['label'] == 'fake']
    
    total_images = len(fake_test_df)
    unique_datasets = len(fake_test_df['dataset_name'].unique())
    
    images_per_subdir = total_images // unique_datasets

    stylegan3_selected_images = stylegan3_images[:images_per_subdir]
    
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['id', 'label', 'dataset_name', 'folder']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for img_path in tqdm(stylegan3_selected_images, desc="Processing StyleGAN3 images"):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path)

            output_img_path = os.path.join(test_dir, f"{img_id}.jpg")
            img.save(output_img_path)

            writer.writerow({'id': img_id, 'label': 'fake', 'dataset_name': 'stylegan3', 'folder': 'test'})

def process_extra_test(input_fake_dir, folder_name, test_dir, csv_file):
    folder_dir = os.path.join(input_fake_dir, folder_name)
    images = [img_path for img_path in Path(folder_dir).rglob('*') if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)

    df = pd.read_csv(csv_file)
    test_df = df[df['folder'] == 'test']
    fake_test_df = test_df[test_df['label'] == 'fake']
    
    last_id = df['id'].max() if not df.empty else 0
    next_id = last_id + 1
    
    total_images = len(fake_test_df)
    unique_datasets = len(fake_test_df['dataset_name'].unique())
    
    images_per_subdir = total_images // unique_datasets
    
    selected_images = images[:images_per_subdir]
    
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['id', 'label', 'dataset_name', 'folder']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for img_path in tqdm(selected_images, desc=f"Processing {folder_name} images"):
            img_id = str(next_id)
            img = Image.open(img_path)
            
            # Resize the image to 200x200 if the input folder name is 'photoshop'
            if folder_name.lower() == 'photoshop':
                img = img.resize((200, 200))

            output_img_path = os.path.join(test_dir, f"{img_id}.jpg")
            img.save(output_img_path)

            writer.writerow({'id': img_id, 'label': 'fake', 'dataset_name': folder_name, 'folder': 'test'})
            next_id += 1
#%% Truncate data

# Initialize the CSV file
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['id', 'label', 'dataset_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

# Process real and fake images
process_images(input_real_dir, 'real', 5000)
process_images(input_fake_dir, 'fake', 5000, excluded_folders=['stylegan3','photoshop'])

#%% Data split

# Call the function to split the data
split_data(output_dir,csv_file)

#%%
process_extra_test(input_fake_dir, 'stylegan3', test_dir, csv_file)
process_extra_test(input_fake_dir,'photoshop',test_dir,csv_file)

#%%
print_dataset_name_distribution(csv_file)

#%%

df = pd.read_csv('image_data.csv')

df['google'] = ''

df.loc[df['folder'] == 'test', 'google'] = 'test'

train_df = df[df['folder'] == 'train']
train_set, valid_set = train_test_split(train_df, test_size=1/8,stratify=train_df['dataset_name'],random_state=42)

df.loc[train_set.index, 'google'] = 'training'
df.loc[valid_set.index, 'google'] = 'validation'

df.to_csv('modified_file_path.csv', index=False)

#%%
with open("hsv_test_list.txt", "r") as file1:
    content1 = file1.read()
    
with open("hsv_train_list.txt", "r") as file2:
    content2 = file2.read()
    
merged_content = content1 + "\n" + content2

with open("hsv.txt", "w") as merged_file:
    merged_file.write(merged_content)
#%%

# Read the merged text file line by line
with open("merged_file.txt", "r") as merged_file:
    image_lines = [line.strip() for line in merged_file if line.strip()]

# Read the existing CSV file into a pandas DataFrame
df = pd.read_csv('modified_file_path.csv')

# Create a new DataFrame to store the new CSV structure
new_df = pd.DataFrame(columns=['google', 'image_file', 'label'])

# Iterate over the image_lines and add the corresponding data to the new DataFrame
for image_line in image_lines:
    image_number = int(image_line.split('/')[-1].replace('.jpg', ''))
    matched_row = df[df['id'] == image_number].iloc[0]
    google_value = matched_row['google']
    label_value = matched_row['label']
    new_df = new_df.append({'google': google_value, 'image_file': image_line.strip(), 'label': label_value}, ignore_index=True)

# Save the new DataFrame to a new CSV file
new_df.to_csv('input_file.csv', index=False)

#%%

# Read the CSV file
csv_file = 'input_file.csv'
df = pd.read_csv(csv_file)

# Read the text file
text_file = 'hsv.txt'
with open(text_file, 'r') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]

# Create a dictionary to map the extracted filenames to their original values in the text file
filename_map = {os.path.basename(line): line for line in lines}

# Define a function to replace the value in the image_file column
def replace_value(path):
    filename = os.path.basename(path)
    return filename_map.get(filename, path)

# Replace the values in the image_file column if they match the values in the text file
df['image_file'] = df['image_file'].apply(replace_value)

# Save the modified DataFrame to a new CSV file

output_csv = 'input_hsv_file.csv'
df.to_csv(output_csv, index=False)

#%% create jsonl for batch prediction

# Read the text file
with open("test_list.txt", "r") as f:
    lines = f.readlines()

# Create a list of dictionaries
data = [{"content": line.strip()} for line in lines]

# Save the list of dictionaries as a JSON file
with open("batch_predic.jsonl", "w") as f:
    for entry in data:
        json.dump(entry, f)
        f.write("\n")
        
#%% prediction result 

# Create an empty list to hold the data
data = []

# Loop through each JSONL file
for filename in os.listdir('/Users/harryzhu/Desktop/small_processed_data/batch prediction result'):
    if filename.endswith('.jsonl'):
        # Open the file
        with open(os.path.join('/Users/harryzhu/Desktop/small_processed_data/batch prediction result', filename), 'r') as f:
            # Loop through each line
            for line in f:
                # Load the JSON string
                obj = json.loads(line)

                # Extract the instance ID, prediction label, and confidence
                instance_id = int(obj['instance']['content'].split('/')[-1].split('.')[0])
                prediction_label = obj['prediction']['displayNames'][0]
                confidence = obj['prediction']['confidences'][0]

                # Append to the data list
                data.append({'id': instance_id, 'prediction': prediction_label, 'confidence': confidence})

# Convert the list of dictionaries to a Pandas DataFrame
df = pd.DataFrame(data)

#%%

# Load the image_data CSV into a dataframe
image_data = pd.read_csv('/Users/harryzhu/Desktop/small_processed_data/image_data.csv')

# Merge the dataframes on the 'id' column
result = pd.merge(image_data, df, on='id')

# Select only the columns we want in the final output
result = result[['id', 'label', 'dataset_name', 'prediction', 'confidence']]

# Write the resulting CSV to a file
result.to_csv('/Users/harryzhu/Desktop/small_processed_data/batch_predict.csv', index=False)