#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:16:02 2023

@author: harryzhu
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json

#%%
#make some training data validation data for autoML
df = pd.read_csv('image_data.csv')

df['google'] = ''

df.loc[df['folder'] == 'test', 'google'] = 'test'

train_df = df[df['folder'] == 'train']
train_set, valid_set = train_test_split(train_df, test_size=1/8,stratify=train_df['dataset_name'],random_state=42)

df.loc[train_set.index, 'google'] = 'training'
df.loc[valid_set.index, 'google'] = 'validation'

df.to_csv('modified_file_path.csv', index=False)

#%% combine the bucket path of two folders
with open("hsv_test_list.txt", "r") as file1:
    content1 = file1.read()
    
with open("hsv_train_list.txt", "r") as file2:
    content2 = file2.read()
    
merged_content = content1 + "\n" + content2

with open("hsv.txt", "w") as merged_file:
    merged_file.write(merged_content)
#%% create the csv file for creating the dataset in Google Cloud for autoML 

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

#%% createing new csv file to create dataset for other buckets base on exisiting csv

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
        
#%% prediction result reading

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

#%% Combine the prediction result with the original metadata

# Load the image_data CSV into a dataframe
image_data = pd.read_csv('/Users/harryzhu/Desktop/small_processed_data/image_data.csv')

# Merge the dataframes on the 'id' column
result = pd.merge(image_data, df, on='id')

# Select only the columns we want in the final output
result = result[['id', 'label', 'dataset_name', 'prediction', 'confidence']]

# Write the resulting CSV to a file
result.to_csv('/Users/harryzhu/Desktop/small_processed_data/batch_predict.csv', index=False)