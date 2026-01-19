"""
Docstring for scripts.preprocess_custom_dataset
Code to preprocess a custom dataset for robotic surgery tool keypoint tracking.]
Take all the images and crop from 1920x1080 to 1440x1080 centered.
"""

import os
import cv2
import glob
import argparse
from tqdm import tqdm
import numpy as np

def preprocess_custom_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))
    
    for img_file in tqdm(image_files, desc="Preprocessing images"):
        img = cv2.imread(img_file)
        h, w, _ = img.shape
        if w > 1440:
            start_x = (w - 1440) // 2
            cropped_img = img[:, start_x:start_x + 1440]
        else:
            cropped_img = img
        
        output_file = os.path.join(output_dir, os.path.basename(img_file))
        cv2.imwrite(output_file, cropped_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess custom dataset for robotic surgery tool keypoint tracking.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory containing images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory to save preprocessed images.')
    args = parser.parse_args()
    
    preprocess_custom_dataset(args.input_dir, args.output_dir)

