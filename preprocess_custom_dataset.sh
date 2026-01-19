#!/bin/bash

# Bash script to preprocess a custom dataset for robotic surgery frame analysis
# This script runs python scripts/preprocess_custom_dataset.py --input_dir <input_directory> --output_dir <output_directory>
# Usage: ./preprocess_custom_dataset.sh \
# --input_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_<idx>/images/
# --output_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_<idx>/images_cropped/
# for idx in {1,2,3,4,5,6}

# Using seq instead of brace expansion
# for idx in $(seq 1 6); do
#     echo "Processing video_$idx..."
#     # ... rest of your code ...
# done

for idx in $(seq 1 6); do
echo "Processing video_$idx images..."
python scripts/preprocess_custom_dataset.py \
--input_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_$idx/images/ \
--output_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_$idx/images_cropped/ 
done

# after doing it for images/ folders, do it for pose_maps/ folders
for idx in $(seq 1 6); do
python scripts/preprocess_custom_dataset.py \
--input_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_$idx/pose_maps/ \
--output_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/train/video_$idx/pose_maps_cropped/ 
done

# repeating the same two steps for the validation set
for idx in $(seq 1 6); do
python scripts/preprocess_custom_dataset.py \
--input_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/val/video_$idx/images/ \
--output_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/val/video_$idx/images_cropped/ 
done

for idx in $(seq 1 6); do
python scripts/preprocess_custom_dataset.py \
--input_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/val/video_$idx/pose_maps/ \
--output_dir /rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/val/video_$idx/pose_maps_cropped/ 
done
