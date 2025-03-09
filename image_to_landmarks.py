import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Output file
output_file = 'landmarks_data_combined.csv'

# Function to extract hand landmarks as a flattened array
def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # Extract landmarks if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])  # x, y, z coordinates
            return landmarks
    return None  # No hand detected

# Main function to process multiple dataset folders
def process_multiple_datasets(dataset_folders):
    data = []

    for dataset_folder in dataset_folders:
        print(f"\nProcessing dataset folder: {dataset_folder}")
        for root, _, files in os.walk(dataset_folder):
            folder_name = os.path.basename(root)  # Subfolder name gives the label
            
            # Ensure folder_name is a valid single digit or letter
            if folder_name.isdigit():  # Digits 0-9
                label = int(folder_name)
            elif len(folder_name) == 1 and folder_name.isalpha():  # Letters a-z
                label = ord(folder_name.lower()) - ord('a') + 10
            else:
                continue  # Skip invalid folders
            
            # Process all images in the folder
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # Only image files
                    image_path = os.path.join(root, file)
                    print(f"Processing {image_path}...")
                    landmarks = extract_landmarks_from_image(image_path)
                    if landmarks is not None:
                        row = landmarks + [label]  # Append landmarks and label
                        data.append(row)

    # Save combined data to CSV
    if data:
        columns = [f"landmark_{i}" for i in range(63)] + ["label"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file, index=False)
        print(f"\nLandmark data saved to '{output_file}' with {len(df)} records.")
    else:
        print("\nNo landmarks were extracted. Check your images or folder structure.")

if __name__ == "__main__":
    # Paths to both dataset folders
    dataset_folders = ['./asldataset', './dataset/asl_dataset'] 
    process_multiple_datasets(dataset_folders)
 