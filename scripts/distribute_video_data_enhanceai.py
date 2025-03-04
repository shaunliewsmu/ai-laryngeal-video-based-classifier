import json
import os
import pandas as pd
import argparse
from collections import defaultdict

def process_json_file(file_path):
    """
    Process a W&B table JSON file and extract unique video filenames and labels.
    Maps grades to binary labels: 0 for non-referral (grade 1), 1 for referral (grades 2-3)
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract column indices
    columns = data['columns']
    video_file_idx = columns.index('video_file')
    referral_grade_idx = columns.index('referral_grade')
    label_idx = columns.index('label')
    
    # Dictionary to store unique video files and their labels
    unique_videos = {}
    
    # Process each row
    for row in data['data']:
        video_file = row[video_file_idx]
        referral_grade = row[referral_grade_idx]
        label = row[label_idx]
        
        # Store the video file and its label if not already stored
        if video_file not in unique_videos:
            # Map grades to binary labels (0: non-referral, 1: referral)
            # Assuming 'grade 1' is non-referral and 'grade 2' or 'grade 3' are referral
            binary_label = 0 if referral_grade.lower() == 'grade 1' else 1
            
            # Use the provided label if it aligns with our binary mapping
            # This is just a check to ensure consistency
            if label == binary_label:
                unique_videos[video_file] = label
            else:
                # Use our mapped binary label if there's a discrepancy
                unique_videos[video_file] = binary_label
    
    return unique_videos

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract unique video files and labels from JSON files.')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Directory containing the JSON files (default: current directory)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save the output CSV files (default: current directory)')
    args = parser.parse_args()
    
    # Define input and output paths
    input_file_names = ['train.table.json', 'val.table.json', 'test.table.json']
    input_files = [os.path.join(args.input_dir, file_name) for file_name in input_file_names]
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for input_file, file_name in zip(input_files, input_file_names):
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} not found. Skipping...")
            continue
        
        # Process the file
        unique_videos = process_json_file(input_file)
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'video_file': list(unique_videos.keys()),
            'label': list(unique_videos.values())
        })
        
        # Define output filename
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_unique.csv")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"Processed {input_file}, found {len(unique_videos)} unique videos.")
        print(f"Saved results to {output_file}")
        
        # Print a summary of the labels
        label_counts = df['label'].value_counts().to_dict()
        print(f"Label distribution: {label_counts}")
        print(f"Non-referral (0): {label_counts.get(0, 0)}, Referral (1): {label_counts.get(1, 0)}")
        print('-' * 50)

if __name__ == "__main__":
    main()
    
"""
python3 scripts/distribute_video_data_enhanceai.py --input_dir artifacts/duhs-gss-split-3:v0 --output_dir artifacts/duhs-gss-split-1:v0
"""