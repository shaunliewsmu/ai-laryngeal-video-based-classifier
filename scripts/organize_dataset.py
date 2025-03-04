import os
import shutil
import argparse
import pandas as pd

def create_dataset_structure(csv_dir, video_dir, output_dir):
    """
    Create a dataset structure based on CSV files containing video filenames and labels.
    
    Args:
        csv_dir (str): Directory containing the CSV files
        video_dir (str): Directory containing the original video files
        output_dir (str): Directory to create the organized dataset structure
    """
    # Define the CSV files and dataset splits
    splits = ['train', 'val', 'test']
    csv_files = [f"{split}.table_unique.csv" for split in splits]
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track copied files and any issues
    copied_files = 0
    missing_files = []
    
    # Process each split
    for split, csv_file in zip(splits, csv_files):
        csv_path = os.path.join(csv_dir, csv_file)
        
        # Skip if CSV file doesn't exist
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file {csv_path} not found. Skipping...")
            continue
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Processing {csv_file} with {len(df)} videos...")
        
        # Create directories for this split
        split_dir = os.path.join(output_dir, split)
        referral_dir = os.path.join(split_dir, "referral")
        non_referral_dir = os.path.join(split_dir, "non-referral")
        
        os.makedirs(referral_dir, exist_ok=True)
        os.makedirs(non_referral_dir, exist_ok=True)
        
        # Process each video in the dataframe
        for idx, row in df.iterrows():
            video_file = row['video_file']
            label = int(row['label'])
            
            # Determine source and destination paths
            src_path = os.path.join(video_dir, video_file)
            dst_dir = referral_dir if label == 1 else non_referral_dir
            dst_path = os.path.join(dst_dir, video_file)
            
            # Copy the file if it exists
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_files += 1
                if copied_files % 100 == 0:
                    print(f"Copied {copied_files} files...")
            else:
                missing_files.append(video_file)
    
    # Print summary
    print("\nDataset creation complete!")
    print(f"Total videos copied: {copied_files}")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} videos were not found in the source directory.")
        print("First 10 missing files:")
        for file in missing_files[:10]:
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Create dataset structure from CSV files.')
    parser.add_argument('--csv_dir', type=str, required=True,
                        help='Directory containing the CSV files')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing the original video files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to create the organized dataset structure')
    
    args = parser.parse_args()
    
    # Create the dataset structure
    create_dataset_structure(args.csv_dir, args.video_dir, args.output_dir)

if __name__ == "__main__":
    main()
    
"""
python3 scripts/organize_dataset.py --csv_dir artifacts/duhs-gss-split-1:v0 --video_dir videos --output_dir artifacts/duhs-gss-split-1:v0/organized_dataset
"""