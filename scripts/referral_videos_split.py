import pandas as pd
import os
import shutil

# Read the CSV file
df = pd.read_csv('data_description.csv')

# Create directories if they don't exist
base_dir = 'data'
referral_dir = os.path.join(base_dir, 'referral')
non_referral_dir = os.path.join(base_dir, 'non_referral')

os.makedirs(referral_dir, exist_ok=True)
os.makedirs(non_referral_dir, exist_ok=True)

# Source directory containing all videos
source_dir = 'videos'  # original videos directory

# Organize videos
for _, row in df.iterrows():
    video_filename = row['File Name']
    label = row['Label']
    
    # Construct source path
    source_path = os.path.join(source_dir, video_filename)
    
    # Skip if source file doesn't exist
    if not os.path.exists(source_path):
        print(f"Warning: Source file not found - {video_filename}")
        continue
    
    # Determine destination directory based on label
    if label == 1:
        dest_dir = referral_dir
    else:
        dest_dir = non_referral_dir
    
    # Construct destination path
    dest_path = os.path.join(dest_dir, video_filename)
    
    # Copy the file
    try:
        shutil.copy2(source_path, dest_path)
        print(f"Copied {video_filename} to {dest_dir}")
    except Exception as e:
        print(f"Error copying {video_filename}: {str(e)}")

print("\nOrganization complete!")
# Print summary
referral_count = len(os.listdir(referral_dir))
non_referral_count = len(os.listdir(non_referral_dir))
print(f"\nSummary:")
print(f"Referral videos: {referral_count}")
print(f"Non-referral videos: {non_referral_count}")