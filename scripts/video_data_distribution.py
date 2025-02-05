import os
import pandas as pd
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import wandb

# Initialize wandb
wandb.init(project="laryngeal_cancer_video_classification", name="data_split_balanced")

def create_balanced_splits(df, videos_dir, dest_base_dir, split_ratios=(0.7, 0.15, 0.15)):
    """
    Create balanced train/val/test splits using the CSV dataframe
    """
    # First split: training set
    train_df, temp_df = train_test_split(
        df,
        train_size=split_ratios[0],
        stratify=df['Label'],  # Stratify based on the label to maintain class balance
        random_state=42
    )
    
    # Second split: validation and test sets
    val_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df['Label'],  # Maintain stratification for second split
        random_state=42
    )
    
    # Create all necessary directories
    splits = ['train', 'val', 'test']
    categories = ['referral', 'non_referral']
    for split in splits:
        for category in categories:
            os.makedirs(os.path.join(dest_base_dir, split, category), exist_ok=True)
    
    # Function to copy files for a specific dataframe
    def copy_files_for_split(split_df, split_name):
        stats = {'referral': 0, 'non_referral': 0}
        for _, row in split_df.iterrows():
            filename = row['File Name']
            label = row['Label']
            category = 'referral' if label == 1 else 'non_referral'
            
            source_path = os.path.join(videos_dir, filename)
            dest_path = os.path.join(dest_base_dir, split_name, category, filename)
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                stats[category] += 1
            else:
                print(f"Warning: File not found - {filename}")
        
        return stats
    
    # Copy files for each split
    train_stats = copy_files_for_split(train_df, 'train')
    val_stats = copy_files_for_split(val_df, 'val')
    test_stats = copy_files_for_split(test_df, 'test')
    
    return {
        'train': train_stats,
        'val': val_stats,
        'test': test_stats
    }

def main():
    # Read the CSV file
    df = pd.read_csv('data_description.csv')
    
    # Print initial class distribution
    print("\nInitial class distribution:")
    print(df['Label'].value_counts(normalize=True))
    
    # Base directories
    videos_dir = 'videos'  # Directory containing all videos
    dataset_dir = 'dataset_balanced'
    
    # Create balanced splits
    stats = create_balanced_splits(df, videos_dir, dataset_dir)
    
    # Print statistics
    print("\nDataset split statistics:")
    for split, split_stats in stats.items():
        total = split_stats['referral'] + split_stats['non_referral']
        ref_ratio = split_stats['referral'] / total if total > 0 else 0
        print(f"\n{split.upper()} set:")
        print(f"Total videos: {total}")
        print(f"Referral: {split_stats['referral']} ({ref_ratio:.2%})")
        print(f"Non-referral: {split_stats['non_referral']} ({1-ref_ratio:.2%})")
    
    # Log to wandb
    wandb.log({
        "referral_train_count": stats['train']['referral'],
        "referral_val_count": stats['val']['referral'],
        "referral_test_count": stats['test']['referral'],
        "non_referral_train_count": stats['train']['non_referral'],
        "non_referral_val_count": stats['val']['non_referral'],
        "non_referral_test_count": stats['test']['non_referral'],
    })
    
    # Create a summary table for wandb
    data = []
    for split, split_stats in stats.items():
        for category, count in split_stats.items():
            data.append([split, category, count])
    
    table = wandb.Table(
        data=data,
        columns=["Split", "Category", "Count"]
    )
    wandb.log({"dataset_splits": table})
    
    # Save the dataset information to a CSV file and log it to wandb
    df = pd.DataFrame(data, columns=["Split", "Category", "Count"])
    csv_path = "dataset_splits.csv"
    df.to_csv(csv_path, index=False)
    wandb.save(csv_path)
    
    print("\nDataset splitting complete! The new directory structure is:")
    print(f"""
    dataset_balanced/
    ├── train/
    │   ├── referral/
    │   └── non_referral/
    ├── val/
    │   ├── referral/
    │   └── non_referral/
    └── test/
        ├── referral/
        └── non_referral/
    """)

if __name__ == "__main__":
    main()