import os
import wandb
import pandas as pd

def save_to_wandb():
    # Initialize wandb run
    run = wandb.init(project="laryngeal_cancer_video_classification", name="dataset_upload")
    
    # Create a new artifact
    artifact = wandb.Artifact(
        name="laryngeal_dataset_balanced",
        type="dataset",
        description="Balanced laryngeal cancer dataset split into train/val/test"
    )
    
    # Base directory of the balanced dataset
    dataset_dir = 'dataset_balanced'
    
    # Add the entire dataset directory to the artifact
    artifact.add_dir(dataset_dir, name="dataset")
    
    # Create metadata about the dataset structure
    dataset_stats = {
        'splits': {}
    }
    
    # Calculate statistics for each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        referral_count = len(os.listdir(os.path.join(split_dir, 'referral')))
        non_referral_count = len(os.listdir(os.path.join(split_dir, 'non_referral')))
        total = referral_count + non_referral_count
        
        dataset_stats['splits'][split] = {
            'total': total,
            'referral': referral_count,
            'non_referral': non_referral_count,
            'referral_ratio': f"{(referral_count/total)*100:.2f}%",
            'non_referral_ratio': f"{(non_referral_count/total)*100:.2f}%"
        }
    
    # Save statistics as a JSON file
    stats_df = pd.DataFrame.from_dict(
        {(i,j): dataset_stats['splits'][i][j] 
         for i in dataset_stats['splits'].keys() 
         for j in dataset_stats['splits'][i].keys()},
        orient='index'
    )
    
    # Save statistics to CSV
    stats_file = "dataset_statistics.csv"
    stats_df.to_csv(stats_file)
    
    # Add statistics file to artifact
    artifact.add_file(stats_file, name="dataset_statistics.csv")
    
    # Log metadata table to wandb
    table_data = []
    for split in ['train', 'val', 'test']:
        stats = dataset_stats['splits'][split]
        table_data.append([
            split,
            stats['total'],
            stats['referral'],
            stats['non_referral'],
            stats['referral_ratio'],
            stats['non_referral_ratio']
        ])
    
    columns = ["Split", "Total", "Referral", "Non-referral", "Referral Ratio", "Non-referral Ratio"]
    table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"dataset_statistics": table})
    
    # Log some summary metrics
    total_videos = sum(stats['total'] for stats in dataset_stats['splits'].values())
    total_referral = sum(stats['referral'] for stats in dataset_stats['splits'].values())
    
    artifact.metadata = {
        'total_videos': total_videos,
        'total_referral': total_referral,
        'total_non_referral': total_videos - total_referral,
        'referral_ratio': f"{(total_referral/total_videos)*100:.2f}%",
        'dataset_structure': {
            'train': dataset_stats['splits']['train'],
            'val': dataset_stats['splits']['val'],
            'test': dataset_stats['splits']['test']
        }
    }
    
    # Save the artifact
    run.log_artifact(artifact)
    
    print(f"\nDataset has been uploaded to W&B:")
    print(f"Total videos: {total_videos}")
    print(f"Total referral: {total_referral}")
    print(f"Total non-referral: {total_videos - total_referral}")
    
    # Clean up temporary files
    os.remove(stats_file)
    
    return artifact

if __name__ == "__main__":
    artifact = save_to_wandb()