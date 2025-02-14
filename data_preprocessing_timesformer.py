# In data_preprocessing.py
from datasets import Dataset
from transformers import AutoImageProcessor
import torch
import numpy as np

image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

def process_example(example):
    """Process a single video example"""
    try:
        # Convert video frames to numpy array if not already
        video_frames = np.array(example['video'])
        
        # Process with image processor
        inputs = image_processor(list(video_frames), return_tensors='pt')
        
        # Convert to tensor and remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Ensure pixel_values is a tensor
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
            
        return {
            'pixel_values': pixel_values,
            'labels': example['labels']
        }
    except Exception as e:
        print(f"Error processing example: {str(e)}")
        print(f"Video shape: {np.array(example['video']).shape}")
        raise e

def create_dataset(video_dictionary):
    """Create and process the dataset"""
    # Create initial dataset
    dataset = Dataset.from_list(video_dictionary)
    
    # Encode labels
    dataset = dataset.class_encode_column("labels")
    
    # Process videos and ensure tensor output
    processed_dataset = dataset.map(
        process_example,
        remove_columns=['video'],
        desc="Processing videos"
    )
    
    # Add debug prints
    sample = processed_dataset[0]
    print("First sample structure:")
    print(f"Keys: {sample.keys()}")
    print(f"Pixel values type: {type(sample['pixel_values'])}")
    if isinstance(sample['pixel_values'], torch.Tensor):
        print(f"Pixel values shape: {sample['pixel_values'].shape}")
    
    # Shuffle dataset
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    split_dataset = shuffled_dataset.train_test_split(test_size=0.1)
    
    return split_dataset