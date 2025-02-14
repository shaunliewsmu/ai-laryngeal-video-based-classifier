import numpy as np
from transformers import  VivitConfig,VivitForVideoClassification
from transformers import TimesformerConfig, TimesformerForVideoClassification
import torch
import evaluate

metric = evaluate.load("accuracy", trust_remote_code=True)
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    """Custom collate function to properly handle the batch creation"""
    pixel_values = []
    labels = []
    
    for example in batch:
        # Convert to tensor if not already
        if not isinstance(example['pixel_values'], torch.Tensor):
            example['pixel_values'] = torch.tensor(example['pixel_values'])
        
        pixel_values.append(example['pixel_values'])
        labels.append(example['labels'])
    
    # Stack all tensors
    pixel_values = torch.stack(pixel_values)
    labels = torch.tensor(labels)
    
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


def initialise_model(shuffled_dataset, device="cpu", model="google/vivit-b-16x2-kinetics400", number_of_frames=10):
    """initialize model
    """ 
    labels = shuffled_dataset['train'].features['labels'].names
    config = VivitConfig.from_pretrained(model)
    config.num_classes=len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames=number_of_frames
    config.video_size= [number_of_frames, 224, 224]
    
    model = VivitForVideoClassification.from_pretrained(
    model,
    ignore_mismatched_sizes=True,
    config=config,).to(device)
    return model 

def initialise_timesformer_model(shuffled_dataset, device="cpu", model="facebook/timesformer-base-finetuned-k400", number_of_frames=10):
    """initialize model
    """ 
    labels = shuffled_dataset['train'].features['labels'].names
    config = TimesformerConfig.from_pretrained(model)
    config.num_classes=len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames=number_of_frames
    config.video_size= [number_of_frames, 224, 224]
    
    model = TimesformerForVideoClassification.from_pretrained(
    model,
    ignore_mismatched_sizes=True,
    config=config,).to(device)
    return model 