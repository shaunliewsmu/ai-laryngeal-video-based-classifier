import wandb
run = wandb.init()
artifact = run.use_artifact('enhance-ai/enhance-ai-data-splits/bagls-split:v0', type='dataset')
artifact_dir = artifact.download()