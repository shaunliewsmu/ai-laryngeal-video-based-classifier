# Video Based Classifier for AI Laryngeal

## First model: Vivit

### How to run the model training

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the model training with Linux Screen

First, you'll need to create a new screen session:

```bash
screen -S training
```

This will open a new screen session named "training". Within this session, you can run your command:

can change the training script command according to the model that you wanna train. 

```bash
python3 resnet50-2d-lstm/main.py \
    --data_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
    --test_dir artifacts/duhs-gss-split-5:v0/organized_dataset \
    --log_dir logs \
    --model_dir resnet50-2d-lstm-models \
    --train_sampling random_window \
    --val_sampling uniform \
    --test_sampling uniform \
    --batch_size 4 \
    --epochs 1 \
    --learning_rate 0.001 \
    --patience 10 \
    --loss_weight 0.3
```

Once your command is running, you can detach from the screen session without stopping the process by pressing:

```bash
Ctrl+A followed by D
```

You'll see a message like "[detached from session_name]" and you'll be returned to your normal SSH session. You can now safely disconnect from your SSH server.

When you reconnect later, you can list all available screen sessions with:

```bash
screen -ls
```

To reattach to your training session:

```bash
screen -r training
```

If you have multiple screen sessions and aren't sure which one is which, you can use screen -ls to see all sessions and their IDs, then attach to a specific one with:

```bash
screen -r [session_id]
```

Additional useful commands:

```bash
To create a new window within the same screen session: Ctrl+A then c
To switch between windows: Ctrl+A then n (next) or p (previous)
To kill a screen session (when attached): Ctrl+A then k
To scroll within a screen session: Ctrl+A then [ (use arrow keys to scroll, press Esc to exit scroll mode)
```
