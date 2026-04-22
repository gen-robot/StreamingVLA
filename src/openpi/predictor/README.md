# Predictor README

## 1. Overview
This directory contains the training and data-processing code for the `openpi/predictor` module. It is used to train a predictor that models visual-state transitions from the current visual state and action inputs.

Core files in this directory:

- `image_embedding.py`: Pre-encodes raw images into embeddings to avoid repeated image processing during training.
- `config.py`: Centralized configuration for hyperparameters, data paths, and checkpoint output paths.
- `dataset.py`: Dataset loading logic for training.
- `train.py`: Distributed training entrypoint.

## 2. Training Workflow

### Step 1. Precompute image embeddings offline (strongly recommended)

First, run `image_embedding.py` to convert images into embeddings. This avoids repeated image encoding at training time and significantly improves training efficiency.

Example command:

```bash
python src/openpi/predictor/image_embedding.py \
	--root /path/to/raw_dataset_root \
	--output /path/to/embedding_output_root \
	--chunks chunk-000,chunk-001 \
	--batch_size 16 \
	--device cuda \
	--paligemma_variant paligemma-3b-pt-224 \
	--action_variant gemma-2b-it \
	--drop_image_bytes
```

Common arguments:

- `--root`: Root directory of the raw dataset (the script reads `root/data/chunk-xxx/episode_*.parquet`).
- `--output`: Output directory for embedding files.
- `--chunks`: Dataset chunks to process, separated by commas.
- `--force`: Optional. Force overwrite if embedding files already exist.

### Step 2. Adjust training configuration

In `src/openpi/predictor/config.py`, set suitable training values. At minimum, verify the following fields:

- `dataset_local_path`: Training dataset path.
- `save_dir`: Checkpoint output directory.
- `batch_size`, `learning_rate`, `num_epochs`, `num_workers`.
- `subset_chunk`, `max_episodes`, `min_horizon`, `max_horizon`.
- `resume_checkpoint` (for resuming training, if needed).

### Step 3. Start training

`train.py` initializes distributed training, so `torchrun` is recommended:

```bash
torchrun --nproc_per_node=1  train.py
```

Multi-GPU example (8 GPUs):

```bash
torchrun --nproc_per_node=8  train.py
```

During training, the script will:

- Split data into train/val and run iterative training.
- Log metrics to wandb.
- Save `latest.pth` and `best_model.pth` under `save_dir`.

## 3. Quick Checklist

- Embedding preprocessing is completed (recommended).
- Paths and hyperparameters in `config.py` are confirmed.
- `save_dir` is writable.
- Training is launched with `torchrun`.
