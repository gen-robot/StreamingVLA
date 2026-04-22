import torch
from dataclasses import dataclass,field
from typing import Union, List  
@dataclass
class TrainConfig:
    
    project_name: str = "your_project_name"
    dataset_id: str = "id_of_your_dataset"
    dataset_local_path: str = "path_to_your_dataset" 
    resume_checkpoint: str = None 
   
# For Training
    batch_size: int = 64
    learning_rate: float = 5e-4
    min_lr: float = 3e-5
    num_epochs: int = 20
    num_workers: int = 4
    n_layers = 12
    in_dim = 2048
    hidden_dim = 1024
    emb_len = 256
    n_heads = 16
    cond_dim = 256
    dim_feedforward = 3072

    
    # predict [min_horizon, max_horizon] steps 
    min_horizon: int = 1
    max_horizon: int = 3
    
    action_dim: int = 6 
    image_name: str = "image_embedding" 

    

# For Dataset
    subset_chunk: List[str] = field(default_factory=lambda: ["chunk-000", "chunk-001"])
    max_episodes: int = 100
    
# For devices
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "path_to_save_checkpoints"

cfg = TrainConfig()