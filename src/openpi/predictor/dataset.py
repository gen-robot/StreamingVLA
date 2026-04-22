import torch
import numpy as np
import h5py
from pathlib import Path
from config import cfg

class LiberoPhysicalDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, min_horizon=1, max_horizon=3, subset_chunk=["chunk-000", "chunk-001"], max_episodes=100):
        self.root = Path(root_path)
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.action_dim = cfg.action_dim
        

        if isinstance(subset_chunk, str):
            chunks = [subset_chunk]
        else:
            chunks = subset_chunk

        self.data_files = []
        for chunk in chunks:
            search_path = self.root / chunk
            chunk_files = sorted(list(search_path.glob("episode_*.h5")))
            self.data_files.extend(chunk_files)
            print(f"Loaded {len(chunk_files)} files from {chunk}")

        if max_episodes is not None:
            self.data_files = self.data_files[:max_episodes]

        if not self.data_files:
            raise FileNotFoundError(f"No .h5 files found in chunks: {chunks} under {self.root}")
        
        print(f"Total files loaded: {len(self.data_files)}")


        self.valid_indices = []
        
        for f in self.data_files:
            with h5py.File(f, 'r') as h5_file:
                num_rows = h5_file['actions'].shape[0]
                if num_rows > self.max_horizon:
                    for local_idx in range(num_rows - self.max_horizon):
                        self.valid_indices.append((str(f), local_idx))

        print(f"Total valid training samples: {len(self.valid_indices)}")
        self._open_files = {}

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        file_path, local_curr_idx = self.valid_indices[idx]
        horizon = np.random.randint(self.min_horizon, self.max_horizon + 1)
        local_next_idx = local_curr_idx + horizon

        with h5py.File(file_path, 'r', swmr=True) as data:
            emb_t = torch.from_numpy(data['embeddings'][local_curr_idx]).float()
            emb_next = torch.from_numpy(data['embeddings'][local_next_idx]).float()
            raw_actions = data['actions'][local_curr_idx : local_next_idx]
            
        actions = torch.zeros((horizon, self.action_dim), dtype=torch.float32)
        act_len = min(len(raw_actions), horizon)
        actions[:act_len, :] = torch.from_numpy(raw_actions[:act_len, :self.action_dim])

        return {
            "observation": emb_t,     
            "next_observation": emb_next, 
            "actions": actions,
            "horizon": int(horizon)
        }