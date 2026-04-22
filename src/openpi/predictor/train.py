import os
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
import wandb
import os
from transformers import get_cosine_schedule_with_warmup
import torch.multiprocessing as mp
from datetime import timedelta

mp.set_sharing_strategy('file_system')
os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_IB_DISABLE"] = "0"
#os.environ["NCCL_DEBUG"] = "INFO"

from model_test import DiTPredictor 
from dataset import LiberoPhysicalDataset
from config import cfg 

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        if key in os.environ:
            pass 

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def collate_fn(batch):
    observations = torch.stack([s["observation"] for s in batch], dim=0)       
    next_observations = torch.stack([s["next_observation"] for s in batch], dim=0) 
    actions_sum_list = []
    for s in batch:
        actions_sum_list.append(s["actions"].sum(dim=0))
    
    actions_sum = torch.stack(actions_sum_list, dim=0) # [B, 6]

    return {
        "observation": observations,      # (B, 256, 2048)
        "next_observation": next_observations, # (B, 256, 2048)
        "actions_sum": actions_sum,       # (B, 6)
    }


def train():
    local_rank = setup_dist()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    
    seed_everything(42 + rank) 

    if is_main_process:
        wandb.init(project=cfg.project_name, config=vars(cfg))
        os.makedirs(cfg.save_dir, exist_ok=True)

    full_dataset = LiberoPhysicalDataset(
        root_path=cfg.dataset_local_path,
        min_horizon=cfg.min_horizon,
        max_horizon=cfg.max_horizon,
        subset_chunk=cfg.subset_chunk,
        max_episodes=cfg.max_episodes
    )
    
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) 
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        sampler=val_sampler,
        num_workers=cfg.num_workers, 
        collate_fn=collate_fn,
        drop_last=False 
    )
    
    model = DiTPredictor(
    in_dim=cfg.in_dim,  
    hidden_dim=cfg.hidden_dim,
    emb_len=cfg.emb_len,   
    action_dim=cfg.action_dim,
    n_layers=cfg.n_layers,
    n_heads=cfg.n_heads, 
    dim_feedforward=cfg.dim_feedforward,
    cond_dim=cfg.cond_dim
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05*total_steps),  
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler('cuda')

    
    start_epoch = 0
    best_val = float("inf")
    if cfg.resume_checkpoint and os.path.isfile(cfg.resume_checkpoint):
        print(f"Rank {rank} | Loading checkpoint: {cfg.resume_checkpoint}")
        ckpt = torch.load(cfg.resume_checkpoint, map_location=device)
        
       
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            
        start_epoch = ckpt.get("epoch", -1) + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"Rank {rank} | Resumed from epoch {start_epoch}")

    
    predictor = DDP(model, device_ids=[local_rank])
    mse_loss = torch.nn.MSELoss()
    
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, cfg.num_epochs):
        train_sampler.set_epoch(epoch)
        predictor.train()  
        epoch_loss = 0.0
        for i, batch in enumerate(train_loader):
            
            z_t = batch["observation"].to(device)      # (B, 256, 2048)
            z_next = batch["next_observation"].to(device) # (B, 256, 2048)
            actions = batch["actions_sum"].to(device)  # (B, 6)
            #if is_main_process and i % 10 == 0:
            #   print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Loss: {loss.item() if 'loss' in locals() else '...'} ")
            with torch.amp.autocast('cuda'):
                
                pred_delta = predictor(z_t, actions) 
                z_predict = z_t + pred_delta
                loss = mse_loss(z_predict, z_next)

          
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if getattr(cfg, "min_lr", None) is not None:
                for group in optimizer.param_groups:
                    group["lr"] = max(group["lr"], cfg.min_lr)
            global_step += 1

            if is_main_process:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "step": global_step
                })
            
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        

        
        predictor.eval()
        local_val_loss = torch.tensor(0.0, device=device)      
        torch.cuda.synchronize() 
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])
        
        with torch.no_grad():
            for j, vb in enumerate(val_loader):
                vz_t = vb["observation"].to(device)
                vz_next = vb["next_observation"].to(device)
                va = vb["actions_sum"].to(device)
                
                with torch.amp.autocast('cuda'):
                    v_delta = predictor(vz_t, va)
                    v_loss = mse_loss(vz_t + v_delta, vz_next)
                
                local_val_loss += v_loss

        dist.barrier(device_ids=[local_rank]) 
        dist.all_reduce(local_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = local_val_loss.item() / (world_size * len(val_loader))

       
        if is_main_process:
            
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})
            
            print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            
            save_dict = {
                "epoch": epoch,
                "model_state_dict": predictor.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "scheduler_state_dict": scheduler.state_dict(), 
                "scaler_state_dict": scaler.state_dict(),       
                "best_val": min(avg_val_loss, best_val)
            }
            torch.save(save_dict, os.path.join(cfg.save_dir, "latest.pth"))
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                torch.save(save_dict, os.path.join(cfg.save_dir, "best_model.pth"))

    dist.destroy_process_group()



if __name__ == "__main__":
    train()