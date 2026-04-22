import os
import argparse
import math
import logging
import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Wrapper around PaliGemma to handle image embedding extraction
class PaliGemmaEncoderWrapper(nn.Module):
    
    def __init__(self, paligemma_variant: str, action_expert_variant: str, device: torch.device):
        super().__init__()
        logging.info(f"Loading PaliGemma variant: {paligemma_variant}")
        
        paligemma_config = _gemma.get_config(paligemma_variant)
        action_expert_config = _gemma.get_config(action_expert_variant)

        self.model = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True], 
            precision=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        self.device = device
        self.to(device)
        self.eval()

    @torch.no_grad()
    def get_embedding(self, img_tensors: torch.Tensor) -> torch.Tensor:
        
        #start_time = time.monotonic()
        img_tensors = img_tensors.to(device=self.device, dtype=self.model.precision)
        emb = self.model.embed_image(img_tensors)
        duration = (time.monotonic() - start_time) * 1000
        # logging.debug(f"PaliGemma embedding took: {duration:.2f} ms")
        
        return emb


def bytes_to_pil(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def get_image_bytes_field(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, dict) and "bytes" in val:
        return val["bytes"]
    return val

def _compute_embeddings_from_bytes_list(bytes_list, encoder, batch_size, device):
    n = len(bytes_list)
    embeddings = [None] * n
    
    from torchvision import transforms as T
    preprocess = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    with torch.no_grad():
        i = 0
        while i < n:
            idxs, imgs = [], []
            j = i
            while j < n and len(idxs) < batch_size:
                b = bytes_list[j]
                if b is not None:
                    try:
                        imgs.append(preprocess(bytes_to_pil(b)))
                        idxs.append(j)
                    except Exception as e:
                        logging.warning(f"Failed to decode image at index {j}: {e}")
                j += 1
            
            if idxs:
                imgs_t = torch.stack(imgs, dim=0).to(device)
                emb = encoder.get_embedding(imgs_t) 
                emb_np = emb.detach().cpu().float().numpy()
                for k, idx in enumerate(idxs):
                    embeddings[idx] = emb_np[k].tolist()
            i = j
    return embeddings

def process_episode(episode_path: Path, out_path: Path, encoder, batch_size, device, drop_bytes, force):
    if out_path.exists() and not force:
        logging.info(f"Skip existing: {out_path}")
        return

    df = pd.read_parquet(episode_path)
    has_img_emb = "image_embedding" in df.columns
    has_wrist_emb = "wrist_image_embedding" in df.columns

    if has_img_emb and has_wrist_emb and not force:
        logging.info(f"Embeddings present, copying: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return

    if "image" in df.columns and (not has_img_emb or force):
        img_bytes_list = df["image"].apply(get_image_bytes_field).tolist()
        logging.info(f"Encoding image for {episode_path.name}")
        df["image_embedding"] = _compute_embeddings_from_bytes_list(img_bytes_list, encoder, batch_size, device)

    if "wrist_image" in df.columns and (not has_wrist_emb or force):
        wrist_bytes_list = df["wrist_image"].apply(get_image_bytes_field).tolist()
        logging.info(f"Encoding wrist_image for {episode_path.name}")
        df["wrist_image_embedding"] = _compute_embeddings_from_bytes_list(wrist_bytes_list, encoder, batch_size, device)

    if drop_bytes:
        df = df.drop(columns=[c for c in ("image", "wrist_image") if c in df.columns])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=" ")
    parser.add_argument("--output", type=str, default=" ")
    parser.add_argument("--chunks", type=str, default="chunk-000")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--paligemma_variant", type=str, default="paligemma-3b-pt-224")
    parser.add_argument("--action_variant", type=str, default="gemma-2b-it")
    
    parser.add_argument("--drop_image_bytes", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    
    encoder = PaliGemmaEncoderWrapper(
        paligemma_variant=args.paligemma_variant,
        action_expert_variant=args.action_variant,
        device=device
    )

    root = Path(args.root)
    out_root = Path(args.output)
    chunk_list = [c.strip() for c in args.chunks.split(",") if c.strip()]

    for chunk in chunk_list:
        chunk_dir = root / "data" / chunk
        if not chunk_dir.exists(): continue

        files = sorted(list(chunk_dir.glob("episode_*.parquet")))
        for f in tqdm(files, desc=f"Processing {chunk}"):
            out_file = out_root / chunk / f.name
            try:
                process_episode(f, out_file, encoder, args.batch_size, device, args.drop_image_bytes, args.force)
            except Exception as e:
                logging.error(f"Error {f.name}: {e}")

if __name__ == "__main__":
    main()