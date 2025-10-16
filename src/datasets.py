"""Auto-generated from VisionNarrate.ipynb — module: datasets.py"""



# Standard imports (adjust as needed)
import os, math, json, random
from typing import Any, Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    pass



class BLIPDataset(Dataset):
    def __init__(self, images_dir, descriptions_dir, processor, max_length=77):
        """
        Each file in descriptions_dir is expected to contain at least two lines:
            - Line 1: A detailed description.
            - Line 2: A caption.
        These two lines are concatenated using " [SEP] " as a separator.
        """
        self.images_dir = images_dir
        self.descriptions_dir = descriptions_dir  # Merged descriptions
        self.processor = processor
        self.max_length = max_length
        self.image_files = [f for f in os.listdir(images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        description_path = os.path.join(self.descriptions_dir, os.path.splitext(image_name)[0] + ".txt")
        
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = np.clip(image_np, 0, 255).astype("uint8")
        image = Image.fromarray(image_np)
        
        with open(description_path, "r", encoding="utf-8") as f:
            merged_text = f.read().strip()
        lines = merged_text.splitlines()
        if len(lines) >= 2:
            target_text = "<DESC> " + lines[0].strip() + " <CAP> " + lines[1].strip()
        else:
            target_text = merged_text
        
        inputs = self.processor(
            images=image,
            text=target_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs



class BLIPDataset(Dataset):
    def __init__(self, images_dir, descriptions_dir, processor, max_length=77):
        self.images_dir = images_dir
        self.descriptions_dir = descriptions_dir
        self.processor = processor
        self.max_length = max_length
        self.image_files = [f for f in os.listdir(images_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        description_path = os.path.join(self.descriptions_dir, os.path.splitext(image_name)[0] + ".txt")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np = np.clip(image_np, 0, 255).astype("uint8")
        image = Image.fromarray(image_np)
        with open(description_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
        inputs = self.processor(
            images=image,
            text=description,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
