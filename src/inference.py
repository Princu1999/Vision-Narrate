"""Auto-generated from VisionNarrate.ipynb — module: inference.py"""



# Standard imports (adjust as needed)
import os, math, json, random
from typing import Any, Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    pass



def generate_caption(image_path):
    """
    Given an image path, load the image, process it using the BLIP processor,
    generate a caption using the BLIP model, and return the decoded caption.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    image_np = np.clip(image_np, 0, 255).astype("uint8")
    image = Image.fromarray(image_np)
    
    inputs = blip_processor(images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = blip_model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True)
    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption



def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, 
                                    max_length=max_length, 
                                    num_beams=num_beams,
                                    early_stopping=True,
                                    use_cache=False)
        caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption
