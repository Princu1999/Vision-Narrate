"""Auto-generated from VisionNarrate.ipynb — module: utils.py"""



# Standard imports (adjust as needed)
import os, math, json, random
from typing import Any, Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    pass



class CaptionParaphraser:
    def __init__(self, model_name="ramsrigouthamg/t5_paraphraser", device=device):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def paraphrase(self, text, num_return_sequences=2, num_beams=5):
        input_text = f"paraphrase: {text} </s>"
        encoding = self.tokenizer.encode_plus(
            input_text,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=128,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
        paraphrases = [self.tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                       for out in outputs]
        return paraphrases
