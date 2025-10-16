"""Auto-generated from VisionNarrate.ipynb — module: models.py"""



# Standard imports (adjust as needed)
import os, math, json, random
from typing import Any, Dict, List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    pass



class CustomBLIPDecoderModel(nn.Module):
    def __init__(self, vision_model_name="Salesforce/blip-image-captioning-base",
                 vocab_size=30522, hidden_dim=512, encoder_dim=768,
                 num_layers=4, num_heads=8, dropout=0.1, max_length=100):
        super().__init__()
        # Used BLIP's vision model as encoder (only for projection in generate)
        self.vision_model = BlipVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # Decoder components
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.encoder_to_decoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length

    def forward(self, tgt_input_ids, encoder_outputs, tgt_mask=None, memory_key_padding_mask=None):
        tgt_embedded = self.token_embedding(tgt_input_ids) + self.position_embedding[:, :tgt_input_ids.size(1), :]
        tgt_embedded = tgt_embedded.transpose(0, 1)  # (tgt_seq_len, batch, hidden_dim)
        memory = self.encoder_to_decoder_proj(encoder_outputs)
        memory = memory.transpose(0, 1)  # (src_seq_len, batch, hidden_dim)
        out = self.transformer_decoder(tgt_embedded, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.fc_out(out.transpose(0, 1))  # (batch, tgt_seq_len, vocab_size)
        return logits

    def generate(self, encoder_outputs, tokenizer, device):
        batch_size = encoder_outputs.size(0)
        # Use T5's pad_token_id as start token for generation
        start_token = tokenizer.pad_token_id  
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)
        for _ in range(self.max_length):
            tgt_embedded = self.token_embedding(generated) + self.position_embedding[:, :generated.size(1), :]
            tgt_embedded = tgt_embedded.transpose(0, 1)
            memory = self.encoder_to_decoder_proj(encoder_outputs)
            memory = memory.transpose(0, 1)
            out = self.transformer_decoder(tgt_embedded, memory)
            logits = self.fc_out(out.transpose(0, 1))  
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
          
            if next_token.numel() == 1:
                if next_token.item() == tokenizer.eos_token_id:
                    break
            else:
                if next_token.eq(tokenizer.eos_token_id).all():
                    break
        return generated



class CustomBLIPFullModel(nn.Module):
    def __init__(self, vision_model_name="Salesforce/blip-image-captioning-base",
                 vocab_size=30522, hidden_dim=512, encoder_dim=768,
                 num_layers=4, num_heads=8, dropout=0.1, max_length=100):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(vision_model_name)
        self.vision_model = BlipVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.decoder = CustomBLIPDecoderModel(
            vision_model_name=vision_model_name,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_length=max_length
        )
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
    
    def forward(self, images, tgt_input_ids, tgt_mask=None):
        if isinstance(images, torch.Tensor):
            inputs = {"pixel_values": images}
        else:
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(next(self.vision_model.parameters()).device) for k, v in inputs.items()}
        vision_outputs = self.vision_model(**inputs)
        encoder_outputs = vision_outputs.last_hidden_state  # (B, src_seq_len, encoder_dim)
        logits = self.decoder(tgt_input_ids, encoder_outputs, tgt_mask=tgt_mask)
        return logits

    def generate(self, pixel_values, max_length=None, num_beams=5, early_stopping=True, use_cache=False):
        if not isinstance(pixel_values, torch.Tensor):
            inputs = self.processor(images=pixel_values, return_tensors="pt")
            inputs = {k: v.to(next(self.vision_model.parameters()).device) for k, v in inputs.items()}
        else:
            inputs = {"pixel_values": pixel_values}
        vision_outputs = self.vision_model(**inputs)
        encoder_outputs = vision_outputs.last_hidden_state  # (B, src_seq_len, encoder_dim)
        generated_ids = self.decoder.generate(encoder_outputs, self.tokenizer, device=pixel_values.device)
        return generated_ids

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
