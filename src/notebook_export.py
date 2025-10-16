"""Auto-generated from VisionNarrate.ipynb — module: notebook_export.py"""

# Residual code exported verbatim from the notebook



# --- Begin residual blocks ---
import wandb
import os
import wandb
import numpy as np
import logging
import os
import warnings
import random
import torch
import torch.nn as nn   
from torch.utils.data import Dataset, DataLoader, random_split 
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration,get_linear_schedule_with_warmup
from transformers import T5ForConditionalGeneration, T5Tokenizer,BlipVisionModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from nltk.translate.meteor_score import meteor_score
from torch.optim import AdamW
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import os
import json
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download("omw-1.4")

wandb.login()

images_dir = "/csehome/m24mac015/Dataset/images"
descriptions_dir = "/csehome/m24mac015/Dataset/descriptions"
test_img_dir = "/csehome/m24mac015/Dataset/test"
model_saver_dir = "/csehome/m24mac015/Dataset/model"
result_dir = "/csehome/m24mac015/Dataset/result"
augmented_captions_path = "/csehome/m24mac015/Dataset/descriptions/augmented_captions.json"
merged_descriptions_dir = "/csehome/m24mac015/Dataset/merged_descriptions"
os.makedirs(merged_descriptions_dir, exist_ok=True)


model_path = os.path.join(model_saver_dir, "blip-finetuned")
processor_path = os.path.join(model_saver_dir, "blip-finetuned")

EPOCH = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
SAMPLE = 1000  
MAX_LENGTH = 150
NUM_BEAMS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# BLIP Caption Generator Setup
# ---------------------------
# Load BLIP processor and model for captioning 
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
blip_model.config.use_cache = False  # Disable caching for gradient checkpointing safety
blip_model.to(device)

epoch = EPOCH
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
sample = SAMPLE  
max_length = MAX_LENGTH
num_beams = NUM_BEAMS

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

# ---------------------------
# T5 Paraphraser Setup
# ---------------------------
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


paraphraser = CaptionParaphraser()
augmented_captions = {}

print("Generating captions and paraphrases for each image...")
for img_file in tqdm(os.listdir(images_dir)):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        base_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(images_dir, img_file)
        try:
            caption = generate_caption(image_path)
            paraphrases = paraphraser.paraphrase(caption, num_return_sequences=2)
            augmented_captions[base_name] = [caption] + paraphrases
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

augmented_caption_file = "augmented_captions.json"
with open(augmented_caption_file, "w", encoding="utf-8") as f:
    json.dump(augmented_captions, f, ensure_ascii=False, indent=2)

print(f"Augmented captions saved to {augmented_caption_file}")

print("Loading original descriptions...")
original_descriptions = {}
for fname in os.listdir(descriptions_dir):
    if fname.endswith(".txt") and fname != "augmented_captions.json":
        image_id = fname.replace(".txt", "")
        with open(os.path.join(descriptions_dir, fname), "r") as f:
            lines = f.read().strip().split("\n")
            if lines:
                original_descriptions[image_id] = lines[0]

print(f"Loaded {len(original_descriptions)} original descriptions.")

# Load augmented captions
print("Loading augmented captions...")
with open(augmented_captions_path, "r") as f:
    augmented_captions = json.load(f)

saved_count = 0

print("Merging original description with first augmented caption ")
for image_id, original_text in original_descriptions.items():
    aug_list = augmented_captions.get(image_id, [])
    if not aug_list:
        print(f"No augmentations for image {image_id}, skipping.")
        continue

    first_caption = aug_list[0]

    with open(os.path.join(merged_descriptions_dir, f"{image_id}.txt"), "w") as f:
        f.write(original_text.strip() + "\n")
        f.write(first_caption.strip())
    print(f"[✓] Saved merged description for: {image_id}")
    saved_count += 1

print(f"\nDone. Total merged files saved: {saved_count}")

# --- Logging & Environment Setup ---
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress non-critical Transformers warnings
warnings.filterwarnings("ignore", message="`use_cache=True` is incompatible with gradient checkpointing")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Dataset Definition ---

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


epoch = EPOCH
batch_size = BATCH_SIZE
learning_rate = LEARNING_RATE
sample = SAMPLE  
max_length = MAX_LENGTH
num_beams = NUM_BEAMS
# --- Model & Processor Setup ---
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
model.config.use_cache = False  # Disable caching to avoid issues with gradient checkpointing
model.gradient_checkpointing_enable()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Dataset & Data Splitting ---
full_dataset = BLIPDataset(images_dir, descriptions_dir, processor, max_length=max_length)
all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
sample_indices = all_indices[:sample]  
sample_dataset = torch.utils.data.Subset(full_dataset, sample_indices)
train_size = int(0.95 * sample)
val_size = sample - train_size
train_dataset, val_dataset = random_split(sample_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- Optimizer & Mixed Precision Setup ---
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
scaler = torch.amp.GradScaler() 

# --- Initialize wandb ---
wandb.init(project="BLIP-FineTuning_Final", name="blip_finetune_run", config={
    "num_epochs": epoch,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "sample_size": sample,
    "max_length": max_length,
    "num_beams": num_beams
})

# --- Training Loop ---
smoothie = SmoothingFunction().method4

for param in model.vision_model.parameters():
    param.requires_grad = False
    
for param in model.text_decoder.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params}/{total_params}")

for ep in range(epoch):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {ep+1}")
    for batch in loop:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type):
            outputs = model(**batch, labels=batch["input_ids"])
            loss_ce = outputs.loss
        loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        wandb.log({"train_loss": loss.item(), "epoch": ep + 1})
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {ep+1} completed. Average Training Loss: {avg_loss:.4f}")
    wandb.log({"epoch": ep+1, "avg_train_loss": avg_loss})

    # --- Validation with Loss, BLEU, ROUGE, BERTScore, and METEOR ---
    model.eval()
    bleu_scores = [0, 0, 0, 0]
    total_rouge1 = total_rouge2 = total_rougeL = 0
    meteor_total = 0
    n_val = 0
    generated_texts_val = []
    reference_texts_val = []
    val_loss = 0
    val_steps = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type=device.type):
                outputs = model(**batch, labels=batch["input_ids"])
            val_loss += outputs.loss.item()
            val_steps += 1

            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                use_cache=False
            )
            for i in range(len(generated_ids)):
                candidate = processor.decode(generated_ids[i], skip_special_tokens=True)
                reference = processor.decode(batch["input_ids"][i], skip_special_tokens=True)
                generated_texts_val.append(candidate)
                reference_texts_val.append(reference)
                candidate_tokens = candidate.split()
                reference_tokens = reference.split()
                bleu_scores = [
                    bleu_scores[j] + sentence_bleu(
                        [reference_tokens],
                        candidate_tokens,
                        weights=w,
                        smoothing_function=smoothie
                    )
                    for j, w in enumerate([(1, 0, 0, 0), (0.5, 0.5, 0, 0),
                                             (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)])
                ]
                scores = scorer.score(reference, candidate)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                total_rougeL += scores['rougeL'].fmeasure
                meteor_total += meteor_score([word_tokenize(reference)], word_tokenize(candidate))
                n_val += 1

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

    if n_val > 0:
        avg_bleu = [s / n_val for s in bleu_scores]
        avg_rouge1 = total_rouge1 / n_val
        avg_rouge2 = total_rouge2 / n_val
        avg_rougeL = total_rougeL / n_val
    else:
        avg_bleu = [0, 0, 0, 0]
        avg_rouge1 = avg_rouge2 = avg_rougeL = 0

    # --- Computing BERTScore ---
    if generated_texts_val and reference_texts_val:
        P, R, F1 = bert_score_fn(generated_texts_val, reference_texts_val, lang="en", verbose=True)
        avg_bert_precision = P.mean().item()
        avg_bert_recall = R.mean().item()
        avg_bert_f1 = F1.mean().item()
    else:
        avg_bert_precision = avg_bert_recall = avg_bert_f1 = 0

    # --- Computing METEOR ---
    avg_meteor = meteor_total / n_val if n_val > 0 else 0

    # Log all validation metrics to wandb
    wandb.log({
        "BLEU-1": avg_bleu[0],
        "BLEU-2": avg_bleu[1],
        "BLEU-3": avg_bleu[2],
        "BLEU-4": avg_bleu[3],
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL,
        "BERTScore_P": avg_bert_precision,
        "BERTScore_R": avg_bert_recall,
        "BERTScore_F1": avg_bert_f1,
        "METEOR": avg_meteor,
        "val_loss": avg_val_loss,
        "epoch": ep + 1
    })
    print(f"Epoch {ep+1} Validation:\n"
          f"  BLEU: {avg_bleu}\n"
          f"  ROUGE: [R-1: {avg_rouge1:.4f}, R-2: {avg_rouge2:.4f}, R-L: {avg_rougeL:.4f}]\n"
          f"  BERTScore F1: {avg_bert_f1:.4f}, METEOR: {avg_meteor:.4f}, Val Loss: {avg_val_loss:.4f}")
    model.train()

# --- Saved the Fine-Tuned Model & Processor ---
model.save_pretrained(model_path)
processor.save_pretrained(processor_path)
wandb.save("blip-finetuned/*")

########################################
# Dataset Definition
########################################
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

########################################
# Custom Decoder (Transformer-based)
########################################
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

########################################
# Full Custom Model: Wrap Vision Encoder + Custom Decoder
########################################
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

########################################
# Instantiate the Model and Prepare Data
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
full_dataset = BLIPDataset(images_dir, descriptions_dir, processor, max_length=max_length)
all_indices = list(range(len(full_dataset)))
random.shuffle(all_indices)
sample_indices = all_indices[:sample]
sample_dataset = torch.utils.data.Subset(full_dataset, sample_indices)
train_size = int(0.95 * sample)
val_size = sample - train_size
train_dataset, val_dataset = random_split(sample_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Instantiate our complete custom model
custom_model = CustomBLIPFullModel(max_length=max_length).to(device)

########################################
# Optimizer, Scheduler, and W&B Logging Setup
########################################
optimizer = AdamW(custom_model.parameters(), lr=learning_rate)
num_training_steps = len(train_loader) * epoch
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)
wandb.init(project="BLIP-FineTuning_custom", config={
    "num_epochs": epoch,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "sample_size": sample,
    "model_architecture": "CustomBLIPFullModel",
    "augment": False
})
smoothie = SmoothingFunction().method4
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

########################################
# Training Loop
########################################
accumulation_steps = 1
best_bleu4 = 0
patience = 10
no_improve_epochs = 0

loss_fn = nn.CrossEntropyLoss(ignore_index=custom_model.tokenizer.pad_token_id)

custom_model.train()
for ep in range(epoch):
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {ep+1}")
    for batch_idx, batch in enumerate(loop):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        logits = custom_model(images=batch["pixel_values"], tgt_input_ids=batch["input_ids"])
        loss = loss_fn(logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        wandb.log({"train_loss": loss.item(), "epoch": ep + 1})
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {ep+1} completed. Average Training Loss: {avg_loss:.4f}")
    wandb.log({"epoch": ep+1, "avg_train_loss": avg_loss})
    
    # --- Validation ---
    custom_model.eval()
    bleu_scores = [0, 0, 0, 0]
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    n_val = 0
    generated_texts_val = []
    reference_texts_val = []
    val_loss = 0
    val_steps = 0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = custom_model(images=batch["pixel_values"], tgt_input_ids=batch["input_ids"])
            val_loss += loss_fn(logits.view(-1, logits.size(-1)), batch["input_ids"].view(-1)).item()
            val_steps += 1
            generated_ids = custom_model.generate(
                pixel_values=batch["pixel_values"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                use_cache=False
            )
            for i in range(len(generated_ids)):
                candidate = custom_model.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                reference = processor.decode(batch["input_ids"][i], skip_special_tokens=True)
                generated_texts_val.append(candidate)
                reference_texts_val.append(reference)
                candidate_tokens = candidate.split()
                reference_tokens = reference.split()
                for j, weights in enumerate([
                    (1, 0, 0, 0),
                    (0.5, 0.5, 0, 0),
                    (0.33, 0.33, 0.33, 0),
                    (0.25, 0.25, 0.25, 0.25)
                ]):
                    bleu_scores[j] += sentence_bleu(
                        [reference_tokens], candidate_tokens,
                        weights=weights,
                        smoothing_function=smoothie
                    )
                meteor_scores.append(meteor_score([reference_tokens], candidate_tokens))
                scores = scorer.score(reference, candidate)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
                n_val += 1
    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    if n_val > 0:
        avg_bleu = [s / n_val for s in bleu_scores]
        avg_rouge1 = sum(rouge1_scores) / n_val
        avg_rouge2 = sum(rouge2_scores) / n_val
        avg_rougeL = sum(rougeL_scores) / n_val
    else:
        avg_bleu = [0, 0, 0, 0]
        avg_rouge1 = avg_rouge2 = avg_rougeL = 0
    try:
        from bert_score import score as bert_score_fn
        if generated_texts_val and reference_texts_val:
            P, R, F1 = bert_score_fn(generated_texts_val, reference_texts_val, lang="en", verbose=True)
            avg_bert_precision = P.mean().item()
            avg_bert_recall = R.mean().item()
            avg_bert_f1 = F1.mean().item()
        else:
            avg_bert_precision = avg_bert_recall = avg_bert_f1 = 0
    except ImportError:
        avg_bert_precision = avg_bert_recall = avg_bert_f1 = 0
    avg_meteor = sum(meteor_scores) / n_val if n_val > 0 else 0

    wandb.log({
        "BLEU-1": avg_bleu[0],
        "BLEU-2": avg_bleu[1],
        "BLEU-3": avg_bleu[2],
        "BLEU-4": avg_bleu[3],
        "METEOR": avg_meteor,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL,
        "BERTScore_P": avg_bert_precision,
        "BERTScore_R": avg_bert_recall,
        "BERTScore_F1": avg_bert_f1,
        "val_loss": avg_val_loss,
        "epoch": ep + 1
    })
    print(f"Epoch {ep+1} Validation:\n  BLEU: {avg_bleu}\n  ROUGE: [R-1: {avg_rouge1:.4f}, R-2: {avg_rouge2:.4f}, R-L: {avg_rougeL:.4f}]\n  BERTScore F1: {avg_bert_f1:.4f}, METEOR: {avg_meteor:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    if n_val > 0:
        if avg_bleu[3] > best_bleu4:
            best_bleu4 = avg_bleu[3]
            custom_model.save_pretrained(f"{model_path}_best")
            processor.save_pretrained(f"{processor_path}_best")
            print(f"New best model saved with BLEU-4: {avg_bleu[3]:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {ep+1} epochs.")
            break
    custom_model.train()

custom_model.save_pretrained(model_path)
processor.save_pretrained(processor_path)
wandb.save("blip-finetuned/*")

os.makedirs(result_dir, exist_ok=True)

processor = BlipProcessor.from_pretrained(processor_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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


captions = {}
for filename in os.listdir(test_img_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        img_path = os.path.join(test_img_dir, filename)
        print(f"Processing: {filename}...")
        caption = generate_caption(img_path)
        captions[filename] = caption

# Save the predicted captions to a text file in the result directory
result_file = os.path.join(result_dir, "predictions.txt")
with open(result_file, "w") as f:
    for img_name, caption in captions.items():
        f.write(f"{img_name}: {caption}\n")

print(f"Predictions saved to {result_file}")
# --- End residual blocks ---
