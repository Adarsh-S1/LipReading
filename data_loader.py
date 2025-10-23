import os
import glob
import string
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from ml2en import ml2en # Import the Malayalam to Manglish library

# Import the LipFormer model from your script
from my_model import LipFormer

# --- 1. Configuration ---
CONFIG = {
    "data": {
        "landmarks": "D:/ADARSH/extracted_landmarks_model_ready",
        "lip_rois": "D:/ADARSH/extracted_lip_crosssection",
        "transcripts": "D:/ADARSH/transcripts",
    },
    "checkpoint_dir": "checkpoints",
    "epochs": 1, # Increased epochs for meaningful training
    "batch_size": 1, # You can try 1 if memory issues persist
    "learning_rate": 1e-4,
    "teacher_forcing_ratio": 0.5,
    "lambda_val": 0.7,
    "image_size": (80, 160),
    "validation_split": 0.1, # 10% of data for validation
}

# --- 2. Vocabulary Definitions ---

# --- Vocabulary for Manglish ---
MANGLISH_PAD_TOKEN = 0
MANGLISH_SOS_TOKEN = 1
MANGLISH_EOS_TOKEN = 2
MANGLISH_UNK_TOKEN = 3
MANGLISH_CHARS = string.ascii_lowercase + string.digits + " .'-"
manglish_to_int = {char: i + 4 for i, char in enumerate(MANGLISH_CHARS)}
manglish_to_int["<pad>"] = MANGLISH_PAD_TOKEN
manglish_to_int["<sos>"] = MANGLISH_SOS_TOKEN
manglish_to_int["<eos>"] = MANGLISH_EOS_TOKEN
manglish_to_int["<unk>"] = MANGLISH_UNK_TOKEN
int_to_manglish = {i: char for char, i in manglish_to_int.items()}
MANGLISH_VOCAB_SIZE = len(manglish_to_int)

# --- Vocabulary for Malayalam ---
MALAYALAM_PAD_TOKEN = 0
MALAYALAM_SOS_TOKEN = 1
MALAYALAM_EOS_TOKEN = 2
MALAYALAM_UNK_TOKEN = 3
malayalam_to_int = {
    "<pad>": MALAYALAM_PAD_TOKEN,
    "<sos>": MALAYALAM_SOS_TOKEN,
    "<eos>": MALAYALAM_EOS_TOKEN,
    "<unk>": MALAYALAM_UNK_TOKEN,
}
int_to_malayalam = {}

def build_malayalam_vocab(transcript_dir):
    """Scans all transcript files to build the Malayalam character vocabulary."""
    vocab = set()
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
    for file_path in tqdm(transcript_files, desc="Building Malayalam Vocab"):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        full_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
        vocab.update(list(full_text))
    
    for i, char in enumerate(sorted(list(vocab))):
        malayalam_to_int[char] = i + 4
        
    global int_to_malayalam
    int_to_malayalam = {i: char for char, i in malayalam_to_int.items()}
    
    return len(malayalam_to_int)

# --- 3. Custom PyTorch Dataset with Robust Checks ---
class LipReadingDataset(Dataset):
    def __init__(self, landmark_dir, lip_roi_dir, transcript_dir, img_size):
        self.img_size = img_size
        self.samples = []

        print("Searching for data samples...")
        landmark_files = sorted(glob.glob(os.path.join(landmark_dir, "*.npy")))
        
        for landmark_path in tqdm(landmark_files, desc="Matching data files"):
            base_name = os.path.basename(landmark_path).replace("_landmarks.npy", "")
            roi_dir = os.path.join(lip_roi_dir, base_name)
            transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")

            if os.path.isdir(roi_dir) and os.path.exists(transcript_path):
                self.samples.append({
                    "landmarks": landmark_path,
                    "rois": roi_dir,
                    "transcript": transcript_path,
                })
        
        print(f"Found {len(self.samples)} complete data samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # MODIFIED: Load landmarks with error handling
        try:
            landmarks_np = np.load(sample["landmarks"])
            if landmarks_np.shape[0] == 0:
                print(f"\n[WARNING] Skipping empty landmark file: {sample['landmarks']}")
                return None
            landmarks = torch.from_numpy(landmarks_np).float()
        except Exception as e:
            print(f"\n[WARNING] Skipping corrupted landmark file: {sample['landmarks']}. Error: {e}")
            return None

        roi_paths = sorted(glob.glob(os.path.join(sample["rois"], "*.png")))
        
        # MODIFIED: Check for empty ROI directory or length mismatch
        if not roi_paths:
            print(f"\n[WARNING] Skipping sample with no ROI frames: {sample['rois']}")
            return None
        if len(roi_paths) != landmarks.shape[0]:
            print(f"\n[WARNING] Mismatch: Frames({len(roi_paths)}) vs Landmarks({landmarks.shape[0]}) for {os.path.basename(sample['landmarks'])}. Skipping.")
            return None

        frames = []
        for frame_path in roi_paths:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                frames.append(frame / 255.0)
        video_tensor = torch.from_numpy(np.array(frames)).float().unsqueeze(0)

        with open(sample["transcript"], 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        malayalam_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
        
        manglish_text = ml2en.transliterate(malayalam_text).lower()

        mal_tokens = [MALAYALAM_SOS_TOKEN] + [malayalam_to_int.get(c, MALAYALAM_UNK_TOKEN) for c in malayalam_text] + [MALAYALAM_EOS_TOKEN]
        mal_label = torch.tensor(mal_tokens, dtype=torch.long)

        man_tokens = [MANGLISH_SOS_TOKEN] + [manglish_to_int.get(c, MANGLISH_UNK_TOKEN) for c in manglish_text] + [MANGLISH_EOS_TOKEN]
        man_label = torch.tensor(man_tokens, dtype=torch.long)

        return {"video": video_tensor, "landmarks": landmarks, "malayalam_label": mal_label, "manglish_label": man_label}

# --- 4. Collate Function to Handle 'None' Samples ---
def collate_fn(batch):
    # MODIFIED: Filter out None values from failed loads in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    videos = [item['video'] for item in batch]
    landmarks = [item['landmarks'] for item in batch]
    mal_labels = [item['malayalam_label'] for item in batch]
    man_labels = [item['manglish_label'] for item in batch]

    padded_videos = pad_sequence([v.permute(1, 0, 2, 3) for v in videos], batch_first=True, padding_value=0).permute(0, 2, 1, 3, 4)
    padded_landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0)
    padded_mal_labels = pad_sequence(mal_labels, batch_first=True, padding_value=MALAYALAM_PAD_TOKEN)
    padded_man_labels = pad_sequence(man_labels, batch_first=True, padding_value=MANGLISH_PAD_TOKEN)
    
    return {"video": padded_videos, "landmarks": padded_landmarks, "malayalam_label": padded_mal_labels, "manglish_label": padded_man_labels}


# --- NEW HELPER FUNCTION ---
def tokens_to_text(token_indices, int_to_vocab_map):
    """Converts a tensor of token indices into a human-readable string."""
    text = ""
    for token_idx in token_indices:
        idx = token_idx.item()
        char = int_to_vocab_map.get(idx)
        
        # Stop at the first EOS or PAD token
        if char == "<eos>" or char == "<pad>":
            break
        # Skip the SOS token
        if char == "<sos>":
            continue
            
        if char is not None:
            text += char
    return text
# --- END NEW HELPER FUNCTION ---

# --- 5. Main Training & Validation Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    MALAYALAM_VOCAB_SIZE = build_malayalam_vocab(CONFIG["data"]["transcripts"])
    print(f"Built Malayalam vocabulary with {MALAYALAM_VOCAB_SIZE} unique characters.")
    print(f"Manglish vocabulary size: {MANGLISH_VOCAB_SIZE}")

    dataset = LipReadingDataset(
        CONFIG["data"]["landmarks"], CONFIG["data"]["lip_rois"], CONFIG["data"]["transcripts"], CONFIG["image_size"]
    )
    if len(dataset) == 0:
        print("No data found. Please check configuration paths.")
        return

    # MODIFIED: Split dataset into training and validation sets
    val_size = int(CONFIG["validation_split"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Data split into {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = LipFormer(
        num_pinyins=MANGLISH_VOCAB_SIZE,
        num_chars=MALAYALAM_VOCAB_SIZE,
        device=device
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters.")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    pinyin_loss_fn = nn.CrossEntropyLoss(ignore_index=MANGLISH_PAD_TOKEN)
    char_loss_fn = nn.CrossEntropyLoss(ignore_index=MALAYALAM_PAD_TOKEN)
    scaler = torch.amp.GradScaler('cuda') # For mixed precision

    best_val_loss = float('inf')

    for epoch in range(CONFIG["epochs"]):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for batch in train_bar:
            if batch is None: continue # Skip empty batches from collate_fn
            
            videos, landmarks, mal_targets, man_targets = [d.to(device) for d in batch.values()]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pinyin_preds, char_preds = model(videos, landmarks, pinyin_targets=man_targets, char_targets=mal_targets, teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"])
                loss_pinyin = pinyin_loss_fn(pinyin_preds.view(-1, MANGLISH_VOCAB_SIZE), man_targets.view(-1))
                loss_char = char_loss_fn(char_preds.view(-1, MALAYALAM_VOCAB_SIZE), mal_targets.view(-1))
                total_loss = CONFIG["lambda_val"] * loss_pinyin + (1 - CONFIG["lambda_val"]) * loss_char
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        outputs_shown = 0 # <-- MODIFIED: Counter for validation outputs
        MAX_VAL_OUTPUTS = 5 # <-- MODIFIED: Max outputs to show
        print(f"\n--- Epoch {epoch+1} Validation Outputs ---")

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                if batch is None: continue

                videos, landmarks, mal_targets, man_targets = [d.to(device) for d in batch.values()]
                
                # No teacher forcing during validation
                pinyin_preds, char_preds = model(videos, landmarks, pinyin_targets=None, char_targets=None, teacher_forcing_ratio=0.0)

                # --- START OF FIX ---
                # Get the sequence length from both predictions and targets
                pinyin_pred_len = pinyin_preds.size(1)
                pinyin_target_len = man_targets.size(1)
                # Find the minimum length to compare
                min_pinyin_len = min(pinyin_pred_len, pinyin_target_len)
                
                char_pred_len = char_preds.size(1)
                char_target_len = mal_targets.size(1)
                # Find the minimum length to compare
                min_char_len = min(char_pred_len, char_target_len)

                # Slice *both* tensors to the minimum length
                pinyin_preds_sliced = pinyin_preds[:, :min_pinyin_len, :]
                man_targets_sliced = man_targets[:, :min_pinyin_len]
                
                char_preds_sliced = char_preds[:, :min_char_len, :]
                mal_targets_sliced = mal_targets[:, :min_char_len]

                # Calculate loss on the *identically-sized* sliced tensors
                loss_pinyin = pinyin_loss_fn(pinyin_preds_sliced.view(-1, MANGLISH_VOCAB_SIZE), man_targets_sliced.view(-1))
                loss_char = char_loss_fn(char_preds_sliced.view(-1, MALAYALAM_VOCAB_SIZE), mal_targets_sliced.view(-1))
                # --- END OF FIX ---

                total_loss = CONFIG["lambda_val"] * loss_pinyin + (1 - CONFIG["lambda_val"]) * loss_char

                val_loss += total_loss.item()
                val_bar.set_postfix(loss=f"{total_loss.item():.4f}")

                # --- START: MODIFIED section to show outputs ---
                if outputs_shown < MAX_VAL_OUTPUTS:
                    # Get predicted indices by taking argmax along the vocabulary dimension
                    pred_pinyin_indices = torch.argmax(pinyin_preds, dim=2)
                    pred_char_indices = torch.argmax(char_preds, dim=2)

                    # We'll just show the first item in the batch (index 0)
                    # Convert token indices to text
                    true_man_text = tokens_to_text(man_targets[0], int_to_manglish)
                    pred_man_text = tokens_to_text(pred_pinyin_indices[0], int_to_manglish)
                    
                    true_mal_text = tokens_to_text(mal_targets[0], int_to_malayalam)
                    pred_mal_text = tokens_to_text(pred_char_indices[0], int_to_malayalam)

                    # Print in a neat format
                    print(f"\n--- Sample {outputs_shown + 1} ---")
                    print(f"  [Manglish] GT:   {true_man_text}")
                    print(f"  [Manglish] Pred: {pred_man_text}")
                    print(f"  [Malayalam] GT:   {true_mal_text}")
                    print(f"  [Malayalam] Pred: {pred_mal_text}")
                    
                    outputs_shown += 1
                    
                    if outputs_shown == MAX_VAL_OUTPUTS:
                        print("--------------------------------------\n") # Footer
                # --- END: MODIFIED section ---

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # MODIFIED: Save only the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "lipformer_malayalam_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"🎉 New best model saved to {checkpoint_path} with Val Loss: {best_val_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()