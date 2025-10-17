import os
import glob
import string
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from ml2en import transliterate # NEW: Import the Malayalam to Manglish library

# Import the LipFormer model from your script
from my_model import LipFormer

# --- 1. Configuration ---
CONFIG = {
    "data": {
        "landmarks": "extracted_landmarks_model_ready",
        "lip_rois": "extracted_lip_crosssection",
        "transcripts": "D:/ADARSH/transcripts",
    },
    "checkpoint_dir": "checkpoints",
    "epochs": 50,
    "batch_size": 2, # Smaller batch size might be needed due to model complexity
    "learning_rate": 1e-4,
    "teacher_forcing_ratio": 0.5,
    "lambda_val": 0.7,
    "image_size": (80, 160),
}

# --- 2. Vocabulary Definitions ---

# --- Vocabulary for Manglish (the intermediate "Pinyin" representation) ---
MANGLISH_PAD_TOKEN = 0
MANGLISH_SOS_TOKEN = 1
MANGLISH_EOS_TOKEN = 2
MANGLISH_UNK_TOKEN = 3
MANGLISH_CHARS = string.ascii_lowercase + string.digits + " .'-" # Characters common in Manglish
manglish_to_int = {char: i + 4 for i, char in enumerate(MANGLISH_CHARS)}
manglish_to_int["<pad>"] = MANGLISH_PAD_TOKEN
manglish_to_int["<sos>"] = MANGLISH_SOS_TOKEN
manglish_to_int["<eos>"] = MANGLISH_EOS_TOKEN
manglish_to_int["<unk>"] = MANGLISH_UNK_TOKEN
int_to_manglish = {i: char for char, i in manglish_to_int.items()}
MANGLISH_VOCAB_SIZE = len(manglish_to_int)

# --- Vocabulary for Malayalam (the final "Character" representation) ---
# This will be built dynamically from the training data
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
int_to_malayalam = {} # Will be populated after building vocab

def build_malayalam_vocab(transcript_dir):
    """Scans all transcript files to build the Malayalam character vocabulary."""
    vocab = set()
    transcript_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
    for file_path in tqdm(transcript_files, desc="Building Malayalam Vocab"):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        full_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
        vocab.update(list(full_text))
    
    # Add unique characters to the mapping
    for i, char in enumerate(sorted(list(vocab))):
        malayalam_to_int[char] = i + 4 # Start after special tokens
        
    # Create the reverse mapping
    global int_to_malayalam
    int_to_malayalam = {i: char for char, i in malayalam_to_int.items()}
    
    return len(malayalam_to_int)

# --- 3. Custom PyTorch Dataset ---
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

        landmarks = torch.from_numpy(np.load(sample["landmarks"])).float()

        roi_paths = sorted(glob.glob(os.path.join(sample["rois"], "*.png")))
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
        
        # NEW: Transliterate Malayalam to Manglish
        manglish_text = transliterate(malayalam_text).lower()

        # Tokenize Malayalam
        mal_tokens = [MALAYALAM_SOS_TOKEN]
        mal_tokens.extend([malayalam_to_int.get(c, MALAYALAM_UNK_TOKEN) for c in malayalam_text])
        mal_tokens.append(MALAYALAM_EOS_TOKEN)
        mal_label = torch.tensor(mal_tokens, dtype=torch.long)

        # Tokenize Manglish
        man_tokens = [MANGLISH_SOS_TOKEN]
        man_tokens.extend([manglish_to_int.get(c, MANGLISH_UNK_TOKEN) for c in manglish_text])
        man_tokens.append(MANGLISH_EOS_TOKEN)
        man_label = torch.tensor(man_tokens, dtype=torch.long)

        return {"video": video_tensor, "landmarks": landmarks, "malayalam_label": mal_label, "manglish_label": man_label}

# --- 4. Collate Function for Padding ---
def collate_fn(batch):
    videos = [item['video'] for item in batch]
    landmarks = [item['landmarks'] for item in batch]
    mal_labels = [item['malayalam_label'] for item in batch]
    man_labels = [item['manglish_label'] for item in batch]

    padded_videos = pad_sequence([v.permute(1, 0, 2, 3) for v in videos], batch_first=True, padding_value=0).permute(0, 2, 1, 3, 4)
    padded_landmarks = pad_sequence(landmarks, batch_first=True, padding_value=0)
    padded_mal_labels = pad_sequence(mal_labels, batch_first=True, padding_value=MALAYALAM_PAD_TOKEN)
    padded_man_labels = pad_sequence(man_labels, batch_first=True, padding_value=MANGLISH_PAD_TOKEN)
    
    return {"video": padded_videos, "landmarks": padded_landmarks, "malayalam_label": padded_mal_labels, "manglish_label": padded_man_labels}

# --- 5. Main Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # Build the Malayalam vocabulary from the dataset
    MALAYALAM_VOCAB_SIZE = build_malayalam_vocab(CONFIG["data"]["transcripts"])
    print(f"Built Malayalam vocabulary with {MALAYALAM_VOCAB_SIZE} unique characters.")
    print(f"Manglish vocabulary size: {MANGLISH_VOCAB_SIZE}")

    dataset = LipReadingDataset(
        CONFIG["data"]["landmarks"], CONFIG["data"]["lip_rois"], CONFIG["data"]["transcripts"], CONFIG["image_size"]
    )
    if len(dataset) == 0:
        print("No data found. Please check configuration paths.")
        return

    data_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = LipFormer(
        num_pinyins=MANGLISH_VOCAB_SIZE, # Pinyin decoder learns Manglish
        num_chars=MALAYALAM_VOCAB_SIZE,  # Character decoder learns Malayalam
        device=device
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters.")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    pinyin_loss_fn = nn.CrossEntropyLoss(ignore_index=MANGLISH_PAD_TOKEN)
    char_loss_fn = nn.CrossEntropyLoss(ignore_index=MALAYALAM_PAD_TOKEN)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch in progress_bar:
            videos, landmarks, mal_targets, man_targets = [d.to(device) for d in batch.values()]
            
            optimizer.zero_grad()
            pinyin_preds, char_preds = model(videos, landmarks, pinyin_targets=man_targets, char_targets=mal_targets, teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"])
            
            loss_pinyin = pinyin_loss_fn(pinyin_preds.view(-1, MANGLISH_VOCAB_SIZE), man_targets.view(-1))
            loss_char = char_loss_fn(char_preds.view(-1, MALAYALAM_VOCAB_SIZE), mal_targets.view(-1))
            total_loss = CONFIG["lambda_val"] * loss_pinyin + (1 - CONFIG["lambda_val"]) * loss_char
            
            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        epoch_loss = total_loss.item()
        print(f"Epoch {epoch+1} finished. Final Batch Loss: {epoch_loss:.4f}")
        checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"lipformer_malayalam_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()

