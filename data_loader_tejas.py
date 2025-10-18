import os
import glob
import cv2
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from ml2en import ml2en  # Malayalam → Manglish transliteration
from my_model import LipFormer  # Import your LipFormer model


# ==============================
# 1. CONFIGURATION
# ==============================
CONFIG = {
    "data": {
        "landmarks": "D:/ADARSH/extracted_landmarks_model_ready",
        "lip_rois": "D:/ADARSH/extracted_lip_crosssection",
        "transcripts": "D:/ADARSH/transcripts",
    },
    "checkpoint_dir": "checkpoints",
    "epochs": 1,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "teacher_forcing_ratio": 0.5,
    "lambda_val": 0.7,
    "image_size": (80, 160),
    "validation_split": 0.1,
}


# ==============================
# 2. VOCABULARY DEFINITIONS
# ==============================

# --- Manglish Alphabet ---
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
MANGLISH_VOCAB_SIZE = len(manglish_to_int)

# --- Malayalam Alphabet ---
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
    """Scan all transcripts to build Malayalam character vocabulary."""
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


# ==============================
# 3. DATASET IMPLEMENTATION
# ==============================

def safe_load_npy(path):
    """Safely load numpy array with error handling."""
    try:
        return np.load(path)
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None


class LipReadingDataset(Dataset):
    """
    LipReading dataset: loads landmarks, ROI frames, and transcripts.
    Converts Malayalam → Manglish → tensors (Malayalam + Manglish).
    """

    def __init__(self, landmark_dir, lip_roi_dir, transcript_dir, img_size,
                 malayalam_vocab, manglish_vocab):
        self.img_size = img_size
        self.landmark_dir = landmark_dir
        self.lip_roi_dir = lip_roi_dir
        self.transcript_dir = transcript_dir
        self.malayalam_vocab = malayalam_vocab
        self.manglish_vocab = manglish_vocab
        self.samples = []

        print("🔍 Searching for valid data samples...")
        landmark_files = sorted(glob.glob(os.path.join(landmark_dir, "*.npy")))
        for landmark_path in tqdm(landmark_files, desc="Matching data files"):
            base_name = os.path.basename(landmark_path).replace("_landmarks.npy", "")
            roi_dir = os.path.join(lip_roi_dir, base_name)
            transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")
            if os.path.isdir(roi_dir) and os.path.exists(transcript_path):
                self.samples.append({
                    "id": base_name,
                    "landmarks": landmark_path,
                    "rois": roi_dir,
                    "transcript": transcript_path
                })
        print(f"✅ Found {len(self.samples)} complete samples.")

    def __len__(self):
        return len(self.samples)

    def load_frames(self, roi_dir):
        """Load and resize grayscale lip ROI video frames."""
        frame_paths = sorted(glob.glob(os.path.join(roi_dir, "*.png")))
        frames = []
        for fp in frame_paths:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, self.img_size)
            frames.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))
        return torch.stack(frames) if frames else None

    def encode_text(self, text, vocab, sos_token=1, eos_token=2, unk_token=3):
        """Encode text string to tensor of indices."""
        indices = [sos_token]
        for ch in text:
            indices.append(vocab.get(ch, unk_token))
        indices.append(eos_token)
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]

            # --- Load landmarks ---
            landmarks = safe_load_npy(sample["landmarks"])
            if landmarks is None or len(landmarks.shape) < 2:
                print(f"[SKIP] Invalid landmarks for {sample['id']}")
                return None
            landmarks = torch.tensor(landmarks, dtype=torch.float32)

            # --- Load video frames ---
            frames = self.load_frames(sample["rois"])
            if frames is None:
                print(f"[SKIP] Frames missing for {sample['id']}")
                return None

            # --- Read Malayalam transcript ---
            with open(sample["transcript"], 'r', encoding='utf-8') as f:
                lines = [line.strip().split() for line in f.readlines()]
            full_text = " ".join([parts[-1] for parts in lines if len(parts) > 2])
            if not full_text.strip():
                print(f"[SKIP] Empty transcript for {sample['id']}")
                return None

            # --- Convert Malayalam to Manglish ---
            manglish_text = ml2en.transliterate(full_text)

            # --- Encode text sequences ---
            mal_tensor = self.encode_text(full_text, self.malayalam_vocab)
            man_tensor = self.encode_text(manglish_text, self.manglish_vocab)

            return {
                "id": sample["id"],
                "video": frames,          # [T, 1, H, W]
                "landmarks": landmarks,   # [T, num_points*2]
                "manglish_seq": man_tensor,
                "malayalam_seq": mal_tensor
            }

        except Exception as e:
            print(f"[ERROR] Dataset index {idx} failed: {e}")
            return None


def collate_fn(batch):
    """Custom collate — pads sequences and batches them safely."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    videos = [b["video"] for b in batch]
    landmarks = [b["landmarks"] for b in batch]
    manglish_seqs = [b["manglish_seq"] for b in batch]
    malayalam_seqs = [b["malayalam_seq"] for b in batch]

    videos = pad_sequence(videos, batch_first=True)
    landmarks = pad_sequence(landmarks, batch_first=True)
    manglish_padded = pad_sequence(manglish_seqs, batch_first=True, padding_value=MANGLISH_PAD_TOKEN)
    malayalam_padded = pad_sequence(malayalam_seqs, batch_first=True, padding_value=MALAYALAM_PAD_TOKEN)

    return {
        "video": videos.permute(0, 2, 1, 3, 4),  # [B, 1, T, H, W]
        "landmarks": landmarks,
        "manglish_seq": manglish_padded,
        "malayalam_seq": malayalam_padded
    }


# ==============================
# 4. TRAINING AND VALIDATION
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # --- Build Vocabulary ---
    MALAYALAM_VOCAB_SIZE = build_malayalam_vocab(CONFIG["data"]["transcripts"])
    print(f"Malayalam vocab size: {MALAYALAM_VOCAB_SIZE}")
    print(f"Manglish vocab size: {MANGLISH_VOCAB_SIZE}")

    # --- Dataset Split ---
    dataset = LipReadingDataset(
        CONFIG["data"]["landmarks"], CONFIG["data"]["lip_rois"],
        CONFIG["data"]["transcripts"], CONFIG["image_size"],
        malayalam_to_int, manglish_to_int
    )
    if len(dataset) == 0:
        print("No data samples detected. Check dataset paths.")
        return

    val_size = int(CONFIG["validation_split"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Dataset split: {train_size} train / {val_size} val samples")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, 
                            collate_fn=collate_fn, num_workers=2)

    # --- Model Setup ---
    model = LipFormer(
        num_pinyins=MANGLISH_VOCAB_SIZE,
        num_chars=MALAYALAM_VOCAB_SIZE,
        device=device
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M parameters.")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    pinyin_loss_fn = nn.CrossEntropyLoss(ignore_index=MANGLISH_PAD_TOKEN)
    char_loss_fn = nn.CrossEntropyLoss(ignore_index=MALAYALAM_PAD_TOKEN)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')

    # --- Epoch Loop ---
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            if batch is None:
                continue
            videos, landmarks, man_targets, mal_targets = [d.to(device) for d in batch.values()]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pinyin_preds, char_preds = model(
                    videos, landmarks,
                    pinyin_targets=man_targets,
                    char_targets=mal_targets,
                    teacher_forcing_ratio=CONFIG["teacher_forcing_ratio"]
                )
                loss_pinyin = pinyin_loss_fn(pinyin_preds.view(-1, MANGLISH_VOCAB_SIZE), man_targets.view(-1))
                loss_char = char_loss_fn(char_preds.view(-1, MALAYALAM_VOCAB_SIZE), mal_targets.view(-1))
                total_loss = CONFIG["lambda_val"] * loss_pinyin + (1 - CONFIG["lambda_val"]) * loss_char

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                if batch is None:
                    continue
                videos, landmarks, man_targets, mal_targets = [d.to(device) for d in batch.values()]
                pinyin_preds, char_preds = model(
                    videos, landmarks,
                    pinyin_targets=None,
                    char_targets=None,
                    teacher_forcing_ratio=0.0
                )
                loss_pinyin = pinyin_loss_fn(pinyin_preds.view(-1, MANGLISH_VOCAB_SIZE), man_targets.view(-1))
                loss_char = char_loss_fn(char_preds.view(-1, MALAYALAM_VOCAB_SIZE), mal_targets.view(-1))
                total_loss = CONFIG["lambda_val"] * loss_pinyin + (1 - CONFIG["lambda_val"]) * loss_char
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CONFIG["checkpoint_dir"], "lipformer_malayalam_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with Val Loss {best_val_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
