import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Acknowledgment: This code is a replication of the LipFormer model based on the
# research paper "LipFormer: Learning to Lipread Unseen Speakers based on Visual-Landmark Transformers"
# (arXiv:2302.02141v1). The implementation details are derived from the descriptions
# and diagrams within the paper.

# -----------------------------------------------------------------------------
# 1. Visual Stream Components
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Implements the Channel Attention module as described in Figure 3(a) of the paper.
    This module helps the model focus on more informative channels.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class VisualStream(nn.Module):
    """
    Implements the Visual Stream of the LipFormer model.
    This stream processes the sequence of mouth region images.
    It consists of a 3D CNN backbone, Channel Attention, and a Bi-GRU.
    """
    def __init__(self, in_channels=1, hidden_dim=256):
        super(VisualStream, self).__init__()
        # 3D CNN Backbone based on standard lip-reading architectures
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(64, 128, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # Channel Attention Module
        self.channel_attention = ChannelAttention(256)

        # Bi-directional GRU to process temporal features
        # The input size is determined by the output of the CNN
        # After CNN: [B, 256, T, H/16, W/16]. Assuming H=80, W=160 -> H'=5, W'=10
        # We flatten the spatial dimensions: 256 * 5 * 10 = 12800
        # Note: Paper uses 160x80 crop. Let's adapt to common sizes like 112x112 or calculate dynamically.
        # For a 160x80 input, output H,W will be 5x10.
        # cnn_output_size = 256 * 5 * 10
        cnn_output_size = 256 * 2 * 5 # For a 80x160 input -> (80/16, 160/32) -> 5,5 not 2,5 -> (80/16, 160/16) -> 5,10
                                      # Lets assume a typical lipreading input H=112, W=112 -> H'=3, W'=3
                                      # cnn_output_size = 256 * 3 * 3 = 2304

        # Let's use an adaptive pool to be robust to input size.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Pool H and W to 1x1
        cnn_output_size = 256

        self.gru = nn.GRU(cnn_output_size, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        # Input x: [B, C, T, H, W] (e.g., [B, 1, 75, 80, 160])
        x = self.cnn(x)
        x = x * self.channel_attention(x) # Apply channel attention

        # Reshape for GRU
        # x shape: [B, 256, T, H', W']
        x = self.adaptive_pool(x) # -> [B, 256, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1) # -> [B, 256, T]
        x = x.permute(0, 2, 1) # -> [B, T, 256]

        # Pass through Bi-GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x) # -> [B, T, 2 * hidden_dim]
        return x

# -----------------------------------------------------------------------------
# 2. Landmark Stream
# -----------------------------------------------------------------------------

class LandmarkStream(nn.Module):
    """
    Implements the Landmark Stream of the LipFormer model.
    This stream processes facial landmark sequences to create a speaker-invariant
    representation of lip motion.
    """
    def __init__(self, landmark_dim=37, hidden_dim=256):
        super(LandmarkStream, self).__init__()
        # Input to Landmark Stream is an angle matrix of size 37x37
        # The paper mentions 'difference operation' then a Bi-GRU.
        # The "angle matrix" size is 37x37 from cosine similarity of 37 landmarks.
        # Let's assume the input is the flattened angle matrix.
        input_dim = landmark_dim * landmark_dim # 37 * 37 = 1369

        # Embedding layer to project high-dim landmark features to a lower dim
        self.embedding = nn.Linear(input_dim, 512)

        self.gru = nn.GRU(512, hidden_dim, num_layers=2,
                          batch_first=True, bidirectional=True)

    def _calculate_angle_matrix(self, landmarks):
        # landmarks: [B, T, N, 2] where N is num_landmarks (37)
        # We need to compute cosine similarity between all pairs of landmark vectors
        # Let's use pairwise distance for simplicity, as paper lacks detail
        # on the exact "angle matrix" calculation from raw coords.
        # A more faithful implementation of "angle matrix":
        # 1. Select a reference point (e.g., center of mouth).
        # 2. Compute vectors from reference to all other points.
        # 3. Compute cosine similarity between all pairs of these vectors.

        # Let's implement the simpler frame differencing mentioned in paper.
        # "we first perform a difference operation on adjacent frames"
        diff_landmarks = landmarks[:, 1:] - landmarks[:, :-1] # [B, T-1, N, 2]

        # For cosine similarity, let's process the 37 landmark vectors (each with x,y coords)
        # Normalize each landmark vector (relative to origin, not ideal)
        landmarks_norm = F.normalize(landmarks, p=2, dim=-1) # [B, T, N, 2]
        # Reshape for batch matrix multiplication
        landmarks_norm_t = landmarks_norm.transpose(-2, -1) # [B, T, 2, N]
        # Cosine similarity matrix: L * L^T
        angle_matrix = torch.matmul(landmarks_norm, landmarks_norm_t) # [B, T, N, N]
        return angle_matrix


    def forward(self, landmarks):
        # Input landmarks: [B, T, 37, 2] (37 landmarks, each with x, y)
        angle_matrix = self._calculate_angle_matrix(landmarks) # -> [B, T, 37, 37]

        # Perform difference operation on adjacent frames
        diff_features = angle_matrix[:, 1:] - angle_matrix[:, :-1] # [B, T-1, 37, 37]

        # Flatten the feature matrix
        B, T, N, _ = diff_features.shape
        diff_features = diff_features.view(B, T, N * N) # [B, T-1, 1369]

        # Project to lower dimension
        embedded_features = self.embedding(diff_features) # [B, T-1, 512]

        # Pass through Bi-GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(embedded_features) # [B, T-1, 2 * hidden_dim]
        return x

# -----------------------------------------------------------------------------
# 3. Cross-Modal Fusion
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Standard Positional Encoding for Transformers."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D] -> permute to [T, B, D] for PE
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x.permute(1, 0, 2)

class CrossModalFusion(nn.Module):
    """
    Implements the Cross-Modal Fusion module using a Transformer Encoder.
    It fuses features from the Visual and Landmark streams.
    The paper describes self-attention followed by cross-attention.
    This is effectively a standard Transformer Encoder layer.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.3):
        super(CrossModalFusion, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important: expects [B, T, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, visual_feat, landmark_feat):
        # visual_feat: [B, T_v, D]
        # landmark_feat: [B, T_l, D]
        # Since T_v = T and T_l = T-1, we need to align them. Let's trim the visual feature.
        visual_feat = visual_feat[:, 1:, :] # Trim first frame to match landmark diff

        # Concatenate features along the feature dimension
        # The paper suggests fusion, let's assume simple addition or concatenation.
        # Given the transformer structure, stacking them and letting self-attention
        # learn the interactions seems more plausible.
        # Let's concatenate along the time dimension and treat them as one sequence.
        fused_input = torch.cat([visual_feat, landmark_feat], dim=1) # [B, (T-1) + (T-1), D]

        # Add positional encoding
        fused_input = self.pos_encoder(fused_input)

        # Pass through Transformer Encoder
        fused_output = self.transformer_encoder(fused_input) # [B, 2*(T-1), D]
        return fused_output


# -----------------------------------------------------------------------------
# 4. Text Generation (Cascaded Seq2Seq)
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    """A general attention module."""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, D]
        # encoder_outputs: [B, T, D]
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1) # [B, T, D]
        attn_energies = self.score(h, encoder_outputs) # [B, T]
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # [B, 1, T]

    def score(self, hidden, encoder_outputs):
        # [B, T, 2*D] -> [B, T, D]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2) # [B, D, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1) # [B, 1, D]
        energy = torch.bmm(v, energy) # [B, 1, T]
        return energy.squeeze(1) # [B, T]


class PinyinDecoder(nn.Module):
    """
    Predicts the Pinyin sequence from the fused visual-landmark features.
    This is the first stage of the cascaded decoder.
    """
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super(PinyinDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim + emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [B] (previous predicted pinyin)
        # hidden: [1, B, D] (previous hidden state)
        # encoder_outputs: [B, T, D]
        input = input.unsqueeze(1) # [B, 1]
        embedded = self.dropout(self.embedding(input)) # [B, 1, emb_dim]

        # Calculate attention
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs) # [B, 1, T]
        context = torch.bmm(attn_weights, encoder_outputs) # [B, 1, D]

        # Concatenate embedded input and context
        gru_input = torch.cat((embedded, context), dim=2) # [B, 1, emb_dim + D]

        # Pass through GRU
        output, hidden = self.gru(gru_input, hidden) # output: [B, 1, D], hidden: [1, B, D]

        # Final prediction
        prediction = self.fc_out(output.squeeze(1)) # [B, output_dim]

        return prediction, hidden


class CharacterDecoder(nn.Module):
    """
    Predicts the final Character sequence.
    Uses "dual attention" on both the fused features (V-L) and the
    pinyin encoder outputs.
    """
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout):
        super(CharacterDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # Attention over fused visual-landmark features
        self.vl_attention = Attention(hidden_dim)
        # Attention over pinyin encoder features
        self.pinyin_attention = Attention(hidden_dim)

        # GRU input: embedded char + context_vl + context_pinyin
        self.gru = nn.GRU(emb_dim + hidden_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, vl_encoder_outputs, pinyin_encoder_outputs):
        # input: [B] (previous predicted character)
        # hidden: [1, B, D]
        # vl_encoder_outputs: [B, T_vl, D]
        # pinyin_encoder_outputs: [B, T_pinyin, D]
        input = input.unsqueeze(1) # [B, 1]
        embedded = self.dropout(self.embedding(input)) # [B, 1, emb_dim]

        # Calculate dual attention
        vl_attn_weights = self.vl_attention(hidden.squeeze(0), vl_encoder_outputs)
        vl_context = torch.bmm(vl_attn_weights, vl_encoder_outputs) # [B, 1, D]

        pinyin_attn_weights = self.pinyin_attention(hidden.squeeze(0), pinyin_encoder_outputs)
        pinyin_context = torch.bmm(pinyin_attn_weights, pinyin_encoder_outputs) # [B, 1, D]

        # Concatenate for GRU input
        gru_input = torch.cat((embedded, vl_context, pinyin_context), dim=2)

        # Pass through GRU
        output, hidden = self.gru(gru_input, hidden)

        # Final prediction
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

# -----------------------------------------------------------------------------
# 5. Main LipFormer Model
# -----------------------------------------------------------------------------

class LipFormer(nn.Module):
    """
    The complete LipFormer model, integrating all components.
    """
    def __init__(self, num_pinyins, num_chars, hidden_dim=256, d_model=512,
                 pinyin_emb_dim=128, char_emb_dim=128, dropout=0.3, device='cpu'):
        super(LipFormer, self).__init__()

        self.device = device

        # Streams
        self.visual_stream = VisualStream(hidden_dim=hidden_dim)
        self.landmark_stream = LandmarkStream(hidden_dim=hidden_dim)

        # Fusion
        # Note: The output of streams is [B, T, 2*hidden_dim]. We need to match d_model.
        self.visual_proj = nn.Linear(2 * hidden_dim, d_model)
        self.landmark_proj = nn.Linear(2 * hidden_dim, d_model)
        self.fusion = CrossModalFusion(d_model=d_model)

        # Pinyin Encoder (to encode predicted pinyin sequence for char decoder)
        self.pinyin_encoder = nn.GRU(pinyin_emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.pinyin_embedding = nn.Embedding(num_pinyins, pinyin_emb_dim)

        # Decoders
        self.pinyin_decoder = PinyinDecoder(num_pinyins, pinyin_emb_dim, d_model, dropout)
        self.char_decoder = CharacterDecoder(num_chars, char_emb_dim, d_model, dropout)

        self.d_model = d_model
        self.num_pinyins = num_pinyins
        self.num_chars = num_chars

    def forward(self, video, landmarks, pinyin_targets=None, char_targets=None, teacher_forcing_ratio=0.5):
        # video: [B, 1, T, H, W]
        # landmarks: [B, T, 37, 2]
        # targets: [B, max_len]

        # Step 1: Process inputs through streams
        visual_feat = self.visual_stream(video)     # [B, T, 512]
        landmark_feat = self.landmark_stream(landmarks) # [B, T-1, 512]

        # Step 2: Project and Fuse features
        visual_feat = self.visual_proj(visual_feat)
        landmark_feat = self.landmark_proj(landmark_feat)
        fused_feat = self.fusion(visual_feat, landmark_feat) # [B, 2*(T-1), d_model]

        # Step 3: Decode Pinyin sequence
        batch_size = video.size(0)
        max_pinyin_len = pinyin_targets.size(1) if pinyin_targets is not None else 20
        pinyin_outputs = torch.zeros(batch_size, max_pinyin_len, self.num_pinyins).to(self.device)
        
        # Initial input for decoder is <sos> token
        pinyin_input = torch.zeros(batch_size, dtype=torch.long).to(self.device) # Assume <sos> is 0
        pinyin_decoder_hidden = torch.zeros(1, batch_size, self.d_model).to(self.device) # Initial hidden state

        for t in range(max_pinyin_len):
            pinyin_output, pinyin_decoder_hidden = self.pinyin_decoder(
                pinyin_input, pinyin_decoder_hidden, fused_feat
            )
            pinyin_outputs[:, t] = pinyin_output
            
            # Decide whether to use teacher forcing
            use_teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pinyin_output.argmax(1)
            pinyin_input = pinyin_targets[:, t] if use_teacher_force and pinyin_targets is not None else top1

        # Step 4: Encode the Pinyin sequence for the Character Decoder
        # This is used as context for the character decoder as per the paper

        if pinyin_targets is not None:
            # --- Training Mode ---
            # Use the ground-truth pinyin targets
            pinyin_embedded = self.pinyin_embedding(pinyin_targets)
            pinyin_encoder_outputs, _ = self.pinyin_encoder(pinyin_embedded)
        else:
            # --- Validation/Inference Mode ---
            # Use the pinyin sequence predicted in Step 3
            # pinyin_outputs shape: [B, max_pinyin_len, num_pinyins]
            predicted_pinyin_tokens = pinyin_outputs.argmax(dim=-1) # Get the index of the highest logit
            
            # Pass the *predicted* tokens to the embedding layer
            pinyin_embedded = self.pinyin_embedding(predicted_pinyin_tokens)
            pinyin_encoder_outputs, _ = self.pinyin_encoder(pinyin_embedded)

        # Step 5: Decode Character sequence
        max_char_len = char_targets.size(1) if char_targets is not None else 20
        char_outputs = torch.zeros(batch_size, max_char_len, self.num_chars).to(self.device)
        
        # Initial input for decoder is <sos> token
        char_input = torch.zeros(batch_size, dtype=torch.long).to(self.device) # Assume <sos> is 0
        char_decoder_hidden = torch.zeros(1, batch_size, self.d_model).to(self.device)

        for t in range(max_char_len):
            char_output, char_decoder_hidden = self.char_decoder(
                char_input, char_decoder_hidden, fused_feat, pinyin_encoder_outputs
            )
            char_outputs[:, t] = char_output

            use_teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = char_output.argmax(1)
            char_input = char_targets[:, t] if use_teacher_force and char_targets is not None else top1
            
        return pinyin_outputs, char_outputs

# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Model Hyperparameters ---
    # Vocabulary sizes for the CMLR dataset (placeholders)
    # The paper uses Chinese Mandarin Lip Reading (CMLR) dataset.
    # Actual vocab size should be determined from the dataset.
    NUM_PINYINS = 1500  # Approximate number of pinyin syllables + special tokens
    NUM_CHARS = 3500    # Approximate number of Chinese characters + special tokens

    # Other parameters
    BATCH_SIZE = 4
    SEQUENCE_LENGTH = 75 # Number of video frames
    MAX_PINYIN_LEN = 25
    MAX_CHAR_LEN = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate the Model ---
    model = LipFormer(
        num_pinyins=NUM_PINYINS,
        num_chars=NUM_CHARS,
        device=DEVICE
    ).to(DEVICE)

    print(f"Model created and moved to {DEVICE}.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f}M")

    # --- Create Dummy Input Data ---
    # Video input: Batch, Channels, Time, Height, Width
    # Paper specifies 160x80 crop.
    dummy_video = torch.randn(BATCH_SIZE, 1, SEQUENCE_LENGTH, 80, 160).to(DEVICE)

    # Landmark input: Batch, Time, Num_Landmarks, Coords(x,y)
    # 20 lip landmarks + 17 facial contour landmarks = 37
    dummy_landmarks = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 37, 2).to(DEVICE)
    
    # Target sequences for teacher forcing
    dummy_pinyin_targets = torch.randint(0, NUM_PINYINS, (BATCH_SIZE, MAX_PINYIN_LEN)).to(DEVICE)
    dummy_char_targets = torch.randint(0, NUM_CHARS, (BATCH_SIZE, MAX_CHAR_LEN)).to(DEVICE)

    print("\n--- Dummy Input Shapes ---")
    print(f"Video:           {dummy_video.shape}")
    print(f"Landmarks:       {dummy_landmarks.shape}")
    print(f"Pinyin Targets:  {dummy_pinyin_targets.shape}")
    print(f"Character Targets: {dummy_char_targets.shape}")

    # --- Forward Pass ---
    # Set teacher forcing to 0.5 for demonstration
    pinyin_preds, char_preds = model(dummy_video, dummy_landmarks, dummy_pinyin_targets, dummy_char_targets, teacher_forcing_ratio=0.5)

    print("\n--- Output Shapes ---")
    print(f"Pinyin Preds:    {pinyin_preds.shape}") # [B, max_pinyin_len, num_pinyins]
    print(f"Character Preds: {char_preds.shape}")   # [B, max_char_len, num_chars]

    # --- Loss Calculation Example ---
    # The paper mentions a joint loss function: L = λ * L_pinyin + (1-λ) * L_char
    # This requires a CrossEntropyLoss that ignores padding tokens.
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # Assuming -1 is the padding index
    
    # Reshape for loss calculation
    # Input: [N, C], Target: [N]
    pinyin_preds_flat = pinyin_preds.view(-1, NUM_PINYINS)
    pinyin_targets_flat = dummy_pinyin_targets.view(-1)
    
    char_preds_flat = char_preds.view(-1, NUM_CHARS)
    char_targets_flat = dummy_char_targets.view(-1)
    
    loss_pinyin = loss_fn(pinyin_preds_flat, pinyin_targets_flat)
    loss_char = loss_fn(char_preds_flat, char_targets_flat)
    
    lambda_val = 0.7 # As mentioned in the paper
    total_loss = lambda_val * loss_pinyin + (1 - lambda_val) * loss_char
    
    print("\n--- Example Loss Calculation ---")
    print(f"Pinyin Loss: {loss_pinyin.item():.4f}")
    print(f"Character Loss: {loss_char.item():.4f}")
    print(f"Total Joint Loss: {total_loss.item():.4f}")