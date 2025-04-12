
import torch


class EnhancedGRU(torch.nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, num_layers=3, dropout=0.5, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout

        # Feature extraction from points with more capacity
        self.point_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(32, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5)
        )

        # Global pooling with both max and average
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)

        # Bidirectional GRU for sequence modeling
        self.gru = torch.nn.GRU(
            input_size=128,  # 64*2 from max+avg pooling
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * (2 if bidirectional else 1), 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

        # Classification head
        output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        batch_size, frame_num, point_num, features = x.shape

        # Process each frame to get frame features
        frame_features = []
        for i in range(frame_num):
            # Get current frame
            frame = x[:, i]  # (B, P, F)

            # Extract point features
            point_features = self.point_encoder(frame)  # (B, P, 64)

            # Dual pooling across points
            max_pooled = self.max_pool(point_features.transpose(1, 2)).squeeze(-1)  # (B, 64)
            avg_pooled = self.avg_pool(point_features.transpose(1, 2)).squeeze(-1)  # (B, 64)

            # Concatenate pooled features
            pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # (B, 128)
            frame_features.append(pooled)

        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # (B, F, 128)

        # Handle masking for packed sequence if mask is provided
        if mask is not None:
            # Pack padded sequence
            lengths = mask.sum(dim=1).cpu().int()

            # Ensure all sequences have at least length 1
            lengths = torch.clamp(lengths, min=1)

            # Pack sequence
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                frame_features,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            # Process with GRU
            packed_output, hidden = self.gru(packed_input)

            # Unpack the sequence
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=frame_num
            )

            # Apply attention mechanism
            attention_weights = []
            for i in range(frame_num):
                if i < output.size(1):
                    feat = output[:, i, :]  # (B, H*2)
                    weight = self.attention(feat)  # (B, 1)
                    attention_weights.append(weight)
                else:
                    # For padding frames
                    attention_weights.append(torch.zeros_like(weight))

            attention_weights = torch.cat(attention_weights, dim=1)  # (B, F)

            # Apply softmax only over valid frames
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)

            # Apply attention weights
            weighted_output = output * attention_weights.unsqueeze(-1)
            context_vector = weighted_output.sum(dim=1)  # (B, H*2)
        else:
            # Process with GRU (without packing)
            output, hidden = self.gru(frame_features)

            # Use the last hidden state
            if self.bidirectional:
                # Concatenate the last hidden state from both directions
                last_hidden_forward = hidden[-2, :, :]
                last_hidden_backward = hidden[-1, :, :]
                context_vector = torch.cat([last_hidden_forward, last_hidden_backward], dim=1)
            else:
                context_vector = hidden[-1]  # (B, H)

        # Classification
        output = self.classifier(context_vector)

        return output
