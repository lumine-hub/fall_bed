
import torch


# Combined model that uses both TCN and GRU
class HybridModel(torch.nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, dropout=0.5):
        super().__init__()

        # Common point feature extractor
        self.point_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.LayerNorm(32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU()
        )

        # TCN path
        self.tcn_path = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        # GRU path
        self.gru_path = torch.nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Fusion and classification
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(256 + hidden_size * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        batch_size, frame_num, point_num, features = x.shape

        # Process each frame to extract point features
        frame_features = []
        for i in range(frame_num):
            frame = x[:, i]  # (B, P, F)
            point_features = self.point_encoder(frame)  # (B, P, 64)
            frame_features.append(point_features)

        # Stack frame features for TCN path
        stacked_features = torch.stack([feat.mean(dim=1) for feat in frame_features], dim=2)  # (B, 64, F)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, stacked_features.size(1), -1)
            stacked_features = stacked_features * mask_expanded

        # Process TCN path
        tcn_features = self.tcn_path(stacked_features).squeeze(-1)  # (B, 256)

        # Stack frame features for GRU path (differently)
        gru_input = torch.stack([feat.mean(dim=1) for feat in frame_features], dim=1)  # (B, F, 64)

        # Handle masking for GRU if provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu().int()
            lengths = torch.clamp(lengths, min=1)

            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                gru_input,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )

            _, hidden = self.gru_path(packed_input)
        else:
            _, hidden = self.gru_path(gru_input)

        # Concatenate last hidden state from both directions
        gru_features = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_size*2)

        # Combine features from both paths
        combined = torch.cat([tcn_features, gru_features], dim=1)

        # Classification
        output = self.fusion(combined)

        return output
