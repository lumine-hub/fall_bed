
import torch

class EnhancedTCN(torch.nn.Module):
    def __init__(self, num_classes=2, max_frames=50, dropout=0.5):
        super().__init__()
        self.max_frames = max_frames
        self.dropout_rate = dropout

        # Feature extraction from each frame
        self.frame_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(5, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Dropout(dropout)
        )

        # Temporal modeling with deeper convolutions
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 256, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1),
            torch.nn.Dropout(dropout)
        )

        # Attention mechanism for weighted pooling
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

        # Classifier with more layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5),
            torch.nn.Linear(128, num_classes)
        )
    # x: (batch_size, frame_num, point_num, features)
    def forward(self, x, mask=None):
        batch_size, frame_num, point_num, features = x.shape

        # Process each frame separately
        frame_features = []
        for i in range(frame_num):
            # Extract features from each frame (B, P, F) -> (B, F, P)
            frame = x[:, i, :, :].permute(0, 2, 1)  # (B, F, P) (batch_size, features, point_num)
            frame_feat = self.frame_feature_extractor(frame).squeeze(-1)  # (B, 128)
            frame_features.append(frame_feat)

        # Stack frame features
        frame_features = torch.stack(frame_features, dim=2)  # (B, 128, F) (32,128,50)

        # Apply mask if provided
        if mask is not None:  # (32, 50)
            # Expand mask to match features dimension
            mask_expanded = mask.unsqueeze(1).expand(-1, frame_features.size(1), -1)
            # Apply mask
            frame_features = frame_features * mask_expanded

            # Optional: Apply attention for weighted pooling
            attention_weights = []
            for i in range(frame_num):
                frame_feat = frame_features[:, :, i]  # (B, 128)
                weight = self.attention(frame_feat)  # (B, 1)
                attention_weights.append(weight)

            attention_weights = torch.cat(attention_weights, dim=1)  # (B, F)
            # Apply softmax only over valid frames
            # attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
            attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)

            # Reshape for broadcasting
            attention_weights = attention_weights.unsqueeze(1).expand(-1, frame_features.size(1), -1)

            # Apply attention weights
            # frame_features = frame_features * attention_weights

        # Temporal modeling
        temporal_features = self.temporal_conv(frame_features).squeeze(-1)

        # Classification
        output = self.classifier(temporal_features)

        return output
