

import torch
class LightweightSpatioTemporalModel(torch.nn.Module):
    def __init__(self, num_classes=2, input_features=5, dropout=0.7):
        super().__init__()

        # 修正输入归一化层
        self.input_norm = torch.nn.BatchNorm2d(input_features)  # 改为BatchNorm2d

        # 空间特征提取
        self.spatial_net = torch.nn.Sequential(
            torch.nn.Conv1d(input_features, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch.nn.MaxPool1d(2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU()
        )

        # 时间特征聚合
        self.temporal_net = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x, mask=None):
        B, T, V, C = x.shape

        # 修正后的归一化步骤
        x = x.permute(0, 3, 1, 2)  # [B, C, T, V]
        x = self.input_norm(x)  # 使用BatchNorm2d处理4D输入
        x = x.permute(0, 2, 3, 1)  # 恢复形状 [B, T, V, C]

        # 空间特征提取
        spatial_features = []
        for t in range(T):
            frame = x[:, t, :, :].permute(0, 2, 1)  # [B, C, V]
            feat = self.spatial_net(frame)  # [B, 32, V/2]
            spatial_features.append(feat)

        # 时间维度处理
        temporal_input = torch.stack(spatial_features, dim=2)  # [B, 32, T, V/2]
        temporal_input = temporal_input.view(B, 32, -1)  # [B, 32, T*(V/2)]

        temporal_feat = self.temporal_net(temporal_input)  # [B, 64, 1]

        return self.classifier(temporal_feat.squeeze(-1))
