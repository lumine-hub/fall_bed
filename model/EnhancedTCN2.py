import torch

# Enhanced Temporal Convolutional Network
class EnhancedTCN2(torch.nn.Module):
    def __init__(self, num_classes=2, max_frames=50, dropout=0.5):
        super().__init__()
        # 添加输入归一化
        self.input_bn = torch.nn.BatchNorm1d(5)

        # 简化后的特征提取
        self.frame_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(5, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        # 更高效的时间卷积
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, 512, 3, padding=1),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        # 修正后的注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(128, 1)
        )

        # 简化后的分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x, mask=None):
        x = self.input_bn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 输入归一化

        # 特征提取
        frame_features = []
        for i in range(x.size(1)):
            frame = x[:, i].permute(0, 2, 1)
            feat = self.frame_feature_extractor(frame).squeeze(-1)
            frame_features.append(feat)
        frame_features = torch.stack(frame_features, dim=2)

        # 注意力加权聚合
        attn_weights = self.attention(frame_features.permute(0, 2, 1))
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = (frame_features * attn_weights.permute(0, 2, 1)).sum(dim=2)

        # 分类
        return self.classifier(weighted)
