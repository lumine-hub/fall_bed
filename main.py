import time
import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


from dataset.dir import RadarDataset

from model import EnhancedTCN, LightweightSpatioTemporalModel


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    start_time = time.time()
    total_loss = 0

    print(f"Epoch {epoch} started...")
    for batch_idx, (point_clouds, mask, label, frame_num) in enumerate(train_loader):
        point_clouds, mask, label = point_clouds.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(point_clouds, mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(point_clouds)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch} finished in {epoch_time:.2f} seconds. Avg Loss: {total_loss / len(train_loader):.6f}")


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for point_clouds, mask, label, frame_num in test_loader:
            point_clouds, mask, label = point_clouds.to(device), mask.to(device), label.to(device)
            output = model(point_clouds, mask)
            test_loss += criterion(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")

    # 计算分类报告
    report = classification_report(all_labels, all_preds, digits=4)
    print("Classification Report:")
    print(report)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_dir = None
    # txt_dir = "F:\\code\\radar\\fall_bed\\mmParse\\fall_bed\\cut_txt\\train_user1.txt"
    file_path_prefix = "F:\\code\\rada\\script\\origin_data_to_csv"
    dir_list = ["fall_bed_data" + str(i) for i in range(2, 6)]
    print(dir_list)
    BATCH_SIZE = 32
    MAX_POINTS = 100
    MAX_FRAMES = 50
    EPOCHS = 30
    LEARNING_RATE = 0.001
    PROCESSING_METHOD = 'mask'

    dataset = RadarDataset(txt_dir,file_path_prefix=file_path_prefix,dir_list=dir_list, max_points=MAX_POINTS, max_frames=MAX_FRAMES, method=PROCESSING_METHOD)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_set, val_set = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    # model = EnhancedTCN(num_classes=2, max_frames=50, dropout=0.5).to(device)
    model = LightweightSpatioTemporalModel(num_classes=2, dropout=0.5).to(device)
    # 计算类别权重
    import torch.nn.functional as F

    # num_pos = 27
    # num_neg = 10
    # total = num_pos + num_neg
    # weight = torch.tensor([total / (2 * num_neg), total / (2 * num_pos)]).to(device)  # 负类权重更高
    #
    # # 使用带权重的交叉熵损失
    # criterion = nn.CrossEntropyLoss(weight=weight)


    # class FocalLoss(nn.Module):
    #     def __init__(self, alpha=0.25, gamma=2):
    #         super(FocalLoss, self).__init__()
    #         self.alpha = alpha
    #         self.gamma = gamma
    #
    #     def forward(self, inputs, targets):
    #         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    #         pt = torch.exp(-ce_loss)
    #         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
    #         return focal_loss.mean()
    #
    #
    # # 使用 Focal Loss 代替 CrossEntropyLoss
    # criterion = FocalLoss(alpha=0.25, gamma=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, val_loader, criterion)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"enhanced_tcn_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_name)
    print("模型保存成功。")
