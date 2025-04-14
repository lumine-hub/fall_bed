import os
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dataset.dir import RadarDataset
from model import EnhancedTCN, LightweightSpatioTemporalModel, EnhancedGRU, HybridModel


def train(model, device, train_loader, optimizer, criterion, epoch, PROCESSING_METHOD="", log_file=None):
    model.train()
    start_time = time.time()
    total_loss = 0

    log_message = f"Epoch {epoch} started...\n"
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_message)
    print(log_message.strip())

    for batch_idx, (point_clouds, mask, label, frame_num) in enumerate(train_loader):
        point_clouds, mask, label = point_clouds.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        output = None
        if PROCESSING_METHOD == "mask":
            mask = mask.to(device)
            output = model(point_clouds, mask)
        else:
            output = model(point_clouds)
        loss = criterion(output, label)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            batch_log = f"Train Epoch: {epoch} [{batch_idx * len(point_clouds)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}\n"
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(batch_log)
            print(batch_log.strip())

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss / len(train_loader)
    epoch_log = f"Epoch {epoch} finished in {epoch_time:.2f} seconds. Avg Loss: {avg_loss:.6f}\n"
    if log_file:
        with open(log_file, 'a') as f:
            f.write(epoch_log)
    print(epoch_log.strip())
    return avg_loss


def test(model, device, test_loader, criterion, log_file=None):
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
    test_log = f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    if log_file:
        with open(log_file, 'a') as f:
            f.write(test_log)
    print(test_log.strip())

    # 计算分类报告
    report = classification_report(all_labels, all_preds, digits=4)
    report_log = "Classification Report:\n" + report + "\n"
    if log_file:
        with open(log_file, 'a') as f:
            f.write(report_log)
    print(report_log)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return test_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_dir = None
    # txt_dir = "F:\\code\\radar\\fall_bed\\mmParse\\fall_bed\\cut_txt\\train_user1.txt"
    file_path_prefix = "F:\\code\\rada\\script\\origin_data_to_csv"
    dir_list = ["fall_bed_data" + str(i) for i in range(1, 5)]
    print(dir_list)
    BATCH_SIZE = 16
    MAX_POINTS = 100
    MAX_FRAMES = 40
    EPOCHS = 300
    LEARNING_RATE = 0.0001
    PROCESSING_METHOD = 'mask'

    dataset = RadarDataset(txt_dir, file_path_prefix=file_path_prefix, dir_list=dir_list, max_points=MAX_POINTS,
                           max_frames=MAX_FRAMES, method=PROCESSING_METHOD)
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_set, val_set = torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4)

    # model = EnhancedTCN(num_classes=2, max_frames=50, dropout=0.5).to(device)
    # model = LightweightSpatioTemporalModel(num_classes=2, dropout=0.5).to(device)
    model = HybridModel(num_classes=2, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 使用 CosineAnnealingLR 调整学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 检查点恢复逻辑
    start_epoch = 1
    best_val_loss = float('inf')
    counter = 0
    saved_checkpoints = []  # 用于跟踪保存的检查点
    # str_time = time.strftime("%Y%m%d_%H%M%S")
    str_time = "111"

    # CHECKPOINT_PATH = 'F:\\code\\rada\\fall_bed\\enhanced_tcn_model_20250413_170855.pth'
    CHECKPOINT_PATH = None
    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        counter = checkpoint['counter']
        print(f"Resuming training from epoch {start_epoch}")

    # 训练循环
    log_file = "training_log" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
    patience = 3

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, PROCESSING_METHOD, log_file)
        val_loss = test(model, device, val_loader, criterion, log_file)
        scheduler.step()

        # 保存检查点（每个epoch都保存）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"./checkpoint/{str_time}/checkpoint_epoch{epoch}_{timestamp}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'counter': counter,
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_name)

        # 维护最多10个检查点
        saved_checkpoints.append(checkpoint_name)
        if len(saved_checkpoints) > 10:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            best_model_name = f"./best_model/{str_time}/best_model_epoch{epoch}_{timestamp}.pth"
            torch.save(model.state_dict(), best_model_name)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("训练完成，最终模型已保存。")