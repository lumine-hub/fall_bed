# Enhanced training loop with learning rate scheduler and more metrics
import torch
import time
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# 设置日志记录
def setup_logging():
    # 训练日志
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.INFO)
    train_handler = logging.FileHandler(f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    train_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    train_logger.addHandler(train_handler)
    
    # 错误预测日志
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.INFO)
    error_handler = logging.FileHandler(f'error_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    error_logger.addHandler(error_handler)
    
    return train_logger, error_logger

# 训练过程可视化
class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.base_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def update(self, train_loss, val_loss, train_acc, val_acc, epoch):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # 每个epoch都更新图表
        self.plot(epoch)
        
    def plot(self, epoch):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title(f'Loss Curves (Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Val Accuracy')
        plt.title(f'Accuracy Curves (Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_curves_{self.base_filename}_epoch{epoch+1}.png')
        plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    best_model_state = None

    train_logger, error_logger = setup_logging()
    visualizer = TrainingVisualizer()

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    # 添加梯度裁剪
    max_grad_norm = 1.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{num_epochs} {'=' * 20}")

        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 添加梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 添加更频繁的进度打印
            if batch_idx % 50 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f}')
            
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_acc = 100. * correct / len(val_loader.dataset)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step(val_loss)

        # Metrics computation
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        acc_score = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'])

        end_time = time.time()
        epoch_time = end_time - start_time

        # Print metrics
        print(f"Train Loss: {avg_loss:.6f}, Train Acc: {100. * correct / len(train_loader.dataset):.2f}%")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Accuracy: {acc_score:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

        # Early stopping and saving best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if best_model_state is None:
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pth')
                print(f"Saved new best model with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # 记录训练信息
        train_logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                         f'Train Loss: {avg_loss:.6f}, '
                         f'Val Loss: {val_loss:.6f}, '
                         f'Val Acc: {val_acc:.2f}%, '
                         f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 更新可视化数据并保存当前epoch的图表
        visualizer.update(avg_loss, val_loss, 100. * correct / len(train_loader.dataset), val_acc, epoch)
        
        # 记录错误预测
        error_logger.info(f'\n=== Epoch {epoch+1} Error Predictions ===')
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            wrong_idx = (predicted != target).nonzero().squeeze()
            if wrong_idx.numel() > 0:
                for idx in wrong_idx:
                    error_logger.info(f'True Label: {target[idx]}, '
                                    f'Predicted: {predicted[idx]}, '
                                    f'Input: {data[idx]}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs


# Focal Loss for handling class imbalance
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
