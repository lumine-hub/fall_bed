

# Enhanced training loop with learning rate scheduler and more metrics
import torch
import time
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device='cuda', early_stopping_patience=5,
                save_best=True, save_path='best_model.pth', class_weights=None):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{num_epochs} {'=' * 20}")

        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for inputs, masks, labels, frame_nums in train_loader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, masks)

            loss = torch.nn.functional.cross_entropy(outputs, labels,
                                                     weight=class_weights) if class_weights is not None else criterion(
                outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        train_losses.append(train_loss / train_total)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, masks, labels, frame_nums in val_loader:
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                outputs = model(inputs, masks)
                loss = torch.nn.functional.cross_entropy(outputs, labels,
                                                         weight=class_weights) if class_weights is not None else criterion(
                    outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_losses.append(val_loss / val_total)
        val_accs.append(val_acc)

        # Update learning rate
        if scheduler:
            scheduler.step(val_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

        # Metrics computation
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted')
        acc_score = accuracy_score(val_labels, val_preds)
        conf_matrix = confusion_matrix(val_labels, val_preds)
        class_report = classification_report(val_labels, val_preds, target_names=['Class 0', 'Class 1'])

        end_time = time.time()
        epoch_time = end_time - start_time

        # Print metrics
        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}")
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
            if save_best:
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, save_path)
                print(f"Saved new best model with val_acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    if save_best and best_model_state is not None:
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
