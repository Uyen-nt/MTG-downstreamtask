# retain_micron/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from retain_micron.evaluate import calculate_class_weights, evaluate_topk_recall, debug_predictions_distribution

class FocalLoss(nn.Module):
    """Focal Loss để xử lý class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
            
def train_model(model, train_loader, val_loader, epochs, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_codes = model.n_codes
    
    # Tính class weights từ training data
    class_weights = calculate_class_weights(train_loader, n_codes)
    class_weights = class_weights.to(device)
    
    # Sử dụng Focal Loss hoặc Weighted BCE Loss
    #criterion = FocalLoss(alpha=0.75, gamma=0.5)
    pos_weight = (class_weights < 10).float() * 50 + (class_weights >= 10).float() * 5
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    os.makedirs(save_path, exist_ok=True)

    best_recall = 0.0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for visits, labels in progress_bar:
            visits = [v for v in visits]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(visits)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        if (epoch + 1) % 5 == 0:
            
            model.eval()
            recall30 = evaluate_topk_recall(model, val_loader, k=30)
            
            # Phân tích phân phối predictions
            dist_analysis = debug_predictions_distribution(model, val_loader, n_codes)
            unique_predicted = dist_analysis['unique_codes_predicted']
            
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
            print(f"  Recall@30: {recall30:.4f}, Unique codes: {unique_predicted}/{n_codes}")
            
            # Tiêu chí lựa chọn model tốt hơn
            score = recall30 * (unique_predicted / n_codes)  # Balance recall và diversity
            
            if best_recall == 0.0:
                print("No improvement found based on score, saving last epoch model...")
                torch.save(model.state_dict(), f"{save_path}/retain_last.pth")
                model_path = f"{save_path}/retain_last.pth"
            else:
                model_path = f"{save_path}/retain_best.pth"
            
            print(f"Training completed! Best score: {best_recall:.4f}")
            
            scheduler.step(avg_loss)

    print(f"✅ Training completed! Best score: {best_recall:.4f}")
