# retain_micron/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os

def train_model(model, train_loader, test_loader, n_codes, epochs=20, save_path="retain_micron/result"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(save_path, exist_ok=True)

    best_recall = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for visits, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            visits = [v for v in visits]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(visits)
            loss = criterion(logits.expand(labels.size(0), -1), labels).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

        # Eval mỗi 5 epoch
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            from .evaluate import evaluate_topk_recall
            recall10 = evaluate_topk_recall(model, test_loader, n_codes, k=10)
            recall20 = evaluate_topk_recall(model, test_loader, n_codes, k=20)
            print(f"Top-10 Recall: {recall10:.4f} | Top-20 Recall: {recall20:.4f}")

            if recall10 > best_recall:
                best_recall = recall10
                torch.save(model.state_dict(), f"{save_path}/retain_best.pth")
                print(f"New best model saved! Top-10: {recall10:.4f}")

    torch.save(model.state_dict(), f"{save_path}/retain_final.pth")
    print("Training completed!")
