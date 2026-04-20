import torch
import torch.nn as nn
import torch.optim as optim
from .preprocess import get_data_loaders
from .model import get_model
import os

def train_model(epochs=50, batch_size=16, lr=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size)
    print(f'Training samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}')

    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'  -> Best accuracy improved to {best_acc:.4f}, model saved.')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break

        scheduler.step()

    print(f'Training complete. Best val accuracy: {best_acc:.4f}')

if __name__ == '__main__':
    train_model()