import torch
from tqdm import tqdm

class Trainer:
    """
    Минимальный тренер для классификации.
    """
    def __init__(self, model, criterion, optimizer, device, train_loader, val_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, num_epochs=5):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_one_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            if self.val_loader is not None:
                val_loss, val_acc = self.validate()
                print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # --- прямой проход ---
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # --- обратное распространение ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return running_loss / len(self.val_loader.dataset), correct / total
