import torch
from tqdm import tqdm

class Trainer:
    """
    Универсальный тренер для классификации с любыми совместимыми метриками.
    """
    def __init__(self, model, criterion, optimizer, device, train_loader, metrics=None, val_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics or {}  # словарь метрик {"accuracy": metric_obj, ...}

        print(f"📌 Model device: {next(model.parameters()).device}")
        for batch in train_loader:
            print(f"📌 First batch device: {batch['image'].device}")
            break

    def train(self, num_epochs=5):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss, train_metrics = self.train_one_epoch()
            self._print_metrics("Train", train_loss, train_metrics)
            
            if self.val_loader is not None:
                val_loss, val_metrics = self.validate()
                self._print_metrics("Val", val_loss, val_metrics)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        self._reset_metrics()

        for batch in tqdm(self.train_loader, desc="Training"):
            batch_loss, batch_preds, batch_labels = self._forward_batch(batch, train=True)
            running_loss += batch_loss * batch_labels.size(0)
            self._update_metrics(batch_preds, batch_labels)

        avg_loss = running_loss / len(self.train_loader.dataset)
        metric_results = self._compute_metrics()
        return avg_loss, metric_results

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch_loss, batch_preds, batch_labels = self._forward_batch(batch, train=False)
                running_loss += batch_loss * batch_labels.size(0)
                self._update_metrics(batch_preds, batch_labels)

        avg_loss = running_loss / len(self.val_loader.dataset)
        metric_results = self._compute_metrics()
        return avg_loss, metric_results
    
    def evaluate(self, loader):
        """
        Оценивает модель на любом DataLoader.
        
        Args:
            loader (DataLoader): даталоадер для оценки (валидация, тест и т.д.)
        
        Returns:
            loss (float), metrics_dict (dict)
        """
        self.model.eval()
        running_loss = 0.0
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                batch_loss, batch_preds, batch_labels = self._forward_batch(batch, train=False)
                running_loss += batch_loss * batch_labels.size(0)
                self._update_metrics(batch_preds, batch_labels)

        test_loss = running_loss / len(loader.dataset)
        test_metrics = self._compute_metrics()
        self._print_metrics("Val", test_loss, test_metrics)
        return test_loss, test_metrics


    def _forward_batch(self, batch, train=True):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        return loss.item(), preds, labels

    def _reset_metrics(self):
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()

    def _update_metrics(self, preds, labels):
        for metric in self.metrics.values():
            if hasattr(metric, "update"):
                metric.update(preds, labels)

    def _compute_metrics(self):
        results = {}
        for name, metric in self.metrics.items():
            if hasattr(metric, "compute"):
                results[name] = metric.compute().item()
            elif callable(metric):
                results[name] = metric()  # fallback
            else:
                results[name] = None
        return results

    def _print_metrics(self, mode, loss, metrics_dict):
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
        print(f"{mode} Loss: {loss:.4f} | {metrics_str}")
