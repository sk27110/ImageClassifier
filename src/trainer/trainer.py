import torch
from tqdm import tqdm
import logging

logger = logging.getLogger("train")

class Trainer:
    """
    Универсальный тренер для классификации. 
    Метрики считаются только на валидации, на трейне только loss.
    """
    def __init__(self, model, criterion, optimizer, device, train_loader, scheduler, metrics=None, val_loader=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.metrics = metrics or {}

        self.model.to(self.device)

        logger.info(f"📌 Model device: {next(model.parameters()).device}")
        logger.info(f"📌 Batch device: {device}")

    def train(self, num_epochs=5):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 1. Обучаем одну эпоху и получаем train loss
            train_loss = self._train_one_epoch()
            train_losses.append(train_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"📊 LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}")
            
            # 2. Валидация (если есть)
            val_loss = None
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate()
                val_losses.append(val_loss)
                
                # Сохраняем лучшую модель
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # torch.save(self.model.state_dict(), 'best_model.pth')
                    logger.info(f"🏆 New best model! Val Loss: {val_loss:.4f}")
                
                # Обновляем scheduler на основе валидационного loss
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
                    # Логируем изменение LR
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        logger.info(f"🔄 Learning Rate updated: {new_lr:.2e}")
            
            # 3. Если нет валидации, обновляем scheduler на основе train loss
            elif self.scheduler is not None:
                self.scheduler.step()
        
        # Финальная статистика
        if self.val_loader is not None:
            logger.info(f"🎯 Training completed! Best Val Loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses

    def _train_one_epoch(self):
        """Обучаем одну эпоху и возвращаем средний loss"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            
            batch_size = labels.size(0)
            total_samples += batch_size

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size

        avg_loss = running_loss / total_samples
        return avg_loss

    def _validate(self):
        """Валидация с метриками"""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                batch_size = labels.size(0)
                total_samples += batch_size

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * batch_size

                preds = torch.argmax(outputs, dim=1)
                self._update_metrics(preds, labels)

        avg_loss = running_loss / total_samples
        metric_results = self._compute_metrics()
        self._print_metrics(avg_loss, metric_results)
        
        return avg_loss, metric_results

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
                results[name] = metric()
            else:
                results[name] = None
        return results

    def _print_metrics(self, loss, metrics_dict):
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
        logger.info(f"Val Loss: {loss:.4f} | {metrics_str}")

    def test(self, test_loader):
        """Отдельный метод для тестирования на финальном датасете"""
        logger.info("🧪 Final testing on test set")
        test_loss, test_metrics = self._evaluate_loader(test_loader, "Test")
        return test_loss, test_metrics

    def _evaluate_loader(self, loader, prefix="Eval"):
        """Универсальная оценка на любом лоадере"""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {prefix}"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                batch_size = labels.size(0)
                total_samples += batch_size

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * batch_size

                preds = torch.argmax(outputs, dim=1)
                self._update_metrics(preds, labels)

        avg_loss = running_loss / total_samples
        metric_results = self._compute_metrics()
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metric_results.items()])
        logger.info(f"{prefix} Loss: {avg_loss:.4f} | {metrics_str}")
        
        return avg_loss, metric_results