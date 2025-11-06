import torch
from tqdm import tqdm
import logging

logger = logging.getLogger("train")

class Trainer:
    """
    Универсальный тренер для классификации с единым подсчётом метрик.
    Метрики считаются в режиме eval для train/val/test, чтобы сравнивать корректно.
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
        best_epoch = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            self._train_one_epoch()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"📊 Current Learning Rate: {current_lr:.2e}")
            
            # Оцениваем на тренировочных данных
            train_metrics = self.evaluate(self.train_loader, prefix="Train")
            
            # Оцениваем на валидационных данных (если есть)
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader, prefix="Val")
                
                # Сохраняем лучшую модель
                if 'loss' in val_metrics and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_epoch = epoch + 1
                    # Сохраняем модель (добавьте путь для сохранения)
                    # torch.save(self.model.state_dict(), f'best_model_epoch_{epoch+1}.pth')
                    logger.info(f"🏆 New best model! Val Loss: {best_val_loss:.4f}")
                
                # Обновляем scheduler на основе валидационной потери
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step'):
                        # Для ReduceLROnPlateau передаем валидационный loss
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            # Для других scheduler'ов (StepLR, CosineAnnealing, etc.)
                            self.scheduler.step()
                    
                    # Логируем новый LR после обновления
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr != current_lr:
                        logger.info(f"🔄 Learning Rate updated: {new_lr:.2e}")
            
            # Если нет валидации, обновляем scheduler на основе тренировочной потери
            elif self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
        
        # Финальное сообщение
        if self.val_loader is not None:
            logger.info(f"🎯 Training completed! Best model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * labels.size(0)

        avg_loss = running_loss / len(self.train_loader.dataset)
        logger.info(f"Train step Loss: {avg_loss:.4f}")

    def evaluate(self, loader, prefix="Val"):
        """
        Единый способ подсчёта метрик в режиме eval.
        Можно использовать для train/val/test.
        """
        self.model.eval()
        running_loss = 0.0
        self._reset_metrics()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating {prefix}"):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                preds = torch.argmax(outputs, dim=1)
                self._update_metrics(preds, labels)

        avg_loss = running_loss / len(loader.dataset)
        metric_results = self._compute_metrics()
        metric_results['loss'] = avg_loss  # Добавляем loss в результаты метрик
        self._print_metrics(prefix, avg_loss, metric_results)
        return metric_results

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

    def _print_metrics(self, mode, loss, metrics_dict):
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items() if k != 'loss'])
        logger.info(f"{mode} Loss: {loss:.4f} | {metrics_str}")