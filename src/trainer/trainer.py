import torch
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datetime import datetime
import os

logger = logging.getLogger("train")

class Trainer:
    """
    Универсальный тренер для классификации. 
    Метрики считаются только на валидации, на трейне только loss.
    """
    def __init__(self, model, criterion, optimizer, device, train_loader, scheduler, metrics=None, val_loader=None, log_dir=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.metrics = metrics or {}

        self.model.to(self.device)

        # Инициализация TensorBoard
        if log_dir is None:
            log_dir = f"runs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        logger.info(f"📊 TensorBoard logging to: {log_dir}")

        # Логируем граф модели
        self._log_model_graph()

        logger.info(f"📌 Model device: {next(model.parameters()).device}")
        logger.info(f"📌 Batch device: {device}")

    def _log_model_graph(self):
        """Логирует архитектуру модели в TensorBoard"""
        try:
            if self.train_loader is not None:
                # Берем один батч для построения графа
                sample_batch = next(iter(self.train_loader))
                sample_images = sample_batch["image"][:1].to(self.device)  # один пример для графа
                self.writer.add_graph(self.model, sample_images)
                logger.info("✅ Model graph logged to TensorBoard")
        except Exception as e:
            logger.warning(f"⚠️ Could not log model graph: {e}")

    def train(self, num_epochs=5, patience=7, min_epochs=0):
        """
        Обучение с Early Stopping
        
        Args:
            num_epochs: максимальное количество эпох
            patience: сколько эпох ждать без улучшения перед остановкой
            min_epochs: минимальное количество эпох перед началом проверки early stopping
        """
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        epochs_without_improvement = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 1. Обучаем одну эпоху и получаем train loss
            train_loss = self._train_one_epoch(epoch)
            train_losses.append(train_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"📊 LR: {current_lr:.2e}, Train Loss: {train_loss:.4f}")
            
            # Логируем train loss и learning rate
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 2. Валидация (если есть)
            val_loss = None
            if self.val_loader is not None:
                val_loss, val_metrics = self._validate(epoch)
                val_losses.append(val_loss)
                
                # Логируем validation loss и метрики
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                
                # Логируем сравнение train/val loss
                self.writer.add_scalars('Loss_comparison', {
                    'train': train_loss,
                    'val': val_loss
                }, epoch)
                
                # Сохраняем лучшую модель
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    logger.info(f"🏆 New best model! Val Loss: {val_loss:.4f}")
                    
                    # Логируем веса модели для лучшей эпохи
                    if epoch % 5 == 0:  # Чтобы не перегружать, логируем каждые 5 улучшений
                        self._log_model_weights(epoch)
                else:
                    epochs_without_improvement += 1
                    logger.info(f"⏳ No improvement for {epochs_without_improvement}/{patience} epochs")
                
                # Проверка Early Stopping
                if (epoch + 1) >= min_epochs and epochs_without_improvement >= patience:
                    logger.info(f"🛑 Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    break
                
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
            
            # Логируем изображения каждые 10 эпох
            if epoch % 10 == 0:
                self._log_sample_images(epoch)
        
        # Финальная статистика
        if self.val_loader is not None:
            logger.info(f"🎯 Training completed! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
            
            # Загружаем лучшую модель для последующего использования
            self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
            logger.info("✅ Best model loaded for final evaluation")
        
        # Закрываем writer
        self.writer.close()
        logger.info("📊 TensorBoard writer closed")
        
        return train_losses, val_losses

    def _train_one_epoch(self, epoch):
        """Обучаем одну эпоху и возвращаем средний loss"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
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
            
            # Логируем batch loss каждые 50 батчей
            if batch_idx % 50 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train_batch', loss.item(), step)

        avg_loss = running_loss / total_samples
        return avg_loss

    def _validate(self, epoch):
        """Валидация с метриками"""
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        self._reset_metrics()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                batch_size = labels.size(0)
                total_samples += batch_size

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * batch_size

                preds = torch.argmax(outputs, dim=1)
                self._update_metrics(preds, labels)
                
                # Логируем примеры изображений из валидации каждые 10 эпох
                if epoch % 10 == 0 and batch_idx == 0:
                    self._log_val_images(images[:8], preds[:8], labels[:8], epoch)

        avg_loss = running_loss / total_samples
        metric_results = self._compute_metrics()
        self._print_metrics(avg_loss, metric_results)
        
        return avg_loss, metric_results

    def _log_model_weights(self, epoch):
        """Логирует распределение весов модели"""
        try:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.numel() > 1:  # Пропускаем bias и т.д.
                    self.writer.add_histogram(f'weights/{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
        except Exception as e:
            logger.warning(f"⚠️ Could not log model weights: {e}")

    def _log_sample_images(self, epoch):
        """Логирует примеры изображений из тренировочного набора"""
        try:
            if self.train_loader is not None:
                batch = next(iter(self.train_loader))
                images = batch["image"][:8]  # Берем 8 изображений
                grid = torchvision.utils.make_grid(images)
                self.writer.add_image('train_samples', grid, epoch)
        except Exception as e:
            logger.warning(f"⚠️ Could not log sample images: {e}")

    def _log_val_images(self, images, predictions, labels, epoch):
        """Логирует примеры изображений из валидации с предсказаниями"""
        try:
            # Создаем grid из изображений
            grid = torchvision.utils.make_grid(images)
            self.writer.add_image('val_samples', grid, epoch)
            
            # Можно добавить дополнительные логи, например, confusion matrix
            # если у вас есть соответствующие метрики
        except Exception as e:
            logger.warning(f"⚠️ Could not log validation images: {e}")

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
        
        # Логируем финальные метрики
        self.writer.add_scalar('Loss/test_final', test_loss)
        for metric_name, metric_value in test_metrics.items():
            self.writer.add_scalar(f'Metrics/test_{metric_name}', metric_value)
        
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