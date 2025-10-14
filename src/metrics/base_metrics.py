# from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

# def get_classification_metrics(cfg, device: str = "cpu"):
#     """
#     Создаёт стандартные метрики для мультиклассовой классификации.

#     Args:
#         num_classes (int): число классов
#         device (str|torch.device): на каком устройстве создавать метрики

#     Returns:
#         dict: словарь с метриками
#     """
#     metrics = {
#         "accuracy": Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes).to(device),
#         "precision": Precision(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device),
#         "recall": Recall(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device),
#         "f1": F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device)
#     }
#     return metrics



import torch
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

class ClassificationMetrics:
    """
    Класс для стандартных метрик мультиклассовой классификации.
    Содержит accuracy, precision, recall и F1.
    """

    def __init__(self, num_classes: int, device: str = "cpu"):
        self.device = device
        self.num_classes = num_classes

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

        self._metrics = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }