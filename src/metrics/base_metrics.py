from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

def get_classification_metrics(cfg, device: str = "cpu"):
    """
    Создаёт стандартные метрики для мультиклассовой классификации.

    Args:
        num_classes (int): число классов
        device (str|torch.device): на каком устройстве создавать метрики

    Returns:
        dict: словарь с метриками
    """
    metrics = {
        "accuracy": Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes).to(device),
        "precision": Precision(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device),
        "recall": Recall(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device),
        "f1": F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro").to(device)
    }
    return metrics
