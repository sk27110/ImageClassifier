import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import random_split, Subset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
import logging
from src.dataset.load_dataset import download_dataset
from src.trainer import Trainer
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

@hydra.main(config_path="../../conf", config_name="food11_exp1", version_base="1.3")
def main(cfg: DictConfig):
    # Создаем тестовый датасет и лоадер
    transforms = instantiate(cfg.transforms)
    test_dataset = instantiate(cfg.dataset, mode="test", transforms=transforms.test)
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset, shuffle=False)

    # Загружаем модель
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load('models/best_model.pth', map_location='cpu'))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Создаем необходимые компоненты для Trainer
    criterion = torch.nn.CrossEntropyLoss()
    metrics = instantiate(cfg.metrics, device=device)._metrics
    
    # Создаем заглушки для ненужных параметров
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dummy_train_loader = None  # Не нужен для тестирования
    dummy_scheduler = None
    
    # Создаем Trainer ТОЛЬКО для тестирования
    trainer = Trainer(
        model=model,
        criterion=criterion,
        device=device,
        metrics=metrics,
    )
    
    # Используем метод test для оценки
    test_loss, test_metrics = trainer.test(test_loader)
    
    print("🎯 Результаты тестирования загруженной модели:")
    print(f"Test Loss: {test_loss:.4f}")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return test_loss, test_metrics

if __name__ == "__main__":
    main()