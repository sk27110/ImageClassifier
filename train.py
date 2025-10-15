import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Subset
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
import logging
from src.dataset.load_dataset import download_dataset
from src.trainer import Trainer
import os


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log_dir = os.getcwd()
    log_path = os.path.join(log_dir, "train.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("main")

    logger.info("\n" + "=" * 50)
    logger.info("⚙ Текущая конфигурация Hydra:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("=" * 50 + "\n")

    # ------------------------------
    # 1️⃣ Конфигурация и скачивание датасета
    # ------------------------------

    _ = download_dataset(cfg.load_dataset.name, cfg.load_dataset.download_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 2️⃣ Трансформации
    # ------------------------------
    transforms = instantiate(cfg.transforms)

    # ------------------------------
    # 3️⃣ Создание датасета и разделение на train/val
    # ------------------------------
    full_dataset = instantiate(cfg.dataset, mode="train", transforms=None)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_indices, val_indices = torch.utils.data.random_split(
        list(range(len(full_dataset))),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(1)
    )

    train_dataset = instantiate(cfg.dataset, mode="train", transforms=transforms.train)
    val_dataset = instantiate(cfg.dataset, mode="train", transforms=transforms.test)

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # ------------------------------
    # 4️⃣ DataLoader’ы
    # ------------------------------
    train_loader = instantiate(cfg.dataloader, dataset=train_dataset)
    val_loader = instantiate(cfg.dataloader, dataset=val_dataset)

    # ------------------------------
    # 5️⃣ Модель, функция потерь, оптимизатор
    # ------------------------------
    model = instantiate(cfg.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # ------------------------------
    # 6️⃣ Метрики
    # ------------------------------
    metrics = instantiate(cfg.metrics, device=device)._metrics

    # ------------------------------
    # 7️⃣ Обучение
    # ------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        metrics=metrics
    )

    trainer.train(num_epochs=cfg.train.num_epoch)

    # ------------------------------
    # 8️⃣ Тестирование
    # ------------------------------
    test_dataset = instantiate(cfg.dataset, mode="test", transforms=transforms.test)
    test_loader = instantiate(cfg.dataloader, dataset=test_dataset)

    _ = trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
