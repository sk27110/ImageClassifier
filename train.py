import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

from src.dataset.load_dataset import download_dataset
from src.dataset.gtsrb_dataset import GtsrbDataset
from src.transforms.transforms import get_normalize_transform
from src.model.base_model import GTSRBCNN
from src.metrics.base_metrics import get_classification_metrics
from src.trainer import Trainer  # наш минимальный тренер


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # ------------------------------
    # 1️⃣ Конфигурация и скачивание датасета
    # ------------------------------
    print("⚙ Конфигурация:\n", OmegaConf.to_yaml(cfg))
    _ = download_dataset(cfg.dataset.name, cfg.dataset.download_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 2️⃣ Трансформации
    # ------------------------------
    transforms = get_normalize_transform(cfg)

    # ------------------------------
    # 3️⃣ Датасеты + train/val split
    # ------------------------------
    full_train_dataset = GtsrbDataset(
        data_path=cfg.dataset.path, mode="train", transforms=transforms
    )
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ------------------------------
    # 4️⃣ DataLoader
    # ------------------------------
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # ------------------------------
    # 5️⃣ Модель, loss, optimizer
    # ------------------------------
    model = GTSRBCNN(num_classes=cfg.dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # ------------------------------
    # 6️⃣ Метрики (для валидации можно использовать SimpleTrainer встроенные)
    # ------------------------------
    metrics = get_classification_metrics(cfg=cfg, device=device)
    # Но в SimpleTrainer метрики встроены только в validate() как точность

    # ------------------------------
    # 7️⃣ Создание и запуск тренера
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
    test_dataset = GtsrbDataset(
        data_path=cfg.dataset.path, mode="test", transforms=transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    test_loss, test_acc = trainer.validate()  # используем метод validate
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()