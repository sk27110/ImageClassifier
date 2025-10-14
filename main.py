import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra

from src.dataset.load_dataset import download_dataset
from src.dataset.gtsrb_dataset import GtsrbDataset
from src.transforms.transforms import get_normalize_transform
from src.dataloader.base_dataloader import get_dataloader
from src.loss.crossentropy_loss import loss_fn
from src.metrics.base_metrics import get_classification_metrics
from src.model.base_model import GTSRBCNN


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
    full_train_dataset = GtsrbDataset(data_path=cfg.dataset.path, mode="train", transforms=transforms)
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
    train_loader = get_dataloader(dataset=train_dataset, cfg=cfg)
    val_loader = get_dataloader(dataset=val_dataset, cfg=cfg)

    # ------------------------------
    # 5️⃣ Модель, loss, optimizer
    # ------------------------------
    model = GTSRBCNN(num_classes=cfg.dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # ------------------------------
    # 6️⃣ Метрики
    # ------------------------------
    metrics = get_classification_metrics(cfg=cfg, device=device)
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    # ------------------------------
    # 7️⃣ Цикл обучения
    # ------------------------------
    for epoch in range(cfg.train.num_epoch):
        # --- тренировка ---
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epoch} [Train]"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # --- валидация ---
        model.eval()
        accuracy.reset(); precision.reset(); recall.reset(); f1.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.train.num_epoch} [Val]"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1.update(preds, labels)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{cfg.train.num_epoch} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy.compute():.4f} | "
            f"Precision: {precision.compute():.4f} | "
            f"Recall: {recall.compute():.4f} | "
            f"F1: {f1.compute():.4f}"
        )



        # ------------------------------
    # 8️⃣ Тестирование на Test.csv
    # ------------------------------
    # Создаём тестовый датасет
    test_dataset = GtsrbDataset(data_path=cfg.dataset.path, mode="test", transforms=transforms)
    test_loader = get_dataloader(dataset=test_dataset, cfg=cfg)

    # Переводим модель в eval
    model.eval()
    accuracy.reset(); precision.reset(); recall.reset(); f1.reset()
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

    test_loss /= len(test_loader.dataset)

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Accuracy: {accuracy.compute():.4f} | "
        f"Precision: {precision.compute():.4f} | "
        f"Recall: {recall.compute():.4f} | "
        f"F1: {f1.compute():.4f}"
    )



if __name__ == "__main__":
    main()
