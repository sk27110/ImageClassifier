from torchvision import transforms
import ast


def get_normalize_transform(cfg):
    normalize_transform = transforms.Compose([
        transforms.Resize(ast.literal_eval(cfg.transforms.resize)),

        # Аугментации для тренировки
        transforms.RandomRotation(15),          # случайный поворот ±15 градусов
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # изменение яркости/контраста/насыщенности
        transforms.RandomHorizontalFlip(),      # случайное горизонтальное отражение (для некоторых знаков можно убрать, если знак не симметричный)

        transforms.ToTensor(),
        transforms.Normalize(
            mean=ast.literal_eval(cfg.transforms.mean),
            std=ast.literal_eval(cfg.transforms.std)
        )
    ])
    return normalize_transform