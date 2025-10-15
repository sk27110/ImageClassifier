from torchvision import transforms
import ast


class Transforms:
    def __init__(
        self,
        mean, 
        std,
        resize
    ):
        self.mean = mean
        self.std = std
        self.resize = resize

        self.test = transforms.Compose([
            transforms.Resize(ast.literal_eval(self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            )
        ])

        self.train = transforms.Compose([
            transforms.Resize(ast.literal_eval(self.resize)),
            transforms.RandomRotation(20),  # случайный поворот ±20°
            transforms.RandomHorizontalFlip(p=0.5),  # отражение
            transforms.RandomVerticalFlip(p=0.1),  # иногда вертикально

            transforms.ColorJitter( 
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
            ),  # цветовые искажения

            transforms.RandomGrayscale(p=0.05),  # иногда делаем ч/б
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # небольшие сдвиги
                scale=(0.9, 1.1),      # масштабирование
                shear=10               # сдвиг по углу
            ),

            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # имитация перспективных искажений
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),   # немного размываем
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ast.literal_eval(self.mean),
                std=ast.literal_eval(self.std)
            )
        ])



