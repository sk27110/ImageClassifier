import os
import kagglehub

def download_dataset(dataset_name: str, download_dir: str) -> str:
    """
    Скачивает датасет с Kaggle в указанную директорию.
    
    Args:
        dataset_name (str): имя датасета в формате "username/dataset"
        download_dir (str): путь к папке, куда сохранить датасет

    Returns:
        str: путь к распакованному датасету
    """
    try:
        os.environ["KAGGLEHUB_CACHE"] = download_dir
        path = kagglehub.dataset_download(dataset_name)
        print(f"✅ Датасет {dataset_name} скачан в: {path}")
        return path
    except:
        raise NotImplementedError
