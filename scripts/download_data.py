"""
Descarga el dataset PlantVillage Potato Disease desde Kaggle
y lo copia a la carpeta data/ del proyecto.
"""
import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def main() -> None:

    cache_path = Path(
        kagglehub.dataset_download(
            "aarishasifkhan/plantvillage-potato-disease-dataset",
        )
    )
    print(f"Dataset descargado en: {cache_path}")

    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        dest = DATA_DIR / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print(f"Dataset disponible en: {DATA_DIR}")


if __name__ == "__main__":
    main()
