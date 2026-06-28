# IA-Aplicada

Proyecto de clasificación de enfermedades en hojas de papa usando el
PlantVillage Potato Disease Dataset.

## Estructura

```
IA-Aplicada/
├── data/                 # Dataset (NO versionado, ver abajo)
├── scripts/              # Scripts auxiliares
│   └── download_data.py  # Descarga el dataset desde Kaggle
├── .gitignore
└── README.md
```

## Dataset

Este proyecto usa el
[PlantVillage Potato Disease Dataset](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset).

El dataset **no está incluido** en el repositorio (es pesado y son archivos
binarios). Descárgalo con cualquiera de estas opciones; quedará en `data/`.

### Opción A — kagglehub (recomendada)

```bash
pip install kagglehub
python scripts/download_data.py
```

### Opción B — Kaggle CLI

1. Crea tu token en Kaggle: **Account → Create New API Token**.
2. Coloca el archivo `kaggle.json` en `C:\Users\<usuario>\.kaggle\`.
3. Ejecuta:

```bash
pip install kaggle
kaggle datasets download -d aarishasifkhan/plantvillage-potato-disease-dataset -p data/ --unzip
```

### Opción C — Manual

Descarga el ZIP desde la web de Kaggle y descomprímelo dentro de `data/`.
