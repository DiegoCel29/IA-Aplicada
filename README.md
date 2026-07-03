# IA-Aplicada

Proyecto de clasificación de enfermedades en hojas de papa usando el
PlantVillage Potato Disease Dataset.

## Estructura

```
IA-Aplicada/
├── data/                        # Dataset (NO versionado, ver abajo)
├── scripts/
│   └── download_data.py         # Descarga el dataset desde Kaggle
├── results/                     # Resultados guardados (JSON y figuras)
├── variantes/                   # Variantes experimentales de prueba del AG
│   ├── AG-CNN_EX1_variante.ipynb
│   └── AG-CNN_EX2_variante.ipynb
├── ag_core.py                   # Módulo del AG, Random Search y CNN (reutilizable)
├── AG-CNN_Comparativa.ipynb     # Corre los experimentos E1–E5 para ambos backbones
├── AG-CNN_EX1.ipynb             # Experimento con MobileNetV2
├── AG-CNN_EX2.ipynb             # Experimento con EfficientNetB0
├── requirements.txt
├── .gitignore
└── README.md
```

## Notebooks y scripts

- **`ag_core.py`**: módulo reutilizable con el Algoritmo Genético, el Random
  Search y las funciones de entrenamiento y evaluación de la CNN.
- **`AG-CNN_Comparativa.ipynb`**: notebook principal; importa `ag_core` y
  ejecuta los experimentos E1–E5 (Default, AG y Random Search) sobre
  MobileNetV2 y EfficientNetB0.
- **`variantes/`**: versiones de prueba de los notebooks EX1/EX2 con ajustes
  experimentales (fine-tuning en dos etapas, fitness determinista y learning
  rate en escala logarítmica). No forman parte de la comparativa principal.

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
