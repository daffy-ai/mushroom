# Mushroom Health Detection

This project detects healthy vs. unhealthy mushrooms using image classification.

## Setup

1. Create and activate your Python virtual environment.
2. Install dependencies:
   ```bash
   .venv\Scripts\pip.exe install -r requirements.txt
   ```

## Download dataset

The dataset is downloaded from Mendeley.

```bash
python data/download.py
```

If the automatic download is blocked, download the archive manually from the dataset page and save it as `data/mushroom_disease.zip`.

## Train

```bash
python src/train.py
```

## Evaluate

```bash
python -m src.evaluate
```

## Run demo

```bash
streamlit run app.py
```

## Model Performance

- **Validation Accuracy:** 92.11%
- **Test Accuracy:** 84.35%
- **Precision:** 87.14%
- **Recall:** 87.14%
- **F1 Score:** 87.14%

Trained on 532 training samples and validated on 114 samples using ResNet18 transfer learning.
