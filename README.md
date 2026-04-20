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
python src/evaluate.py
```

## Run demo

```bash
streamlit run app.py
```
