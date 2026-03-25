# text-classification (PyTorch text classification)

A small PyTorch project for training/benchmarking text classifiers on DBpedia.

---

## ✅ Quick Overview

| Folder | Purpose |
|-------|---------|
| `data/` | Dataset CSVs (`train.csv`, `test.csv`) |
| `models/` | Saved checkpoints (`models/<model>/...`) |
| `src/` | Training script + models + utils |

---

## ✅ Requirements

- **Python 3.11** (recommended)
- **PyTorch 2.x**
- **torchtext 0.15.x**

### Install dependencies (Windows)

```powershell
cd text-classification
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 📁 Dataset layout

Place your DBpedia CSVs under `data/`.

**Required format** (CSV):
- Columns: `class,title,content`

Example:
- `data/train.csv`
- `data/test.csv`

> You can still store raw copies elsewhere (e.g. `data/raw/`), but the scripts default to `data/train.csv` and `data/test.csv`.

---

## 🚀 Train (Quick Start)

Run from repo root (inside `text-classification/`):

```powershell
python -m src.train --config_file config.yml
```

### Model-specific config (YAML)

`src/train.py` reads YAML config (the script loads `yaml.safe_load`).
Set common options at top level and model-specific options under `transformer` / `rnn`:

```yaml
model: transformer  # options: transformer, rnn
train_csv: data/train.csv
test_csv: data/test.csv
output_dir: models
max_len: 128
batch_size: 128
epochs: 5
max_rows: null
lr: 0.0002
weight_decay: 0.00001
max_vocab: 50000
seed: 42
save_name: null
save_all: false

transformer:
  embed_dim: 128
  num_heads: 4
  hidden_dim: 256
  num_layers: 2
  dropout: 0.1

rnn:
  embed_dim: 128
  hidden_dim: 256
  num_layers: 2
  dropout: 0.1
```


### Train RNN model

```powershell
python -m src.train --config_file config.yml
```

`train.py` currently supports only one CLI argument:

- `--config_file`: path to a YAML config file (default `config.yml`)

All other training parameters are read from the config YAML


## 🖥️ Streamlit UI (Simple inference)

1) Install dependencies:

```powershell
pip install streamlit
```

2) Run the app from repository root (`text-classification`):

```powershell
streamlit run streamlit_app.py
```

3) On the Streamlit page:
- Enter the path to `config.yml` (default `config.yml` in root).
- Upload a checkpoint file (`.pth` or `.pt`) via the `Upload checkpoint file` button (required).
- Input text to classify.
- Click `Predict`.

4) Output:
- Predicted class label (0-based) and class name from `data/classes.txt`.

Notes:
- `config.yml` should include `model`, `max_len`, and `num_classes`.
- `data/classes.txt` should exist and list labels in order 0..13.
- The app uses uploaded checkpoint first; if not uploaded, it falls back to `checkpoint` in `config.yml` (if provided).


## 📌 Model checkpoint saving (auto-numbering + naming)

Checkpoints are stored under:

- `models/transformer/` (Transformer)
- `models/rnn/` (RNN)


### Custom name
```powershell
python -m src.train --model transformer --save_name my_run
```
=> `models/transformer/my_run.pth`

### Keep all improving checkpoints
```powershell
python -m src.train --model transformer --save_all
```
=> saves extra checkpoints like:
- `.../001_epoch03_val0.9123_20260319_142212.pth`

---

## 🧠 Model defaults (if you don’t override)

### Transformer defaults
| Param | Default |
|------|--------|
| `embed_dim` | 128 |
| `num_heads` | 4 |
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.1 |

### RNN defaults
| Param | Default |
|------|--------|
| `embed_dim` | 128 |
| `hidden_dim` | 256 |
| `num_layers` | 2 |
| `dropout` | 0.1 |

---

## 🧪 Inference (Predict)

`src/predict.py` CLI options:

- `--config_file`: path to configuration YAML (default `config.yml`)
- `--checkpoint`: optional override checkpoint path (if not provided, read from config)
- `--input_text`: single text string to classify
- `--input_file`: file path with one text sample per line
- `--max_len`: optional override sequence length (default from config or 128)

Example (single text):

```powershell
python -m src.predict --config_file config.yml \
  --checkpoint models/transformer/001.pth \
  --input_text "This is a test sentence"
```

Example (batch text file):

```powershell
python -m src.predict --config_file config.yml \
  --checkpoint models/transformer/001.pth \
  --input_file input.txt
```

If `--checkpoint` is missing, `predict.py` will use the `checkpoint` field from `config.yml` (if present).

---

## 📂 Project structure

- `data/` — Dataset CSVs
- `models/` — Saved checkpoints by model type
- `src/` — Code
  - `src/train.py` — training script
  - `src/predict.py` — inference script
  - `src/models/` — model definitions (Transformer/RNN)
  - `src/utils/` — preprocessing + helpers
