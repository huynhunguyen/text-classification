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

### Model-specific config (JSON or YAML)

`src/train.py` supports both JSON and YAML input.  
Set common options at top level and model-specific options under `transformer` / `rnn`:

```json
{
  "model": "transformer",
  "batch_size": 128,
  "epochs": 5,
  "transformer": {
    "embed_dim": 128,
    "num_heads": 4,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1
  },
  "rnn": {
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1
  }
}
```

### Train RNN model

```powershell
# change model in config.json to "rnn" then run
python -m src.train --config_file config.json
```

### Common options (CLI)

| Option | Default | Description |
|-------|--------|-------------|
| `--model` | `transformer` | `transformer` or `rnn` |
| `--max_rows` | (none) | Limit number of samples (quick test) |
| `--max_len` | `128` | Max token length |
| `--batch_size` | `128` | Batch size |
| `--epochs` | `5` | Training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |

---

## 📌 Model checkpoint saving (auto-numbering + naming)

Checkpoints are stored under:

- `models/transformer/` (Transformer)
- `models/rnn/` (RNN)

### Auto numbered (default)
If you run train repeatedly, the code auto-numbers:
- `models/transformer/001.pth`
- `models/transformer/002.pth`

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

Use `src/predict.py` and point it to a checkpoint:

```powershell
python -m src.predict --model transformer \
  --checkpoint models/transformer/001.pth \
  --train_csv data/train.csv \
  --input_text "This is a test sentence"
```

Or use a file with one sentence per line:

```powershell
python -m src.predict --model transformer \
  --checkpoint models/transformer/001.pth \
  --train_csv data/train.csv \
  --input_file input.txt
```

---

## 📂 Project structure

- `data/` — Dataset CSVs
- `models/` — Saved checkpoints by model type
- `src/` — Code
  - `src/train.py` — training script
  - `src/predict.py` — inference script
  - `src/models/` — model definitions (Transformer/RNN)
  - `src/utils/` — preprocessing + helpers
