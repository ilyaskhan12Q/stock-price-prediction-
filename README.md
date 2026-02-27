# 📈 Stock Price Prediction — RNN · LSTM · Attention

A clean, modular Python project that trains three deep learning models on historical stock data and compares their performance.

| Model | Architecture |
|-------|-------------|
| Simple RNN | `SimpleRNN(64) → Dense(1)` |
| LSTM | `LSTM(64) → Dense(1)` |
| LSTM + Attention ⭐ | `LSTM(64, return_sequences=True) → AttentionLayer → Dense(1)` |

---

## 📂 Project Structure

```
stock-price-prediction/
├── src/
│   ├── train.py          # ← main entry point
│   ├── predict.py        # ← inference / future forecasting
│   ├── data_loader.py    # load from yfinance, CSV, or synthetic GBM
│   ├── preprocessor.py   # MinMaxScaler + sliding window sequences
│   ├── models.py         # RNN, LSTM, AttentionLayer, LSTM+Attention
│   ├── evaluate.py       # RMSE, MAE, comparison table
│   └── visualize.py      # all matplotlib charts
├── tests/
│   └── test_all.py       # pytest unit tests
├── data/                 # place your CSV files here (gitignored)
├── plots/                # generated at runtime (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/ilyaskhan12Q/stock-price-prediction.git
cd stock-price-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train all models (auto-downloads AAPL data)

```bash
python src/train.py
```

### 3. Customise

```bash
# Different ticker and longer window
python src/train.py --ticker TSLA --window 20 --epochs 20

# Load your own CSV
python src/train.py --csv data/MSFT.csv --window 15

# Skip Attention model (faster)
python src/train.py --no_attention
```

### 4. Forecast future prices

```bash
# First save a model inside train.py (see note below), then:
python src/predict.py --model_path saved_models/lstm_model.keras --days 30
```

> **Saving a model:** add `lstm_model.save("saved_models/lstm_model.keras")` inside `train.py` after training.

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

All 14 tests cover: data loading, validation, preprocessing, model output shapes, training smoke tests, and metric correctness.

---

## 📊 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ticker` | `AAPL` | Yahoo Finance ticker |
| `--start` | `2015-01-01` | Start date |
| `--end` | `2024-12-31` | End date |
| `--csv` | `None` | Path to local CSV (overrides ticker) |
| `--window` | `10` | Sliding window size (days) |
| `--epochs` | `10` | Training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--split` | `0.80` | Train/test split ratio |
| `--plots_dir` | `plots` | Output directory for charts |
| `--no_attention` | `False` | Skip LSTM+Attention model |

---

## 📈 Output Files

After training, the `plots/` directory contains:

| File | Description |
|------|-------------|
| `close_price.png` | Historical close + volume chart |
| `rnn_loss.png` | RNN train/val loss |
| `lstm_loss.png` | LSTM train/val loss |
| `attn_loss.png` | LSTM+Attention train/val loss |
| `rnn_predictions.png` | RNN predictions vs actual |
| `lstm_predictions.png` | LSTM predictions vs actual |
| `attn_predictions.png` | LSTM+Attention predictions vs actual |
| `all_predictions.png` | All models overlaid |
| `model_comparison.png` | RMSE & MAE bar chart |
| `metrics.json` | Numeric results (RMSE, MAE, time) |

---

## 🔬 Key Concepts

### Why LSTM outperforms Simple RNN
Simple RNNs suffer from the **vanishing gradient problem** — gradients shrink exponentially during backpropagation through time, preventing the model from learning patterns more than a few steps back.

LSTM solves this with three learnable gates:

| Gate | Role |
|------|------|
| **Forget** | Discard irrelevant past information |
| **Input** | Write new information to cell state |
| **Output** | Expose relevant memory to next timestep |

The **cell state** acts as a memory highway with additive updates, keeping gradients alive across 50–100+ timesteps.

### Why Attention helps further
Even LSTM compresses everything into a single final hidden state. The **Attention layer** has access to every hidden state in the sequence and learns to dynamically weight the most relevant time steps — e.g., a price spike 15 days ago might matter more than yesterday's value.

### Long-term dependency
A value at time *t* is influenced by values many steps earlier. For stock prices, monthly earnings cycles, quarterly patterns, and multi-week momentum are all long-term dependencies that require LSTM/Attention to capture.

---

## 📋 Dataset Requirements

Your CSV (if using `--csv`) must:
- Have at least **1,000 rows**
- Contain a **`Close`** column
- Contain a **`Date`** column (parseable by pandas)

Download free stock data from [Yahoo Finance](https://finance.yahoo.com) → Historical Data → Download.

---

## 🛠 Dependencies

- **TensorFlow ≥ 2.13** — model building & training
- **scikit-learn** — MinMaxScaler, RMSE/MAE
- **pandas / numpy** — data handling
- **matplotlib** — all visualisations
- **yfinance** — automatic data download (optional)
- **pytest** — unit tests

---

## 📄 License

MIT — free to use, modify, and distribute.
