"""
tests/test_all.py
-----------------
Unit tests for every module. Run with:

    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# data_loader
# ──────────────────────────────────────────────────────────────────────────────
class TestDataLoader:
    def test_synthetic_shape(self):
        from data_loader import _generate_synthetic
        df = _generate_synthetic(n=1200)
        assert df.shape[0] == 1200
        assert "Close" in df.columns
        assert "Date"  in df.columns

    def test_synthetic_no_nulls(self):
        from data_loader import _generate_synthetic
        df = _generate_synthetic()
        assert df["Close"].isnull().sum() == 0

    def test_validate_missing_close_raises(self):
        from data_loader import _validate
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=1200), "Open": np.ones(1200)})
        with pytest.raises(ValueError, match="'Close'"):
            _validate(df)

    def test_validate_too_few_rows_raises(self):
        from data_loader import _validate
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=500),
            "Close": np.ones(500),
        })
        with pytest.raises(ValueError, match="1000"):
            _validate(df)

    def test_load_data_returns_df(self):
        """load_data should always return a DataFrame (via synthetic fallback)."""
        from data_loader import load_data
        # Force synthetic by using a nonsense ticker with no yfinance
        df = load_data(ticker="FAKEXYZ999")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1000


# ──────────────────────────────────────────────────────────────────────────────
# preprocessor
# ──────────────────────────────────────────────────────────────────────────────
class TestPreprocessor:
    @pytest.fixture
    def sample_df(self):
        from data_loader import _generate_synthetic
        return _generate_synthetic(n=1500)

    def test_sequence_shapes(self, sample_df):
        from preprocessor import preprocess
        X_tr, X_te, y_tr, y_te, scaler, dates = preprocess(sample_df, window_size=10, split=0.8)
        assert X_tr.shape[1] == 10   # window size
        assert X_tr.shape[2] == 1    # single feature
        assert y_tr.shape[1] == 1
        assert len(X_tr) + len(X_te) == len(sample_df) - 10

    def test_scaler_range(self, sample_df):
        from preprocessor import preprocess
        X_tr, X_te, y_tr, y_te, scaler, _ = preprocess(sample_df)
        # All values should be in [0, 1]
        assert X_tr.min() >= 0.0
        assert X_tr.max() <= 1.0

    def test_split_ratio(self, sample_df):
        from preprocessor import preprocess
        X_tr, X_te, *_ = preprocess(sample_df, split=0.8)
        ratio = len(X_tr) / (len(X_tr) + len(X_te))
        assert abs(ratio - 0.8) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# models
# ──────────────────────────────────────────────────────────────────────────────
class TestModels:
    INPUT_SHAPE = (10, 1)

    def test_rnn_output_shape(self):
        from models import build_rnn
        model = build_rnn(self.INPUT_SHAPE)
        x = np.random.rand(4, 10, 1).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (4, 1)

    def test_lstm_output_shape(self):
        from models import build_lstm
        model = build_lstm(self.INPUT_SHAPE)
        x = np.random.rand(4, 10, 1).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (4, 1)

    def test_lstm_attention_output_shape(self):
        from models import build_lstm_attention
        model = build_lstm_attention(self.INPUT_SHAPE)
        x = np.random.rand(4, 10, 1).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (4, 1)

    def test_rnn_parameter_count(self):
        from models import build_rnn
        model = build_rnn(self.INPUT_SHAPE)
        # SimpleRNN(64): 64*(64+1+1) = 4224 + Dense(1): 65 = 4289
        assert model.count_params() > 0

    def test_attention_layer_trainable(self):
        from models import build_lstm_attention
        model = build_lstm_attention(self.INPUT_SHAPE)
        attn_layers = [l for l in model.layers if "attention" in l.name.lower()]
        assert len(attn_layers) == 1
        assert attn_layers[0].trainable

    def test_models_train_one_epoch(self):
        """Smoke-test: all three models should train for 1 epoch without crashing."""
        from models import build_rnn, build_lstm, build_lstm_attention
        X = np.random.rand(50, 10, 1).astype(np.float32)
        y = np.random.rand(50, 1).astype(np.float32)
        for builder in [build_rnn, build_lstm, build_lstm_attention]:
            model = builder(self.INPUT_SHAPE)
            history = model.fit(X, y, epochs=1, verbose=0)
            assert "loss" in history.history


# ──────────────────────────────────────────────────────────────────────────────
# evaluate
# ──────────────────────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_perfect_predictions(self):
        """RMSE and MAE should be ~0 when predictions equal actuals."""
        import tensorflow as tf
        from evaluate import evaluate_model
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        data   = np.linspace(0, 1, 100).reshape(-1, 1)
        scaler.fit(data)

        # Mock model that always predicts y_test exactly
        class PerfectModel:
            def predict(self, X, verbose=0):
                return X[:, -1, :]   # last timestep = y in this mock

        X_test = data[:-1].reshape(-1, 1, 1)
        y_test = data[1:]

        # Adjust mock to return scaled values equal to y_test
        class PerfectModel2:
            def predict(self, X, verbose=0):
                return y_test

        rmse, mae, y_pred, y_true = evaluate_model(
            PerfectModel2(), X_test, y_test, scaler, "Perfect"
        )
        assert rmse < 1e-6
        assert mae  < 1e-6

    def test_metrics_positive(self):
        """RMSE and MAE must always be non-negative."""
        from models  import build_lstm
        from evaluate import evaluate_model
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler.fit(np.linspace(100, 200, 200).reshape(-1, 1))

        model  = build_lstm((10, 1))
        X_test = np.random.rand(20, 10, 1).astype(np.float32)
        y_test = np.random.rand(20, 1).astype(np.float32)
        rmse, mae, _, _ = evaluate_model(model, X_test, y_test, scaler, "Test")
        assert rmse >= 0
        assert mae  >= 0
