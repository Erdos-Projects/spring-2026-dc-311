"""
LSTM model — stretch goal.

Requires PyTorch (pip install torch).  With only ~250 training rows this
model is unlikely to outperform gradient-boosted trees; include it only
when multi-year data is available.
"""

import numpy as np
import pandas as pd


class LSTMModel:
    """Sliding-window LSTM regressor (PyTorch backend)."""

    name = "lstm"

    def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                 epochs: int = 100, learning_rate: float = 1e-3,
                 dropout: float = 0.2, seq_len: int = 14, **kwargs):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seq_len = seq_len
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def _build_sequences(self, X: np.ndarray, y: np.ndarray | None = None):
        """Convert flat feature matrix into (N, seq_len, n_features) tensor."""
        import torch

        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len: i])
            if y is not None:
                targets.append(y[i])

        X_seq = torch.tensor(np.array(seqs), dtype=torch.float32)
        if y is not None:
            y_seq = torch.tensor(np.array(targets), dtype=torch.float32)
            return X_seq, y_seq
        return X_seq

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(
                "PyTorch and scikit-learn are required for LSTMModel."
            ) from e

        X_arr = X.values.astype(float)
        y_arr = y.values.astype(float)

        self._scaler_X = StandardScaler().fit(X_arr)
        X_scaled = self._scaler_X.transform(X_arr)

        X_seq, y_seq = self._build_sequences(X_scaled, y_arr)
        n_features = X_seq.shape[2]

        class _Net(nn.Module):
            def __init__(self, n_feat, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                                    dropout=drop if layers > 1 else 0.0)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self._model = _Net(n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            preds = self._model(X_seq)
            loss = loss_fn(preds, y_seq)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        X_arr = self._scaler_X.transform(X.values.astype(float))
        X_seq = self._build_sequences(X_arr)

        self._model.eval()
        with torch.no_grad():
            preds = self._model(X_seq).numpy()

        # Pad the first seq_len rows with the mean prediction
        pad = np.full(self.seq_len, preds.mean())
        return np.clip(np.concatenate([pad, preds]), 0, None)
