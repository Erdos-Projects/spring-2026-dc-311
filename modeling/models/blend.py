"""Post-hoc weighted blend model."""

from __future__ import annotations

import numpy as np


class WeightedBlendModel:
    """Blend fitted component models with fixed validation-selected weights."""

    def __init__(self, components: dict[str, object], weights: dict[str, float], name: str):
        self.components = components
        self.weights = {key: float(value) for key, value in weights.items()}
        self.name = name
        self.device = "cpu"

    def predict(self, X, **kwargs):
        blended = np.zeros(len(X), dtype=float)
        for name, model in self.components.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0:
                continue
            preds = np.asarray(model.predict(X, **kwargs), dtype=float)
            blended += weight * np.clip(preds, 0, None)
        return np.clip(blended, 0, None)
