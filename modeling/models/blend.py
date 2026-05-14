"""Post-hoc weighted blend model."""

from __future__ import annotations

import numpy as np

from modeling.models.utils import normalize_predict_kwargs


class WeightedBlendModel:
    """Blend fitted component models with fixed validation-selected weights."""

    def __init__(self, components: dict[str, object], weights: dict[str, float], name: str):
        self.components = components
        self.weights = {key: float(value) for key, value in weights.items()}
        self.name = name
        self.device = "cpu"

    def predict(
        self,
        X,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        assimilate: bool = False,
        Ys=None,
        **kwargs,
    ):
        horizon_h, Ys, kwargs = normalize_predict_kwargs(
            horizon_h=horizon_h,
            Ys=Ys,
            kwargs=kwargs,
        )
        blended = np.zeros(len(X), dtype=float)
        for name, model in self.components.items():
            weight = self.weights.get(name, 0.0)
            if weight == 0:
                continue
            preds = np.asarray(
                model.predict(
                    X,
                    recursive=recursive,
                    horizon_h=horizon_h,
                    assimilate=assimilate,
                    Ys=Ys,
                    **kwargs,
                ),
                dtype=float,
            )
            blended += weight * np.clip(preds, 0, None)
        return np.clip(blended, 0, None)
