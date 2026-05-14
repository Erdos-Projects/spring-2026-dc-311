"""Post-hoc calibrated model wrappers."""

import numpy as np

from modeling.models.utils import normalize_predict_kwargs


class MultiplicativeCalibratedModel:
    """Apply a fixed multiplicative factor to a fitted model's predictions."""

    def __init__(self, base_model, calibration_factor: float, name: str):
        self.base_model = base_model
        self.calibration_factor = float(calibration_factor)
        self.name = name
        self.device = getattr(base_model, "device", "cpu")

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
        preds = self.base_model.predict(
            X,
            recursive=recursive,
            horizon_h=horizon_h,
            assimilate=assimilate,
            Ys=Ys,
            **kwargs,
        )
        return np.clip(np.asarray(preds, dtype=float) * self.calibration_factor, 0, None)

    @property
    def feature_importances_(self):
        return getattr(self.base_model, "feature_importances_")
