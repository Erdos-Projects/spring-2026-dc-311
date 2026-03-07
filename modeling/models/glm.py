import numpy as np
import pandas as pd


class NegBinGLM:
    """
    Negative Binomial GLM for overdispersed count data.

    Wraps statsmodels NegativeBinomial with a sklearn-style interface.
    Falls back to a Poisson GLM if fitting fails (e.g. convergence issues
    with small datasets).
    """

    name = "negbin_glm"

    def __init__(self, alpha: float = 1.0, **kwargs):
        self.alpha = alpha
        self._result = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NegBinGLM":
        import statsmodels.api as sm

        self._feature_names = list(X.columns)
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        y_arr = y.astype(float).values

        try:
            model = sm.NegativeBinomial(y_arr, X_const)
            self._result = model.fit(disp=0, method="nm", maxiter=2000)
        except Exception:
            # Fallback: Poisson GLM
            model = sm.GLM(y_arr, X_const, family=sm.families.Poisson())
            self._result = model.fit()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import statsmodels.api as sm

        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        preds = self._result.predict(X_const)
        return np.clip(preds, 0, None)

    def summary(self):
        return self._result.summary() if self._result else None
