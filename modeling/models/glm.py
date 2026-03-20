import numpy as np
import pandas as pd
import statsmodels.api as sm


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

        # Try BFGS first (gradient-based, reliable for GLMs), then Nelder-Mead,
        # then fall back to Poisson if both NegBin attempts fail to converge.
        nb_model = sm.NegativeBinomial(y_arr, X_const)
        for method in ("bfgs", "nm"):
            try:
                result = nb_model.fit(disp=0, method=method, maxiter=2000)
                if result.mle_retvals.get("converged", True):
                    self._result = result
                    return self
            except Exception:
                continue

        # Last resort: Poisson GLM (always converges via IRLS)
        self._result = sm.GLM(
            y_arr, X_const, family=sm.families.Poisson()
        ).fit(disp=0)
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:

        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        preds = self._result.predict(X_const)
        return np.clip(preds, 0, None)

    def summary(self):
        return self._result.summary() if self._result else None
