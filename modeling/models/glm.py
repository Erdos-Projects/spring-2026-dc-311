import numpy as np
import pandas as pd
import statsmodels.api as sm

from modeling.models.utils import predict_in_blocks


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

    def predict(
        self,
        X: pd.DataFrame,
        *,
        recursive: bool = False,
        horizon_h: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")

        ar_cols = [c for c in X.columns if c.startswith("pothole_lag")]
        if not recursive or len(ar_cols) == 0:
            X_const = sm.add_constant(X.astype(float), has_constant="add")
            preds = self._result.predict(X_const)
            return np.clip(preds, 0, None)

        if horizon_h is not None:
            return predict_in_blocks(self, X, horizon_h)

        k_AR = max(int(c.replace("pothole_lag", "")) for c in ar_cols)
        X_work = X.copy().astype(float)
        preds = []
        print(f"Using recursive prediction with k_AR = {k_AR}")
        for i in range(len(X)):
            if i > 0:
                for k in range(1, min(i, k_AR) + 1):
                    col = f"pothole_lag{k}"
                    if col in X_work.columns:
                        X_work.iloc[i, X_work.columns.get_loc(col)] = preds[i - k]
            X_const = sm.add_constant(X_work.iloc[[i]], has_constant="add")
            pred_i = self._result.predict(X_const).item()
            preds.append(max(0.0, pred_i))
        return np.array(preds)

    def summary(self):
        return self._result.summary() if self._result else None

class PoissonGLM(NegBinGLM):
    """
    Poisson GLM for count data.
    """

    name = "poisson_glm"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NegBinGLM":
        import statsmodels.api as sm

        self._feature_names = list(X.columns)
        X_const = sm.add_constant(X.astype(float), has_constant="add")
        y_arr = y.astype(float).values
        self._result = sm.GLM(
            y_arr, X_const, family=sm.families.Poisson()
        ).fit(disp=0)
        return self