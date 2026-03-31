from modeling.models.baseline import SeasonalNaive
from modeling.models.glm import NegBinGLM, PoissonGLM
from modeling.models.gbm import LGBMModel, XGBModel
from modeling.models.seasonal_baselines import LastWeekNaive, LastYearNaive
from modeling.models.xgb_sarimax import xgb_sarimax

MODEL_REGISTRY = {
    "modeling.models.baseline.SeasonalNaive": SeasonalNaive,
    "modeling.models.glm.NegBinGLM": NegBinGLM,
    "modeling.models.glm.PoissonGLM": PoissonGLM,
    "modeling.models.gbm.LGBMModel": LGBMModel,
    "modeling.models.gbm.XGBModel": XGBModel,
    "modeling.models.seasonal_baselines.LastWeekNaive": LastWeekNaive,
    "modeling.models.seasonal_baselines.LastYearNaive": LastYearNaive,
    "modeling.models.xgb_sarimax.xgb_sarimax": xgb_sarimax,
}


def build_model(cfg_model):
    """Instantiate a model from its config (requires a _target_ key)."""
    target = cfg_model._target_ if hasattr(cfg_model, "_target_") else cfg_model["_target_"]
    cls = MODEL_REGISTRY[target]
    if hasattr(cfg_model, "items"):
        kwargs = {k: v for k, v in cfg_model.items() if k != "_target_"}
    else:
        kwargs = {k: v for k, v in vars(cfg_model).items() if k != "_target_"}
    return cls(**kwargs)
