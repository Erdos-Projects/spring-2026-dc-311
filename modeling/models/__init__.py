from modeling.models.baseline import SeasonalNaive
from modeling.models.glm import NegBinGLM
from modeling.models.gbm import LGBMModel, XGBModel

MODEL_REGISTRY = {
    "modeling.models.baseline.SeasonalNaive": SeasonalNaive,
    "modeling.models.glm.NegBinGLM": NegBinGLM,
    "modeling.models.gbm.LGBMModel": LGBMModel,
    "modeling.models.gbm.XGBModel": XGBModel,
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
