__all__ = ['build_model']

from .normflow_new import BiTrapSGlow
from .Glow import Glow

_MODELS_ = {
    'STGlow': BiTrapSGlow,
    'BiTrapSGlow': BiTrapSGlow,
}

def make_model(cfg):
    model = _MODELS_[cfg.METHOD]
    try:
        return model(cfg, dataset_name=cfg.DATASET.NAME)
    except:
        return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)
