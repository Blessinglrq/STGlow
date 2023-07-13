from bitrap.engine.trainer import do_train_flow
from bitrap.engine.trainer import do_val_flow
from bitrap.engine.trainer import inference_flow

ENGINE_ZOO = {
                'STGlow': (do_train_flow, do_val_flow, inference_flow),
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]
