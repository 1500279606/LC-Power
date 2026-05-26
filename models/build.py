import torch

from models.causal.modeling_causal import LC-Power
from models.timesnet.modeling_timesnet import TimesNet


def build_model(cfg):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "LC-Power": LC-Power,
        "Causal": LC-Power,  # 兼容 Causal 名称
        "TIMESNET": TimesNet,
    }

    if model_name in model_mapping:
        model = model_mapping[model_name](cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if torch.cuda.is_available():
        model = model.cuda()

    return model
