import torch
import torch.nn as nn

from models.causal.encoder import iTransformer_ENC, TimesNet_ENC, LSTM_ENC, GRU_ENC, GATV2_ENC, Sparse_ENC


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class LC-Power(nn.Module):
    def __init__(self, cfg):
        super(LC-Power, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg.DATA
        self.cfg_causal = cfg.Causal

        self.device = get_device()

        self.encoder = self._init_encoder()
        self.projector = self._init_projector()
        self.causal_discoverer = self._init_causal_discoverer()
        self.positive_augmentor = self._init_positive_augmentor()
        self.negative_augmentor = self._init_negative_augmentor()

    def _init_encoder(self):
        arch = self.cfg_causal.ENCODER_ARCH
        encoder_classes = {
            'lstm': LSTM_ENC,
            'iTransformer': iTransformer_ENC,
            'GATv2': GATV2_ENC,
            'TimesNet': TimesNet_ENC,
            'gru': GRU_ENC,
            'sparse': Sparse_ENC,
        }

        if arch not in encoder_classes:
            raise ValueError(f"Unsupported encoder architecture: {arch}")

        encoder = encoder_classes[arch](self.cfg)
        return encoder

    def _init_projector(self):
        cfg_projector = self.cfg_causal.PROJECTOR
        projector = nn.Sequential(
            nn.Linear(cfg_projector.INPUT_DIM, cfg_projector.HIDDEN_DIM),
            nn.BatchNorm1d(cfg_projector.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg_projector.HIDDEN_DIM, cfg_projector.OUTPUT_DIM),
        )

        return projector

    def _init_causal_discoverer(self):
        from models.causal.modeling_cuts_plus import Causal_Net
        self.cfg_cuts_plus = self.cfg.CUTS_PLUS
        causal_discoverer = Causal_Net(self.cfg).to(self.device)

        return causal_discoverer

    def _init_positive_augmentor(self):
        from models.causal.modeling_positive_augmentor import PositiveAugmentor
        positive_augmentor = PositiveAugmentor(self.cfg).to(self.device)

        return positive_augmentor

    def _init_negative_augmentor(self):
        from models.causal.modeling_negative_augmentor import NegativeAugmentor
        negative_augmentor = NegativeAugmentor(self.cfg).to(self.device)

        return negative_augmentor

    def forward(self, x, positive_augment=True, negative_augment=True):
        x_all = x

        # Apply positive augmentation if enabled
        if positive_augment:
            positive_samples = self.positive_augmentor(x_all, self.causal_discoverer.causality_mtx)
            x_all = torch.concat([x_all, positive_samples], dim=0)

        # Apply negative augmentation if enabled
        if negative_augment:
            negative_samples = self.negative_augmentor(x_all, self.causal_discoverer.causality_mtx)
            x_all = torch.concat([x_all, negative_samples], dim=0)

        # Encode the input based on the specified architecture
        if self.cfg_causal.ENCODER_ARCH == 'GATv2':
            enc_out = self.encoder(x_all, self.causal_discoverer.causality_mtx)
        elif self.cfg_causal.ENCODER_ARCH in ('lstm', 'iTransformer', 'TimesNet', 'gru', 'sparse'):
            enc_out = self.encoder(x_all)
        else:
            raise ValueError(f"Unsupported encoder architecture: {self.cfg_causal.ENCODER_ARCH}")

        # Project the encoded output
        out = self.projector(enc_out)

        return out
