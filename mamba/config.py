# -*- coding: UTF-8 -*-
"""
@Project ：Network-zrb 
@File    ：config.py
@Author  ：Ronglin
@Date    ：2025/1/9 14:33 
"""

"""
Configuration classes for Mamba model architecture.
Implements hierarchical configuration with clear separation of concerns.
"""


from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Base configuration for model-wide settings"""
    # device: Optional[torch.device] = None   # 这个暂时不支持
    # dtype: Optional[torch.dtype] = None     # 这个暂时不支持
    is_pretrain: bool = True


@dataclass
class DataConfig:
    """Configuration for input data characteristics"""
    seq_len: int = 1085
    in_chans: int = 1
    num_classes: int = 14
    stride_size: int = 5


@dataclass
class EncoderConfig:
    """Configuration specific to encoder architecture"""
    embed_dim: int = 256
    depth: int = 4
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1

@dataclass
class DecoderConfig:
    """Configuration specific to decoder architecture"""
    embed_dim: int = 128  # decoder_embed_dim in original
    depth: int = 2  # decoder_depth in original
    norm_pix_loss: bool = False


@dataclass
class PretrainingConfig:
    """Configuration for pretraining settings"""
    mask_ratio: float = 0.8
    use_normalized_loss: bool = False  # renamed from norm_pix_loss for clarity


@dataclass
class MambaConfig:
    """
    Main configuration class that aggregates all sub-configurations.
    Provides convenient access to all settings while maintaining separation of concerns.
    """

    def __init__(
            self,
            model_config: Optional[ModelConfig] = None,
            data_config: Optional[DataConfig] = None,
            encoder_config: Optional[EncoderConfig] = None,
            decoder_config: Optional[DecoderConfig] = None,
            pretrain_config: Optional[PretrainingConfig] = None
    ):
        self.model = model_config or ModelConfig()
        self.data = data_config or DataConfig()
        self.encoder = encoder_config or EncoderConfig()
        self.decoder = decoder_config or DecoderConfig()
        self.pretrain = pretrain_config or PretrainingConfig()

    # @property
    # def device(self) -> Optional[torch.device]:
    #     return self.model.device
    #
    # @property
    # def dtype(self) -> Optional[torch.dtype]:
    #     return self.model.dtype

    @property
    def is_pretrain(self) -> bool:
        return self.model.is_pretrain

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MambaConfig':
        """
        Create a MambaConfig instance from a dictionary of parameters.
        Useful for loading config from JSON/YAML files.
        """
        model_params = {
            'device': config_dict.get('device'),
            'dtype': config_dict.get('dtype'),
            'is_pretrain': config_dict.get('is_pretrain', True)
        }

        data_params = {
            'seq_len': config_dict.get('seq_len', 1085),
            'in_chans': config_dict.get('in_chans', 1),
            'num_classes': config_dict.get('num_classes', 14),
            'stride_size': config_dict.get('stride_size', 5)
        }

        encoder_params = {
            'embed_dim': config_dict.get('embed_dim', 256),
            'depth': config_dict.get('depth', 4),
            'drop_rate': config_dict.get('drop_rate', 0.0),
            'drop_path_rate': config_dict.get('drop_path_rate', 0.1)
        }

        decoder_params = {
            'embed_dim': config_dict.get('decoder_embed_dim', 128),
            'depth': config_dict.get('decoder_depth', 2),
            'norm_pix_loss': config_dict.get('norm_pix_loss', False)
        }

        pretrain_params = {
            'mask_ratio': config_dict.get('mask_ratio', 0.8),
            'use_normalized_loss': config_dict.get('norm_pix_loss', False)
        }

        return cls(
            model_config=ModelConfig(**model_params),
            data_config=DataConfig(**data_params),
            encoder_config=EncoderConfig(**encoder_params),
            decoder_config=DecoderConfig(**decoder_params),
            pretrain_config=PretrainingConfig(**pretrain_params)
        )

    def to_dict(self) -> dict:
        """
        Convert configuration to a flat dictionary.
        Useful for serialization and backwards compatibility.
        """
        return {
            # Model settings
            # 'device': self.model.device,
            # 'dtype': self.model.dtype,
            'is_pretrain': self.model.is_pretrain,

            # Data settings
            'seq_len': self.data.seq_len,
            'in_chans': self.data.in_chans,
            'num_classes': self.data.num_classes,
            'stride_size': self.data.stride_size,

            # Encoder settings
            'embed_dim': self.encoder.embed_dim,
            'depth': self.encoder.depth,
            'drop_rate': self.encoder.drop_rate,
            'drop_path_rate': self.encoder.drop_path_rate,

            # Decoder settings
            'decoder_embed_dim': self.decoder.embed_dim,
            'decoder_depth': self.decoder.depth,
            'norm_pix_loss': self.decoder.norm_pix_loss,

            # Pretraining settings
            'mask_ratio': self.pretrain.mask_ratio,
        }

    def __str__(self):
        return str(self.to_dict())