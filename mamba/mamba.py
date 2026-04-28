# -*- coding: UTF-8 -*-
"""
@Project ：Network-zrb 
@File    ：mamba.py
@Author  ：Ronglin
@Date    ：2025/1/9 15:09 
"""
import torch
import torch.nn as nn
from torch import Tensor

from model.mamba.encoder import Encoder
from model.mamba.decoder import Decoder, Head
from model.mamba.config import MambaConfig



class MambaMain(nn.Module):
    """
    Mamba model architecture for sequence classification.
    """

    def __init__(self, config: MambaConfig, *args, **kwargs):
        """
        Initialize the Mamba model.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(*args, **kwargs)

        self.config = config
        self.encoder = Encoder(config)
        self.head = Head(config)

    def forward(self, x, feature=False):
        """
        Forward pass for the Mamba model.

        Args:
            x: Input tensor of shape (B, C, L)
            feature : If True, return features before classification head

        Returns:
            Tensor of shape (B, num_classes)
            Classification logits  or (B, num_classes), features if feature=True
        """

        B, C, L = x.shape
        assert C == 1, "Input sequences should be single channel"
        assert not self.config.is_pretrain, "Model is configured for pretraining"

        x_feature = self.encoder(x)
        x = self.head(x_feature)

        if feature:
            return x, x_feature
        else:
            return x

    def load_encoder(self, path: str):
        """
        Loading encoder parameters from pre trained models

        Args:
            path: Path to the pre-trained model

        Returns:
            load loaded_params length
        """
        pretrained_dict = torch.load(path, weights_only=True)
        encoder_dict = self.encoder.state_dict()

        # 过滤掉不匹配的参数
        pretrained_need_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}

        # 更新当前模型的参数
        encoder_dict.update(pretrained_need_dict)
        self.encoder.load_state_dict(encoder_dict)

        # 记录加载情况
        loaded_params = set(pretrained_need_dict.keys())

        return loaded_params


class MambaPretrain(nn.Module):
    """
    Mamba model architecture for sequence pretraining.
    """

    def __init__(self, config: MambaConfig, *args, **kwargs):
        """
        Initialize the Mamba model.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(*args, **kwargs)

        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def _patchify_sequence(self, seqs: Tensor) -> Tensor:
        """Convert input sequence to patches

        Args:
            seqs: Input sequence tensor [B, C, L]

        Returns:
            Patchified sequence tensor
        """
        B, C, L = seqs.shape
        assert C == 1, "Input sequences should be single channel"

        return seqs.reshape(B, L // self.config.data.stride_size, self.config.data.stride_size)

    def forward_loss(self, seqs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """Calculate reconstruction loss

        Args:
            seqs: Original input sequence
            pred: Model predictions
            mask: Masking tensor (0=keep, 1=remove)

        Returns:
            Reconstruction loss value
        """
        target = self._patchify_sequence(seqs)

        # 添加数值检查
        # print("Target range:", torch.min(target).item(), torch.max(target).item())
        # print("Pred range:", torch.min(pred).item(), torch.max(pred).item())

        if self.config.pretrain.use_normalized_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        # print("MSE range:", torch.min(loss).item(), torch.max(loss).item())

        loss = loss.mean(dim=-1)  # mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        # Check if the final loss retains the gradient
        assert loss.requires_grad, "Loss does not have gradients"

        return loss

    def forward(self, x):
        """
        Forward pass for the Mamba model.

        Args:
            x: Input tensor of shape (B, C, L)

        Returns:
            Reconstruction loss
            Predictions
            Mask tensor
        """
        B, C, L = x.shape
        assert C == 1, "Input sequences should be single channel"
        assert self.config.is_pretrain, "Model is not configured for pretraining"

        latent, mask, ids_restore = self.encoder(x)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask


    def save_encoder(self, path: str):
        """
        Save the encoder parameters to a file.

        Args:
            path: Path to save the encoder parameters
        """
        torch.save(self.encoder.state_dict(), path)

