# -*- coding: UTF-8 -*-
"""
@Project ：Network-zrb 
@File    ：decoder.py
@Author  ：Ronglin
@Date    ：2025/1/9 14:20 
"""
from typing import Optional, Set

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
from torch import Tensor


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import model.mamba.core as core
import model.mamba.config as config


class Decoder(nn.Module):
    """
    Decoder module that reconstructs input sequences using masked token prediction.

    Attributes:
        config (MambaConfig): Model configuration parameters
        decoder_embed (nn.Linear): Input embedding layer
        mask_token (nn.Parameter): Learnable mask token
        decoder_pos_embed (nn.Parameter): Positional embeddings
        decoder_blocks (nn.ModuleList): List of decoder transformer blocks
        decoder_norm_f (RMSNorm): Final layer normalization
        decoder_pred (nn.Linear): Output prediction layer
    """

    def __init__(self, config: config.MambaConfig):
        """
        Initialize decoder components.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config

        # Initialize embedding layers
        self._init_embeddings()

        # Initialize decoder blocks
        self._init_mamba_blocks()
        self.drop_path = DropPath(self.config.encoder.drop_path_rate) if self.config.encoder.drop_path_rate > 0. else nn.Identity()

        # Initialize normalization and prediction layers
        self.decoder_norm_f = RMSNorm(
            self.config.decoder.embed_dim,
            eps=1e-5
        )
        self.decoder_pred = nn.Linear(
            self.config.decoder.embed_dim,
            self.config.data.stride_size * self.config.data.in_chans,
            bias=True
        )

        # Initialize weights
        self._init_weights()

    def _init_embeddings(self) -> None:
        """Initialize embedding layers and mask token."""
        self.decoder_embed = nn.Linear(
            self.config.encoder.embed_dim,
            self.config.decoder.embed_dim,
            bias=True
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.config.decoder.embed_dim)
        )

        self.num_patches = self.config.data.seq_len // self.config.data.stride_size

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.config.decoder.embed_dim)
        )

    def _init_mamba_blocks(self)-> None :
        """Initialize Mamba blocks with progressive drop path rates."""

        # Calculate drop path rates
        decoder_dpr = torch.linspace(
            0,
            self.config.encoder.drop_path_rate,
            self.config.decoder.depth
        ).tolist()
        decoder_inter_dpr = [0.0] + decoder_dpr

        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList([
            core.create_mamba_block(
                self.config.decoder.embed_dim,
                layer_idx=i,
                drop_path=decoder_inter_dpr[i],
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
            ) for i in range(self.config.decoder.depth)
        ])

    def _init_weights(self):
        """Initialize model weights following model-specific schemes"""
        trunc_normal_(self.decoder_pos_embed, std=.02)
        trunc_normal_(self.mask_token, std=.02)


    def _process_sequence(
            self,
            x: Tensor,
            ids_restore: Tensor
    ) -> Tensor:
        """
        Process input sequence by adding mask tokens and restoring order.

        Args:
            x: Input tensor
            ids_restore: Indices for restoring original sequence order

        Returns:
            Processed sequence tensor
        """
        # Create and append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1
        )

        # Combine visible and mask tokens
        x_ = torch.cat([x[:, :-1, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )

        # Append CLS token and add positional embedding
        x = torch.cat([x_, x[:, -1:, :]], dim=1)
        return x + self.decoder_pos_embed

    def forward(
            self,
            x: Tensor,
            ids_restore: Tensor
    ) -> Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Input tensor from encoder
            ids_restore: Indices for restoring original sequence order

        Returns:
            Decoded output tensor (excluding CLS token)
        """
        # Embed input
        x = self.decoder_embed(x)

        # Process sequence
        x = self._process_sequence(x, ids_restore)

        # Apply decoder blocks
        residual: Optional[Tensor] = None
        hidden_states = x

        for block in self.decoder_blocks:
            hidden_states, residual = block(hidden_states, residual)

        # Apply final normalization
        x = rms_norm_fn(
            self.drop_path(hidden_states),
            self.decoder_norm_f.weight,
            self.decoder_norm_f.bias,
            eps=self.decoder_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

        # Final prediction
        x = self.decoder_pred(x)

        # Remove CLS token
        return x[:, :-1, :]

    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Parameters that should not use weight decay"""
        return {"decoder_pos_embed"}


class Head(nn.Module):
    def __init__(self, config: config.MambaConfig):
        super().__init__()
        self.config = config
        self.head = (nn.Linear(self.config.encoder.embed_dim, self.config.data.num_classes)
                     if self.config.data.num_classes > 0 else nn.Identity())

        self.head.apply(core.init_segm_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x[:, -1, :])