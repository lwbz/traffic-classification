# -*- coding: UTF-8 -*-
"""
@Project ：Network-zrb 
@File    ：encoder.py
@Author  ：Ronglin
@Date    ：2025/1/5 20:46 
"""


from typing import Tuple, Set

import torch
import torch.nn as nn
from torch import Tensor


from timm.layers import DropPath, trunc_normal_, lecun_normal_

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import math
import model.mamba.core as core
import model.mamba.config as config


class Encoder(nn.Module):
    """
    Mamba-based encoder that supports both standard forward pass and masked encoding.
    """

    def __init__(self, config: config.MambaConfig):
        """
        Initialize the encoder.

        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.d_model = self.embed_dim = self.config.encoder.embed_dim

        # Initialize components
        self._init_embedding_layer()
        self._init_cls_token()
        self._init_layers()
        self._init_mamba_blocks()
        self.drop_path = DropPath(
            self.config.encoder.drop_path_rate) if self.config.encoder.drop_path_rate > 0. else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_embedding_layer(self):
        """Initialize patch embedding layer."""
        self.patch_embed = core.StrideEmbed(
            self.config.data.seq_len,
            self.config.data.stride_size,
            self.config.data.in_chans,
            self.config.encoder.embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

    def _init_cls_token(self):
        """Initialize CLS token and related components."""
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))

    def _init_layers(self):
        """Initialize dropout and normalization layers."""
        self.pos_drop = nn.Dropout(p=self.config.encoder.drop_rate)
        self.norm_f = RMSNorm(self.embed_dim, eps=1e-5)

        # Global drop path
        self.drop_path = (DropPath(self.config.encoder.drop_path_rate)
                          if self.config.encoder.drop_path_rate > 0.
                          else nn.Identity())

    def _init_mamba_blocks(self):
        """Initialize Mamba blocks with progressive drop path rates."""
        # Generate drop path rates
        dpr = [x.item() for x in torch.linspace(0, self.config.encoder.drop_path_rate, self.config.encoder.depth)]
        inter_dpr = [0.0] + dpr  # Add 0 at the start for first block

        # Create blocks
        self.blocks = nn.ModuleList([
            core.create_mamba_block(
                self.embed_dim,
                layer_idx=i,
                drop_path=inter_dpr[i],
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True
            ) for i in range(self.config.encoder.depth)
        ])


    def _init_weights(self):
        """Initialize model weights following model-specific schemes"""

        def _init_weight_module(
                module: nn.Module,
                n_layer: int = self.config.encoder.depth,
                initializer_range: float = 0.02,
                rescale_prenorm_residual: bool = True,
                n_residuals_per_layer: int = 1
        ):
            """Initialize weights for a single module following GPT-2 scheme"""
            if isinstance(module, nn.Linear):
                if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=initializer_range)

            # Apply GPT-2 rescaling scheme for pre-norm residual paths
            if rescale_prenorm_residual:
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight", "fc2.weight"]:
                        # Special Scaled Initialization for Layer Norms
                        # Scale by 1/sqrt(2 * n_layer) following GPT-2 paper
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                        with torch.no_grad():
                            p /= math.sqrt(n_residuals_per_layer * n_layer)

        # Apply initialization to all modules
        self.apply(lambda m: _init_weight_module(m))
        # Special initialization for specific components
        self.patch_embed.apply(core.init_segm_weights)

        # Initialize positional embeddings and tokens
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)


    def _process_embeddings(self, x: Tensor) -> Tensor:
        """
        Process input through embedding layers.

        Args:
            x: Input tensor

        Returns:
            Processed tensor with embeddings
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :-1, :]
        return x

    def _append_cls_token(self, x: Tensor) -> Tensor:
        """
        Append CLS token to the input sequence.

        Args:
            x: Input tensor

        Returns:
            Tensor with CLS token appended
        """
        cls_token = self.cls_token + self.pos_embed[:, -1:, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((x, cls_tokens), dim=1)

    def _apply_mamba_blocks(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply Mamba blocks to input sequence.

        Args:
            x: Input tensor

        Returns:
            Tuple of (hidden_states, residual)
        """
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)
        return hidden_states, residual

    def _apply_final_norm(self, hidden_states: Tensor, residual: Tensor) -> Tensor:
        """
        Apply final normalization to the output.

        Args:
            hidden_states: Hidden states from Mamba blocks
            residual: Residual connection

        Returns:
            Normalized tensor
        """
        return rms_norm_fn(
            self.drop_path(hidden_states),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )

    @staticmethod
    def random_masking(x: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply random masking to the input sequence.

        Args:
            x: Input tensor of shape (B, N, D)
            mask_ratio: Ratio of tokens to mask

        Returns:
            Tuple of (masked_tensor, mask, restore_indices)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def _forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the feature extraction layers.

        Args:
            x: Input tensor

        Returns:
            Tuple of (hidden_states, residual)
        """
        x = self._append_cls_token(x)
        x = self.pos_drop(x)
        return self._apply_mamba_blocks(x)

    def forward(self, x: Tensor):
        """
        Standard forward pass without masking.

        Args:
            x: Input tensor

        Returns:
            If not pretraining:
                - Encoded tensor
            If pretraining:
                - Tuple of (encoded_tensor, mask, restore)
        """
        if self.config.is_pretrain:
            return self.forward_pre(x, mask_ratio=self.config.pretrain.mask_ratio)
        else:
            x = self._process_embeddings(x)
            hidden_states, residual = self._forward_features(x)
            return self._apply_final_norm(hidden_states, residual)

    def forward_pre(self, x: Tensor, mask_ratio: float = 0.75) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with masking for pretraining.

        Args:
            x: Input tensor
            mask_ratio: Ratio of tokens to mask

        Returns:
            Tuple of (encoded_tensor, mask, restore_indices)
        """
        x = self._process_embeddings(x)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        hidden_states, residual = self._forward_features(x)
        x = self._apply_final_norm(hidden_states, residual)
        return x, mask, ids_restore


    @torch.jit.ignore
    def no_weight_decay(self) -> Set[str]:
        """Parameters that should not use weight decay"""
        return {"pos_embed", "cls_token"}

