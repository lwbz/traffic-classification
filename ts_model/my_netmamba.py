import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import math
from torch import Tensor
from typing import Optional

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class LinearEmbed(nn.Module):
    """适配时序数据的嵌入层"""

    def __init__(self, seq_len=50, input_dim=1, embed_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, L) -> (B, L, 1)
        return self.proj(x)  # (B, L, D)


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm,
            fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls()
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)

        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        print(f"Before Mamba: hidden_states shape={hidden_states.shape}")  # 调试
        hidden_states = self.mixer(hidden_states)
        print(f"After Mamba: hidden_states shape={hidden_states.shape}")  # 调试
        return hidden_states, residual


def create_block(
        d_model,
        ssm_cfg=None,
        drop_path=0.,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs
):
    from mamba_ssm.modules.mamba_simple import Mamba

    # 确保 ssm_cfg 覆盖所有 Mamba 参数
    if ssm_cfg is None:
        ssm_cfg = {
            "d_model": d_model,
            "expand": 1,
            "bimamba_type": "none",
            "d_models": d_model,
            "feat_dim": d_model,
            "max_len": 51,
            "window_size": 51,
            "dim_output": d_model,
            "d_state": 16,
            "d_conv": 4,
            "dt_rank": "auto",
        }
    else:
        ssm_cfg = {
            "d_model": d_model,
            "expand": 1,
            "bimamba_type": "none",
            "d_models": d_model,
            "feat_dim": d_model,
            "max_len": 51,
            "window_size": 51,
            "dim_output": d_model,
            "d_state": 16,
            "d_conv": 4,
            "dt_rank": "auto",
            **ssm_cfg
        }

    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    # 调试：验证 Mamba 初始化
    mixer = mixer_cls()
    print(f"Mamba in_proj.weight shape: {mixer.in_proj.weight.shape}")
    print(f"Mamba d_inner: {mixer.d_inner}, d_model: {mixer.d_model}")

    norm_cls = partial(RMSNorm, eps=1e-5, **factory_kwargs)

    return Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=True,
        residual_in_fp32=True,
    )


class NetMambaFeatureExtractor(nn.Module):
    def __init__(
            self,
            seq_len=50,
            input_dim=1,
            embed_dim=256,
            depth=4,
            drop_rate=0.,
            drop_path_rate=0.1,
            device=None,
            **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # 1. 输入嵌入
        self.patch_embed = LinearEmbed(seq_len, input_dim, embed_dim)

        # 2. 位置编码和CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Mamba块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            create_block(
                embed_dim,
                drop_path=dpr[i],
                layer_idx=i,
                device=device,
            ) for i in range(depth)
        ])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)

        # 初始化权重
        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, return_type="cls"):
        """
        参数:
            x: 输入张量 (B, seq_len)
            return_type:
                "cls" - 返回CLS Token特征 [B, D]
                "all" - 返回所有特征 [B, L+1, D]
                "mean" - 返回时间步平均 [B, D]
        """
        # 1. 嵌入层
        x = self.patch_embed(x)  # (B, L, D)
        print(f"After patch_embed: {x.shape}")  # 调试

        # 2. 添加位置编码
        x = x + self.pos_embed[:, :-1, :]
        print(f"After pos_embed: {x.shape}")  # 调试

        # 3. 添加CLS Token
        cls_token = self.cls_token + self.pos_embed[:, -1:, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)  # (B, L+1, D)
        x = self.pos_drop(x)
        print(f"After cls_token: {x.shape}")  # 调试

        # 4. 通过Mamba块
        residual = None
        for blk in self.blocks:
            x, residual = blk(x, residual)
        x = self.norm_f(x)
        print(f"After blocks: {x.shape}")  # 调试

        # 5. 返回指定特征
        if return_type == "all":
            return x
        elif return_type == "cls":
            return x[:, 0, :]
        elif return_type == "mean":
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")


def create_feature_extractor(
        seq_len=50,
        input_dim=1,
        embed_dim=256,
        depth=4,
        **kwargs
):
    """创建特征提取器实例"""
    return NetMambaFeatureExtractor(
        seq_len=seq_len,
        input_dim=input_dim,
        embed_dim=embed_dim,
        depth=depth,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_feature_extractor(seq_len=50).to(device)

    # 模拟输入数据 (128, 50)
    x = torch.randn(128, 50).to(device)

    # 提取特征
    print("CLS Token特征形状:", model(x).shape)  # 应输出 torch.Size([128, 256])
    print("全部特征形状:", model(x, return_type="all").shape)  # 应输出 torch.Size([128, 51, 256])