import torch
import torch.nn as nn
from mamba_ssm import Mamba


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerBranch(nn.Module):
    def __init__(self, in_channels, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super(TransformerBranch, self).__init__()
        self.embedding = nn.Linear(in_channels, d_model)  # 输入投影
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, seq_len, in_channels)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # 全局平均池化 (batch_size, d_model)
        return x


class ParallelTransformerModel(nn.Module):
    def __init__(self, num_classes=18):
        super(ParallelTransformerModel, self).__init__()

        # 时域分支
        self.time_branch = TransformerBranch(
            in_channels=1,  # 时域输入特征维度
            d_model=64,  # Transformer 特征维度
            nhead=4,  # 多头注意力头数
            num_layers=2,  # Transformer 层数
            dim_feedforward=256
        )

        # 频域分支
        self.freq_branch = TransformerBranch(
            in_channels=2,  # 频域输入特征维度
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256
        )

        # 融合与分类
        self.fc = nn.Linear(64 + 64, num_classes)  # 64 (time) + 64 (freq)
        self.dropout = nn.Dropout(0.2)

    def forward(self, time_data, freq_data):
        time_data = time_data.permute(0, 2, 1)  # (n_samples, 50, 1)
        freq_data = freq_data.permute(0, 2, 1)  # (n_samples, 50, 2)

        # 时域分支
        time_out = self.time_branch(time_data)  # (batch_size, 64)

        # 频域分支
        freq_out = self.freq_branch(freq_data)  # (batch_size, 64)

        # 融合
        combined = torch.cat([time_out, freq_out], dim=1)  # (batch_size, 128)

        combined = self.dropout(combined)

        # 分类
        output = self.fc(combined)  # (batch_size, num_classes)
        return output

# 这段代码定义了一个自定义的 PyTorch 模块 SqueezeChannels，它的功能是压缩输入张量的第 2 维（索引为 2 的维度）。
class SqueezeChannels(nn.Module):
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # (batch, channels, length)
        y = self.avg_pool(x).view(b, c)  # (batch, channels)
        y = self.fc(y).view(b, c, 1)     # (batch, channels, 1)
        return x * y  # 通道加权


class FCN(nn.Module):
    def __init__(self, num_classes, input_size=1):
        super(FCN, self).__init__()

        self.num_classes = num_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            nn.AdaptiveAvgPool1d(1),
            SqueezeChannels(),
        )

    def forward(self, x, vis=False):
        if x.shape[1] == 50:
            x = x.permute(0, 2, 1)  # (128, 50, 1) -> (128, 1, 50)
        if vis:
            with torch.no_grad():
                vis_out = self.conv_block1(x)
                vis_out = self.conv_block2(vis_out)
                vis_out = self.conv_block3(vis_out)
                return self.network(x), vis_out

        return self.network(x)


class Classifier(nn.Module):
    # input_dims: 输入特征的维度（即每个样本的特征数）。
    # output_dims: 输出类别的数量。
    def __init__(self, input_dims, output_dims) -> None:
        super(Classifier, self).__init__()

        self.dense = nn.Linear(input_dims, output_dims)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.dense(x))


# 这段代码实现了一个 Projection Head 模块，用于将输入特征映射到嵌入空间（通常用于对比学习任务中的特征投影）。
# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, embedding_dim=64, output_dim=32) -> None:
#         super(ProjectionHead, self).__init__()
#         self.projection_head = nn.Sequential(
#             nn.Linear(input_dim, embedding_dim),
#             nn.BatchNorm1d(embedding_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embedding_dim, output_dim),
#         )
#
#     def forward(self, x):
#         return self.projection_head(x)

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, embedding_dim1=128, embedding_dim2=64, output_dim=32) -> None:
#         super(ProjectionHead, self).__init__()
#         self.projection_head = nn.Sequential(
#             nn.Linear(input_dim, embedding_dim1),
#             nn.BatchNorm1d(embedding_dim1),
#             nn.ReLU(inplace=True),
#             nn.Linear(embedding_dim1, embedding_dim2),
#             nn.BatchNorm1d(embedding_dim2),
#             nn.ReLU(inplace=True),
#             nn.Linear(embedding_dim2, output_dim),
#         )
#
#     def forward(self, x):
#         return self.projection_head(x)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=32) -> None:
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.projection_head(x)


# class ParallelModel(nn.Module):
#     def __init__(self, num_classes=21):
#         super(ParallelModel, self).__init__()
#
#         # 时域分支
#         self.time_branch = nn.Sequential(
#             nn.Conv1d(1, 128, kernel_size=8, padding=3),  # in_channels=1
#             nn.ReLU(),
#             nn.Conv1d(128, 256, kernel_size=8, padding=3),
#             nn.ReLU(),
#             nn.Conv1d(256, 128, kernel_size=8, padding=3),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)  # 池化到 (batch, 128, 1)
#         )
#
#         # 频域分支
#         self.freq_branch = nn.Sequential(
#             nn.Conv1d(2, 64, kernel_size=8, padding=3),  # in_channels=2
#             nn.ReLU(),
#             nn.Conv1d(64, 128, kernel_size=8, padding=3),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, kernel_size=8, padding=3),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)  # 池化到 (batch, 128, 1)
#         )
#
#         # 融合后的全连接层
#         self.fc = nn.Linear(128 + 128, num_classes)  # 128 (time) + 128 (freq)
#
#     def forward(self, time_data, freq_data):
#         # 时域分支
#         time_out = self.time_branch(time_data)  # (batch, 128, 1)
#         time_out = time_out.squeeze(-1)  # (batch, 128)
#
#         # 频域分支
#         freq_out = self.freq_branch(freq_data)  # (batch, 128, 1)
#         freq_out = freq_out.squeeze(-1)  # (batch, 128)
#
#         # 融合
#         combined = torch.cat([time_out, freq_out], dim=1)  # (batch, 256)
#
#         # 分类
#         output = self.fc(combined)  # (batch, num_classes)
#         return output, combined
#
#     def get_embedding(self, time_data, freq_data):
#         # 直接调用 forward，避免重复计算
#         _, embedding = self.forward(time_data, freq_data)
#         return embedding


class ParallelModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ParallelModel, self).__init__()

        # 时域分支
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=8, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=8, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.time_se = SEBlock(128)  # SE 块

        # 频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.freq_se = SEBlock(128)  # SE 块

        # 池化和分类
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 + 128, num_classes)  # 拼接后 256

    def forward(self, time_data, freq_data):
        # 时域分支
        time_out = self.time_branch(time_data)  # (batch, 128, 50)
        time_out = self.time_se(time_out)  # SE 加权
        time_out = self.pool(time_out).squeeze(-1)  # (batch, 128)

        # 频域分支
        freq_out = self.freq_branch(freq_data)  # (batch, 128, 50)
        freq_out = self.freq_se(freq_out)  # SE 加权
        freq_out = self.pool(freq_out).squeeze(-1)  # (batch, 128)

        # 融合
        combined = torch.cat([time_out, freq_out], dim=1)  # (batch, 256)

        # 分类
        output = self.fc(combined)  # (batch, num_classes)
        return output, combined



class LSTMBranch(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMBranch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=False)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # -> (batch, seq_len, input_size)
        out, _ = self.lstm(x)   # -> (batch, seq_len, hidden_size)
        out = out.reshape(out.shape[0], -1)  # 展平所有时间步： (batch, seq_len * hidden_size)
        return out  # 返回的是展开后的特征


class ParallelLSTMModel(nn.Module):
    def __init__(self, num_classes=21):
        super(ParallelLSTMModel, self).__init__()
        self.time_branch = LSTMBranch(input_size=1, hidden_size=128, num_layers=2)
        self.freq_branch = LSTMBranch(input_size=2, hidden_size=128, num_layers=2)

        # 假设序列长度为50，那么每个分支输出为 (batch, 128*50)
        self.fc = nn.Linear(128*50 + 128*50, num_classes)

    def forward(self, time_data, freq_data):
        time_feat = self.time_branch(time_data)  # (batch, 6400)
        freq_feat = self.freq_branch(freq_data)  # (batch, 6400)

        combined = torch.cat([time_feat, freq_feat], dim=1)  # (batch, 12800)
        out = self.fc(combined)  # (batch, num_classes)
        return out, combined


class MambaModel(nn.Module):
    def __init__(self, d_model=128, num_classes=10, d_state=16, d_conv=4, expand=2, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # 投影到d_model维度

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(num_layers)
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)     # -> (B, 50, d_model)

        for layer in self.mamba_layers:
            x = layer(x)           # 每一层 Mamba 处理

        x = x.permute(0, 2, 1)     # (B, d_model, 50)
        x = self.pooling(x).squeeze(-1)  # (B, d_model)
        # return self.classifier(x)        # (B, num_classes)
        return x



class ParallelMambaModel(nn.Module):
    def __init__(self, num_classes=10, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=1):
        super(ParallelMambaModel, self).__init__()

        self.time_proj = nn.Linear(1, d_model)
        self.freq_proj = nn.Linear(2, d_model)

        self.time_branch = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])
        self.freq_branch = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model * 2, num_classes)

    def forward(self, time_data, freq_data):
        time_out = time_data.permute(0, 2, 1)  # (B, 50, 1)
        time_out = self.time_proj(time_out)    # (B, 50, d_model)
        for layer in self.time_branch:
            time_out = layer(time_out)         # (B, 50, d_model)
        time_out = time_out.permute(0, 2, 1)   # (B, d_model, 50)
        time_out = self.pool(time_out).squeeze(-1)  # (B, d_model)

        freq_out = freq_data.permute(0, 2, 1)  # (B, 50, 2)
        freq_out = self.freq_proj(freq_out)    # (B, 50, d_model)
        for layer in self.freq_branch:
            freq_out = layer(freq_out)         # (B, 50, d_model)
        freq_out = freq_out.permute(0, 2, 1)   # (B, d_model, 50)
        freq_out = self.pool(freq_out).squeeze(-1)  # (B, d_model)

        combined = torch.cat([time_out, freq_out], dim=1)  # (B, 2*d_model)
        output = self.fc(combined)                         # (B, num_classes)
        return output, combined






class CNNMambaBranch(nn.Module):
    def __init__(self, in_channels, d_model, num_mamba_layers=1):
        super(CNNMambaBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            # nn.BatchNorm1d(d_model),
            # nn.ReLU()
        )
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(num_mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: (B, C, L)
        x = self.conv(x)  # → (B, d_model, L)
        x = x.permute(0, 2, 1)  # (B, L, d_model) → Mamba 输入格式
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return x

class CNNMambaFusionModel(nn.Module):
    def __init__(self, num_classes=10, d_model=128):
        super(CNNMambaFusionModel, self).__init__()
        self.time_branch = CNNMambaBranch(in_channels=1, d_model=d_model)
        self.freq_branch = CNNMambaBranch(in_channels=2, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, time_data, freq_data):
        # 输入为 (B, C, L)
        time_feat = self.time_branch(time_data)
        freq_feat = self.freq_branch(freq_data)
        fused = torch.cat([time_feat, freq_feat], dim=1)
        out = self.classifier(fused)
        return out, fused


class LSTMMambaBranch(nn.Module):
    def __init__(self, input_size, d_model, hidden_size=128, num_layers=2, mamba_layers=1):
        super(LSTMMambaBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=False
        )
        self.proj = nn.Linear(hidden_size, d_model)

        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: (B, C, L)
        x = x.permute(0, 2, 1)  # → (B, L, C) for LSTM
        lstm_out, _ = self.lstm(x)  # → (B, L, hidden_size)
        x = self.proj(lstm_out)     # → (B, L, d_model)

        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # → (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # → (B, d_model)
        return x

class LSTMMambaFusionModel(nn.Module):
    def __init__(self, num_classes=10, d_model=128):
        super(LSTMMambaFusionModel, self).__init__()
        self.time_branch = LSTMMambaBranch(input_size=1, d_model=d_model)
        self.freq_branch = LSTMMambaBranch(input_size=2, d_model=d_model)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, time_data, freq_data):  # (B, C, L)
        time_feat = self.time_branch(time_data)
        freq_feat = self.freq_branch(freq_data)
        fused = torch.cat([time_feat, freq_feat], dim=1)
        out = self.classifier(fused)
        return out, fused



if __name__ == "__main__":
    batch_size, seq_len = 64, 50
    time_data = torch.randn(batch_size, 1, seq_len).cuda()
    freq_data = torch.randn(batch_size, 2, seq_len).cuda()

    # model = LSTMMambaFusionModel(num_classes=10).cuda()
    model = MambaModel(num_classes=10).cuda()
    out = model(time_data)
    print("out.shape =", out.shape)  # (64, 10)
    # print("fused.shape =", fused.shape)  # (64, 128)




    # B, L = 64, 50
    # time_data = torch.randn(B, 1, L).cuda()
    # freq_data = torch.randn(B, 2, L).cuda()
    #
    # model = CNNMambaFusionModel(num_classes=10, d_model=64).cuda()
    # out, fused = model(time_data, freq_data)
    #
    # print("Output:", out.shape)  # (B, num_classes)
    # print("Fused feature:", fused.shape)  # (B, 128)






    # batch_size = 64
    # seq_len = 50
    # d_model = 32
    # num_classes = 10
    #
    # # 模拟数据：注意 shape 是 (B, C, L)
    # time_data = torch.randn(batch_size, 1, seq_len).cuda()  # 时域：(B, 1, 50)
    # freq_data = torch.randn(batch_size, 2, seq_len).cuda()  # 频域：(B, 2, 50)
    #
    # # 模型初始化
    # model = ParallelMambaModel(num_classes=num_classes, d_model=d_model).cuda()
    #
    # # 前向传播
    # output, combined = model(time_data, freq_data)
    #
    # print("✅ Output shape:", output.shape)       # (64, 10)
    # print("✅ Combined shape:", combined.shape)   # (64, 64)


