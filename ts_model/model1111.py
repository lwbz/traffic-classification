import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=10, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = x.permute(0, 2, 1)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)

        # 取序列最后一个时间步的输出
        out = out[:, -1, :]  # (batch, hidden_size)

        # 全连接层
        # out = self.fc(out)  # (batch, num_classes)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=False) # 单向 LSTM

        # 全连接层
        self.fc = nn.Linear(hidden_size * 50, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = x.permute(0, 2, 1)

        # LSTM前向传播
        # out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out.reshape(out.shape[0], -1)
        # 取序列最后一个时间步的输出
        # out = out[:, -1, :]  # (batch, hidden_size)

        embed = out

        # 全连接层
        out = self.fc(out)  # (batch, num_classes)
        return out, embed




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
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, output_dim=32) -> None:
        super(ProjectionHead, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, x):
        return self.projection_head(x)

class ParallelModel(nn.Module):
    def __init__(self, num_classes=21):
        super(ParallelModel, self).__init__()

        # 时域分支
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=8, padding=3),  # in_channels=1
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 池化到 (batch, 128, 1)
        )

        # 频域分支
        self.freq_branch = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=8, padding=3),  # in_channels=2
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 池化到 (batch, 128, 1)
        )

        # 融合后的全连接层
        self.fc = nn.Linear(128 + 128, num_classes)  # 128 (time) + 128 (freq)

    def forward(self, time_data, freq_data):
        # 时域分支
        time_out = self.time_branch(time_data)  # (batch, 128, 1)
        time_out = time_out.squeeze(-1)  # (batch, 128)

        # 频域分支
        freq_out = self.freq_branch(freq_data)  # (batch, 128, 1)
        freq_out = freq_out.squeeze(-1)  # (batch, 128)

        # 融合
        combined = torch.cat([time_out, freq_out], dim=1)  # (batch, 256)

        # 分类
        output = self.fc(combined)  # (batch, num_classes)
        return output

    def get_embedding(self, time_data, freq_data):
        time_out = self.time_branch(time_data)
        time_out = time_out.squeeze(-1)
        freq_out = self.freq_branch(freq_data)
        freq_out = freq_out.squeeze(-1)
        return torch.cat([time_out, freq_out], dim=1)








# 示例用法
if __name__ == "__main__":
    # # 假设你的数据是这样的 (50, 150)，即 50 个样本，150 个序列长度
    # data = torch.randn(50, 150).to("cuda")  # 模拟数据
    #
    # # 数据需要调整形状为 (batch_size, seq_length, feature_dim)
    # data = data.unsqueeze(-1)  # 变成 (50, 150, 1)
    #
    # # 创建 Mamba 模型实例
    # model = MambaModel(d_model=1, d_state=16, d_conv=4, expand=2).to("cuda")
    #
    # # 进行预测
    # output = model(data)
    #
    # # 输出结果
    # print(f"Mamba model output shape: {output.shape}")

    data = torch.randn(128, 50, 1).to("cuda")  # 模拟数据
    # model = FCN(10, 3).to("cuda")
    model = LSTMModel(num_classes=17).to("cuda")
    print("Output shape:", model(data).shape)  # 应为 (batch_size, 10)


