import torch
import torch.nn as nn


# class LSTMModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=10, dropout=0.5):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # LSTM层
#         self.lstm = nn.LSTM(input_size=input_size,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             batch_first=True,
#                             dropout=dropout if num_layers > 1 else 0)
#
#         # 全连接层
#         self.fc = nn.Linear(hidden_size, num_classes)
#
#     def forward(self, x):
#         # 初始化隐藏状态和细胞状态
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#
#         x = x.permute(0, 2, 1)
#
#         # LSTM前向传播
#         out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_size)
#
#         # 取序列最后一个时间步的输出
#         out = out[:, -1, :]  # (batch, hidden_size)
#
#         # 全连接层
#         # out = self.fc(out)  # (batch, num_classes)
#         return out

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, flow_len=20, num_classes=10, dropout=0.3):
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
        self.fc = nn.Linear(hidden_size * flow_len, num_classes)

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


class MLP(torch.nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()  # 将(1024,1,50)展平为(1024,50)
        # 定义三层网络
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一层全连接
        self.ln1 = nn.LayerNorm(hidden_size)  # 第一层 LayerNorm
        self.act1 = nn.ReLU()  # 第一层激活函数

        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 第二层全连接
        self.ln2 = nn.LayerNorm(hidden_size)  # 第二层 LayerNorm
        self.act2 = nn.ReLU()  # 第二层激活函数

        self.fc3 = nn.Linear(hidden_size, num_classes)  # 第三层全连接
        self.ln3 = nn.LayerNorm(num_classes)  # 第三层 LayerNorm
        self.act3 = nn.ReLU()  # 第三层激活函数

    def forward(self, x):
        # 输入x形状: (batch_size, 1, 50) -> (1024, 1, 50)
        x = self.flatten(x)  # 展平为(1024, 50)
        # 前向传播
        x = torch.abs(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.ln3(x)
        y = self.act3(x)
        return y, []











