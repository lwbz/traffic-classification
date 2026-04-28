# -*- coding: utf-8 -*-
# @Project : PyCharm
# @File    : bilstm.py
# @Author  : Ronglin
# @Date    : 2025/11/20 14:26


import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim=8, n_layers=2, hidden_dim=256) -> None:
        super().__init__()
        self.input_dim = input_dim

        # 1. 开启双向 (bidirectional=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)

        # 2. 分类器的输入维度需要乘 2 (正向 hidden + 反向 hidden)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, feature=False):
        # 保持和原来的 LSTM 一样的输入处理逻辑
        x = x.view(len(x), 1, -1)

        # LSTM 前向传播
        # out: (batch, seq_len, hidden_dim * 2)
        # h_n: (num_layers * 2, batch, hidden_dim)
        out, (h_n, c_n) = self.lstm(x)

        # 3. 获取最终特征 (Feature Extraction)
        # h_n 的结构是 [layer1_fwd, layer1_back, layer2_fwd, layer2_back, ...]
        # 我们取最后两层（即最后一层的正向和反向）进行拼接
        # h_n[-2, :, :] -> 正向最后一步
        # h_n[-1, :, :] -> 反向最后一步
        x_f = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        # 分类
        x = self.classifier(x_f)

        # 保持统一的返回接口
        if feature:
            return x_f, x

        return x