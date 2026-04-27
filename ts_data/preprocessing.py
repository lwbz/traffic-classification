import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# data-root：数据的根目录。
# dataset：具体数据集的名称。
# 此函数的目标是从 data-root/dataset/ 目录中读取指定的时间序列数据，并返回经过初步处理的特征和标签。
# def load_data(dataroot, dataset):
#     train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.csv'), sep=',', header=None)
#     train_x = train.iloc[:, 1:]      # 取出从第2列开始的所有列（特征部分）。
#     train_target = train.iloc[:, 0]  # 取出第1列（标签部分）。
#
#     test = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TEST.csv'), sep=',', header=None)
#     test_x = test.iloc[:, 1:]
#     test_target = test.iloc[:, 0]
#
#     sum_dataset = pd.concat([train_x, test_x]).to_numpy(dtype=np.float32)  # 将训练集和测试集的特征部分（train_x 和 test_x）按行合并。
#     # sum_dataset = sum_dataset.fillna(sum_dataset.mean()).to_numpy(dtype=np.float32)
#     sum_target = pd.concat([train_target, test_target]).to_numpy(dtype=np.float32)
#     # sum_target = sum_target.fillna(sum_target.mean()).to_numpy(dtype=np.float32)
#
#     num_classes = len(np.unique(sum_target))  # 找出标签中的唯一值（即类别），然后计算其长度。
#
#     return sum_dataset, sum_target, num_classes  # 合并后的所有样本特征 合并后的所有样本标签 类别总数

# def load_data(dataroot, dataset):
#     train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.csv'), sep=',', header=None)
#     train_x = train.iloc[:, 1:]  # 取出从第2列开始的所有列（特征部分）。
#     train_target = train.iloc[:, 0]  # 取出第1列（标签部分）。
#
#     dataset = train_x.to_numpy(dtype=np.float32)  # 直接将训练集特征转换为numpy数组。
#     target = train_target.to_numpy(dtype=np.float32)  # 直接将训练集标签转换为numpy数组。
#
#     num_classes = len(np.unique(target))  # 找出标签中的唯一值（即类别），然后计算其长度。
#
#     return dataset, target, num_classes  # 所有样本特征，所有样本标签，类别总数

def load_data(dataroot, dataset):
    # 读取训练数据
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.csv'), sep=',', header=None)
    train_x = train.iloc[:, 1:]  # 取出从第2列开始的所有列（特征部分）
    train_target = train.iloc[:, 0]  # 取出第1列（标签部分）

    # 转换为 numpy 数组
    dataset = train_x.to_numpy(dtype=np.float32)
    target = train_target.to_numpy(dtype=np.float32)

    # 计算每个类别的样本数
    class_counts = Counter(target)
    num_classes = len(class_counts)

    # 上采样：将样本数少于200的类别上采样到200
    new_dataset = []
    new_target = []

    for cls in class_counts:
        # 获取当前类别的样本索引
        cls_indices = np.where(target == cls)[0]
        cls_samples = dataset[cls_indices]
        cls_targets = target[cls_indices]

        # 如果当前类别样本数少于100，进行上采样
        if len(cls_samples) < 200:
            # 计算需要重复的次数
            repeat_times = int(np.ceil(200 / len(cls_samples)))
            # 重复样本
            upsampled_samples = np.repeat(cls_samples, repeat_times, axis=0)[:200]
            upsampled_targets = np.repeat(cls_targets, repeat_times)[:200]
        else:
            # 如果样本数已经足够，直接使用原始样本
            upsampled_samples = cls_samples
            upsampled_targets = cls_targets

        # 将上采样后的样本和标签添加到新列表
        new_dataset.append(upsampled_samples)
        new_target.append(upsampled_targets)

    # 合并所有类别的样本和标签
    dataset = np.vstack(new_dataset)
    target = np.concatenate(new_target)

    return dataset, target, num_classes



# 这段代码定义了一个函数 transfer_labels，用于将标签转换为新的数值形式，通常是为了归一化标签，
# 使它们从 0 开始编号，并确保标签的格式一致。
def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def my_k_fold(data, target, args):
    # 划分 80% 训练集 + 20% 测试集和验证集
    train_set, temp_set, train_target, temp_target = train_test_split(data, target, test_size=0.2, stratify=target,
                                                                      shuffle=True, random_state=args.random_seed)

    # 将剩下的 20% 再划分为 10% 验证集 + 10% 测试集
    val_set, test_set, val_target, test_target = train_test_split(temp_set, temp_target, test_size=0.5,
                                                                  stratify=temp_target, shuffle=True, random_state=args.random_seed)

    return [train_set], [train_target], [val_set], [val_target], [test_set], [test_target]

# 这段代码定义了一个 k_fold 函数，用于实现数据集的分层交叉验证 (Stratified K-Fold Cross Validation)。
def k_fold(data, target):
    skf = StratifiedKFold(5, shuffle=True)
    train_sets = []
    train_targets = []

    val_sets = []
    val_targets = []

    test_sets = []
    test_targets = []

    for raw_index, test_index in skf.split(data, target):
        raw_set = data[raw_index]
        raw_target = target[raw_index]

        train_index, val_index = next(StratifiedKFold(4, shuffle=True).split(raw_set, raw_target))
        train_sets.append(raw_set[train_index])
        train_targets.append(raw_target[train_index])

        val_sets.append(raw_set[val_index])
        val_targets.append(raw_target[val_index])

        test_sets.append(data[test_index])
        test_targets.append(target[test_index])

    return train_sets, train_targets, val_sets, val_targets, test_sets, test_targets


# 这段代码定义了一个 normalize_per_series 函数，用于对时间序列数据按序列进行标准化处理。
# def normalize_per_series(data):
#     std_ = data.std(axis=1, keepdims=True)
#     std_[std_ == 0] = 1.0
#     return (data - data.mean(axis=1, keepdims=True)) / std_
#
# def normalize_per_series(data, mtu):
#     return np.minimum(data, mtu) / mtu

def normalize_per_series(data, mtu, flag):
    if flag == 0:
        std_ = data.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        return (data - data.mean(axis=1, keepdims=True)) / std_
    else:
        return np.minimum(data, mtu) / mtu


import torch


# def normalize_freq_data(freq_data):
#     """
#     归一化频域数据到 [0, 1] 范围，按样本独立进行。
#
#     Args:
#         freq_data (torch.Tensor): 频域数据，形状为 (n_samples, 2, 26)，包含实部和虚部
#
#     Returns:
#         torch.Tensor: 归一化后的频域数据，形状不变
#     """
#     # 计算每个样本的幅度（实部和虚部的 L2 范数）
#     magnitude = torch.sqrt(freq_data[:, 0, :] ** 2 + freq_data[:, 1, :] ** 2)  # (n_samples, 26)
#
#     # 按样本计算最小值和最大值
#     min_val = magnitude.min(dim=1, keepdim=True)[0]  # (n_samples, 1)
#     max_val = magnitude.max(dim=1, keepdim=True)[0]  # (n_samples, 1)
#
#     # 避免除以零
#     range_val = max_val - min_val
#     range_val[range_val == 0] = 1.0  # 如果范围为 0，设为 1 防止除零
#
#     # 归一化到 [0, 1]
#     normalized_magnitude = (magnitude - min_val) / range_val  # (n_samples, 26)
#
#     # 将归一化后的幅度应用到实部和虚部（保持相位）
#     scale = normalized_magnitude / (magnitude + 1e-8)  # 避免除以零
#     normalized_freq = freq_data.clone()
#     normalized_freq[:, 0, :] *= scale  # 实部
#     normalized_freq[:, 1, :] *= scale  # 虚部
#
#     return normalized_freq

def normalize_freq_data(freq_data, mode='both'):
    """
    归一化频域数据到 [0, 1] 范围，按样本独立进行。

    Args:
        freq_data (torch.Tensor): 频域数据
            - 如果 mode='both': 形状为 (n_samples, 2, 26)，包含振幅和相位
            - 如果 mode='amp_only' 或 'phase_only': 形状为 (n_samples, 1, 26)
        mode (str): 'both', 'amp_only' 或 'phase_only'

    Returns:
        torch.Tensor: 归一化后的频域数据，形状不变
    """
    if mode == 'both':
        # 原始复数归一化逻辑（保持相位）
        magnitude = torch.sqrt(freq_data[:, 0, :] ** 2 + freq_data[:, 1, :] ** 2)
    else:
        # 仅振幅或仅相位的情况，直接使用输入数据
        magnitude = freq_data[:, 0, :]  # (n_samples, 26)

    # 计算最小最大值
    min_val = magnitude.min(dim=1, keepdim=True)[0]  # (n_samples, 1)
    max_val = magnitude.max(dim=1, keepdim=True)[0]  # (n_samples, 1)

    # 避免除以零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0

    # 归一化
    normalized = (magnitude - min_val) / range_val

    if mode == 'both':
        # 复数情况：缩放实部和虚部
        scale = normalized / (magnitude + 1e-8)
        normalized_freq = freq_data.clone()
        normalized_freq[:, 0, :] *= scale
        normalized_freq[:, 1, :] *= scale
        return normalized_freq
    else:
        # 仅振幅或仅相位：直接返回归一化结果
        return normalized.unsqueeze(1)  # 恢复形状 (n_samples, 1, 26)


# 使用示例
# train_fft = normalize_freq_data(train_fft)
# val_fft = normalize_freq_data(val_fft)
# test_fft = normalize_freq_data(test_fft)


# 这段代码定义了一个 normalize_uea_set 函数，用于对多变量时间序列数据集进行标准化处理。
def normalize_uea_set(data_set):
    '''
    The function is the same as normalize_per_series, but can be used for multiple variables.
    '''
    return TimeSeriesScalerMeanVariance().fit_transform(data_set)


# 这段代码定义了一个 fill_nan_value 函数，用于填充训练集、验证集和测试集中缺失值（NaN）的处理逻辑。
def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set


if __name__ == '__main__':
    pass
