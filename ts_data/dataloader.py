import torch
import torch.utils.data as data

from ts_tfc_ssl.ts_model.graph_baselines import build_single_graph

from tqdm import *
import pickle


# UCRDataset 类继承了 torch.utils.data.Dataset，是一个用于加载 UCR 数据集的类。
# 在 __init__ 方法中，dataset 是输入数据（通常是一个多维数组或张量），target 是对应的标签。
# 如果 dataset 是一个 2D 张量（即形状为 (num_samples, series_length)），则通过 torch.unsqueeze() 将其扩展为 3D 张量，
# 形状变为 (num_samples, series_length, 1)。这是因为某些模型可能要求输入的维度是三维的（例如，包含一个维度表示特征通道）。
# self.target 保存了标签数据。
class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            # 2025/2/11 start
            # self.dataset = torch.unsqueeze(self.dataset, 1)
            self.dataset = torch.unsqueeze(self.dataset, -1)
            # 2025/2/11 end
        self.target = target

    # __getitem__ 方法使得我们能够通过索引访问数据集中的每一项，返回该索引处的数据和对应的标签。
    # self.dataset[index] 和 self.target[index] 分别取出数据和标签。
    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


class UEADataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset.permute(0, 2, 1)  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)



class IOTDataset(data.Dataset):
    def __init__(self, time_data, freq_data, target=None, is_labeled=True):
        self.time_data = time_data  # 时域数据
        self.freq_data = freq_data  # 频域数据
        if len(self.time_data.shape) == 2:
            self.time_data = torch.unsqueeze(self.time_data, -1)
        self.target = target if target is not None else torch.zeros(len(time_data), dtype=torch.int64)
        self.is_labeled = torch.tensor([is_labeled] * len(time_data), dtype=torch.bool)  # 添加标识

    # __getitem__ 方法使得我们能够通过索引访问数据集中的每一项，返回该索引处的数据和对应的标签。
    # self.dataset[index] 和 self.target[index] 分别取出数据和标签。
    def __getitem__(self, index):
        return self.time_data[index], self.freq_data[index], self.target[index], self.is_labeled[index]

    def __len__(self):
        return len(self.time_data)


class GraphDataset(data.Dataset):
    def __init__(self, x, y, dataset, mode, r):
        self.x = x
        self.y = y

        self.max_packet_length = 1500

        if mode == 'train':
            address = f'./IOT_Dataset/{dataset}/{dataset}_{mode}_{r}_graph.pickle'
        else:
            address = f'./IOT_Dataset/{dataset}/{dataset}_{mode}_graph.pickle'

        try:
            with open(address, 'rb') as f:
                self.all_graph = pickle.load(f)
        except FileNotFoundError as e:
            print(e)
            self.all_graph = []
            for i in trange(len(x)):
                seq_input = self.x[i]
                seq_input = torch.tensor(seq_input)
                # time_interval = seq_input[:-self.config.flow_length_limit]
                time_interval = None
                # seq_input = seq_input[-self.config.flow_length_limit:]
                seq_input /= self.max_packet_length
                graph = build_single_graph(seq_input, time_interval)
                self.all_graph.append(graph)
            with open(address, 'wb') as f:
                pickle.dump(self.all_graph, f)


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # seq_input = self.x[idx]
        # seq_input = torch.tensor(seq_input)
        # seq_input /= 1500
        seq_input = self.all_graph[idx]
        label = self.y[idx]
        return seq_input, label





if __name__ == '__main__':
    pass
