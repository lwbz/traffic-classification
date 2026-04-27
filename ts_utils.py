import random
from collections import Counter
from sklearn.metrics import accuracy_score

import numpy as np
import torch
import torch.optim

from ts_tfc_ssl.ts_data.preprocessing import load_data, transfer_labels, k_fold, my_k_fold
from ts_tfc_ssl.ts_model.dapp import DAPP
from ts_tfc_ssl.ts_model.graph_baselines import get_gnn_model, get_gcn_model, get_gat_model, get_gin_model
from ts_tfc_ssl.ts_model.graphiot import GraphIoT
from ts_tfc_ssl.ts_model.loss import cross_entropy, reconstruction_loss
from ts_tfc_ssl.ts_model.model import FCN, Classifier, ParallelModel, ParallelTransformerModel, ParallelLSTMModel, \
    ParallelMambaModel, CNNMambaFusionModel, LSTMMambaFusionModel, MambaModel
from ts_tfc_ssl.ts_model.baseline_model import LSTM, MLP
# from ts_tfc_ssl.ts_model.models_net_mamba import NetMamba

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def build_model(args):
    if args.backbone == 'FCN_Time' or args.backbone == 'FCN_Freq' or args.backbone == 'FCN':
        model = FCN(args.num_classes, args.input_size)

    if args.backbone == 'Time_Freq':
        model = ParallelModel(args.num_classes)

    if args.backbone == 'ParallelLSTMModel':
        model = ParallelLSTMModel(args.num_classes)

    if args.backbone == 'ParallelMambaModel':
        model = ParallelMambaModel(args.num_classes)

    if args.backbone == 'CNNMambaFusionModel':
        model = CNNMambaFusionModel(args.num_classes)

    if args.backbone == 'LSTMMambaFusionModel':
        model = LSTMMambaFusionModel(args.num_classes)

    if args.backbone == 'MambaModel':
        model = MambaModel()

    if args.backbone == 'Transformer_TF':
        model = ParallelTransformerModel(args.num_classes)

    if args.backbone == 'LSTM':
        model = LSTM(flow_len=20, num_classes=args.num_classes)

    if args.backbone == 'MLP':
        model = MLP(input_size=args.seq_len, num_classes=args.num_classes)

    if args.backbone == 'DAPP':
        model = DAPP(num_classes=args.num_classes, max_flow_length=args.max_flow_length)

    if args.backbone == 'GCN':
        model = get_gcn_model(num_classes=args.num_classes, max_flow_length=args.max_flow_length)

    if args.backbone == 'GAT':
        model = get_gat_model(num_classes=args.num_classes, max_flow_length=args.max_flow_length)

    if args.backbone == 'GIN':
        model = get_gin_model(num_classes=args.num_classes, max_flow_length=args.max_flow_length)

    if args.backbone == 'GraphIoT':
        model = GraphIoT(num_classes=args.num_classes, max_flow_length=args.max_flow_length)

    if args.classifier == 'linear':
        classifier = Classifier(args.classifier_input, args.num_classes)

    return model, classifier


def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)

    sum_target = transfer_labels(sum_target)
    return sum_dataset, sum_target, num_classes


def build_loss(args, class_weights):
    if args.loss == 'cross_entropy':
        return cross_entropy(class_weights)
    elif args.loss == 'reconstruction':
        return reconstruction_loss()


def shuffler(x_train, y_train):
    indexes = np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    return x_train, y_train


def get_all_datasets(data, target, args):
    # return k_fold(data, target)
    return my_k_fold(data, target, args)

import dgl
def custom_collate_fn(batch):
    from torch.utils.data.dataloader import default_collate
    seq_input, labels = zip(*batch)
    seq_input = dgl.batch(seq_input)
    label = torch.as_tensor(labels, dtype=torch.int64)
    return seq_input, label


# convert_coeff 函数的目的是将复数形式的数据进行转换，将其表示为 幅值 和 相位，并返回处理后的张量以及相位值。
# x：一个复数张量，包含实部和虚部，通常是通过 FFT 变换得到的复数频率系数。
# eps：一个非常小的值，用于避免除零或对零值进行平方根操作导致的数值不稳定。
def convert_coeff(x, eps=1e-6, return_type='both'):
    # 计算幅值：x.real 是复数张量的实部，x.imag 是复数张量的虚部。
    amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
    # 计算相位：torch.atan2 是计算复数相位的函数
    phase = torch.atan2(x.imag, x.real + eps)

    if return_type == 'amp_only':
        return amp.unsqueeze(1)  # 保持维度一致性 (n_samples, 1, 26)
    elif return_type == 'phase_only':
        return phase.unsqueeze(1)  # (n_samples, 1, 26)
    else:
        # 将幅值和相位堆叠为新的张量：torch.stack 将幅值和相位在最后一个维度（-1）上合并，形成一个新的张量。
        stack_r = torch.stack((amp, phase), -1)
        # 调整维度顺序：原始维度是 (batch_size, freq_bins, 2)，调整为 (batch_size, 2, freq_bins)。这样可以让幅值和相位沿着通道维度分开。
        # 2025/2/11 start
        stack_r = stack_r.permute(0, 2, 1)
        # 2025/2/11 end
        return stack_r, phase


# def evaluate(val_loader, model, classifier, loss, device, flag):
#     val_loss = 0
#     val_accu = 0
#
#     sum_len = 0
#     for data, target in val_loader:
#         '''
#         data, target = data.to(device), target.to(device)
#         target = target.to(torch.int64)
#         '''
#         # 2025/2/11 start 满足mamba模型输入
#         # data = data.permute(0, 2, 1)
#         # 2025/2/11 end
#         with torch.no_grad():
#             val_pred = model(data)
#             val_pred = classifier(val_pred)
#             val_loss += loss(val_pred, target).item()
#
#             val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
#             sum_len += len(target)
#
#     return val_loss / sum_len, val_accu / sum_len


def evaluate(val_loader, model, classifier, loss, device, flag):
    val_loss = 0
    val_accu = 0
    sum_len = 0

    all_targets = []
    all_preds = []

    if flag == 0 or flag == 1:   # time_only | freq_only
        for data, target in val_loader:
            with torch.no_grad():
                val_pred = model(data)
                val_pred = classifier(val_pred)
                val_loss += loss(val_pred, target).item()

                preds = torch.argmax(val_pred.data, axis=1)
                val_accu += torch.sum(preds == target)
                sum_len += len(target)

                # 收集预测结果和真实标签
                all_targets.append(target.cpu())
                all_preds.append(preds.cpu())
    # elif flag == 1:   # freq_only
    #     pass
    elif flag == 2:   # time+freq
        for time_data, freq_data, target, _ in val_loader:
            with torch.no_grad():
                val_pred, _ = model(time_data, freq_data)
                # val_pred = model(time_data, freq_data)
                val_loss += loss(val_pred, target).item()

                preds = torch.argmax(val_pred.data, axis=1)
                val_accu += torch.sum(preds == target)
                sum_len += len(target)

                # 收集预测结果和真实标签
                # all_targets.extend(target.cpu().numpy())
                # all_preds.extend(preds.cpu().numpy())

                all_targets.append(target.cpu())
                all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # for cls in range(21):
    #     mask = all_targets == cls
    #     if mask.sum() > 0:
    #         acc = accuracy_score(all_targets[mask], all_preds[mask])
    #         print(f"Class {cls} Acc: {acc:.4f}")


    # average='macro'	对每个类别分别计算召回率/F1分数，再取平均，适合类别样本数量均衡的任务
    # average='micro'	全部类别一起计算，不区分类别
    # average='weighted'	按类别样本数量加权平均，适合类别样本数量不平衡的任务

    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    micro_precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_preds, average='micro')
    micro_f1 = f1_score(all_targets, all_preds, average='micro')

    weighted_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_targets, all_preds, average='weighted')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # return val_loss / sum_len, val_accu / sum_len, recall, f1
    return val_loss / sum_len, val_accu / sum_len, (macro_precision, micro_precision, weighted_precision), (macro_recall, micro_recall, weighted_recall), (
    macro_f1, micro_f1, weighted_f1), conf_matrix


def evaluate_base(val_loader, model, loss, device):
    val_loss = 0
    val_accu = 0
    sum_len = 0

    all_targets = []
    all_preds = []

    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            val_pred, _ = model(data)
            val_loss += loss(val_pred, target).item()

            preds = torch.argmax(val_pred.data, axis=1)
            val_accu += torch.sum(preds == target)
            sum_len += len(target)

            # 收集预测结果和真实标签
            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # average='macro'	对每个类别分别计算召回率/F1分数，再取平均，适合类别样本数量均衡的任务
    # average='micro'	全部类别一起计算，不区分类别
    # average='weighted'	按类别样本数量加权平均，适合类别样本数量不平衡的任务

    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    micro_precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_preds, average='micro')
    micro_f1 = f1_score(all_targets, all_preds, average='micro')

    weighted_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_targets, all_preds, average='weighted')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # return val_loss / sum_len, val_accu / sum_len, recall, f1
    return val_loss / sum_len, val_accu / sum_len, (macro_precision, micro_precision, weighted_precision), (macro_recall, micro_recall, weighted_recall), (
    macro_f1, micro_f1, weighted_f1), conf_matrix


def evaluate_graph(val_loader, model, loss, device):
    val_loss = 0
    val_accu = 0
    sum_len = 0

    all_targets = []
    all_preds = []

    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            val_pred = model(data)
            val_loss += loss(val_pred, target).item()

            preds = torch.argmax(val_pred.data, axis=1)
            val_accu += torch.sum(preds == target)
            sum_len += len(target)

            # 收集预测结果和真实标签
            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # average='macro'	对每个类别分别计算召回率/F1分数，再取平均，适合类别样本数量均衡的任务
    # average='micro'	全部类别一起计算，不区分类别
    # average='weighted'	按类别样本数量加权平均，适合类别样本数量不平衡的任务

    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    micro_precision = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    micro_recall = recall_score(all_targets, all_preds, average='micro')
    micro_f1 = f1_score(all_targets, all_preds, average='micro')

    weighted_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_targets, all_preds, average='weighted')
    weighted_f1 = f1_score(all_targets, all_preds, average='weighted')

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # return val_loss / sum_len, val_accu / sum_len, recall, f1
    return val_loss / sum_len, val_accu / sum_len, (macro_precision, micro_precision, weighted_precision), (macro_recall, micro_recall, weighted_recall), (
    macro_f1, micro_f1, weighted_f1), conf_matrix


def construct_graph_via_knn_cpl_nearind_gpu(data_embed, y_label, mask_label, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.85, num_real_class=2):
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    data_embed = data_embed
    emb_all = data_embed / (sigma + eps)  # n*d
    emb1 = torch.unsqueeze(emb_all, 1)  # n*1*d
    emb2 = torch.unsqueeze(emb_all, 0)  # 1*n*d
    w = ((emb1 - emb2) ** 2).mean(2)  # n*n*d -> n*n
    w = torch.exp(-w / 2)

    ## keep top-k values
    topk, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)  # union, knn graph
    # mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, knn graph
    w = w * mask

    ## normalize
    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    # step2: label propagation, f = (i-\alpha s)^{-1}y
    y = torch.zeros(y_label.shape[0], num_real_class)
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))
    all_knn_label = torch.argmax(f, 1).cpu().numpy()
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    class_counter = Counter(y_label)
    for i in range(num_real_class):
        class_counter[i] = 0
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i]
        else:
            class_counter[all_knn_label[i]] += 1

    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        classwise_acc[i] = class_counter[i] / max(class_counter.values())
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))

    # print(f"Max Probabilities: {max_probs}")
    # print(f"Number of selected pseudo-labels: {cpl_mask.sum().item()} / {len(cpl_mask)}")
    #
    # pseudo_label_counts = Counter(all_knn_label)  # all_knn_label 是标签传播得到的伪标签
    # print(f"伪标签类别分布: {pseudo_label_counts}")

    return all_knn_label, end_knn_label, cpl_mask, indices


# 定义伪标签生成函数
def generate_pseudo_labels(model, unlabeled_loader, device, threshold=0.95):
    model.eval()
    pseudo_labels, pseudo_time_list, pseudo_freq_list = [], [], []
    with torch.no_grad():
        for time_data, freq_data, _, _ in unlabeled_loader:
            time_data, freq_data = time_data.to(device), freq_data.to(device)
            output = model(time_data, freq_data)
            probs = F.softmax(output, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            mask = max_probs > threshold
            pseudo_labels.append(preds[mask].cpu())
            pseudo_time_list.append(time_data[mask].cpu())
            pseudo_freq_list.append(freq_data[mask].cpu())
    return (torch.cat(pseudo_labels) if pseudo_labels else torch.tensor([]),
            torch.cat(pseudo_time_list) if pseudo_time_list else torch.tensor([]),
            torch.cat(pseudo_freq_list) if pseudo_freq_list else torch.tensor([]))


# CPL 伪标签生成函数
# def generate_cpl_pseudo_labels(model, unlabeled_loader, device, num_classes=21, gamma=0.99):
#     model.eval()
#     all_preds, all_probs, all_time, all_freq = [], [], [], []
#     with torch.no_grad():
#         for time_data, freq_data, _ in unlabeled_loader:
#             time_data, freq_data = time_data.to(device), freq_data.to(device)
#             output = model(time_data, freq_data)
#             probs = F.softmax(output, dim=-1)
#             max_probs, preds = torch.max(probs, dim=-1)
#             all_preds.append(preds.cpu())
#             all_probs.append(max_probs.cpu())
#             all_time.append(time_data.cpu())
#             all_freq.append(freq_data.cpu())
#
#     all_preds = torch.cat(all_preds)
#     all_probs = torch.cat(all_probs)
#     all_time = torch.cat(all_time)
#     all_freq = torch.cat(all_freq)
#
#     # 计算每个类别的样本计数
#     class_counter = Counter(all_preds.numpy())
#     for i in range(num_classes):
#         class_counter[i] = class_counter.get(i, 0)  # 补齐未出现的类别
#
#     # 计算类别的“学习难度”（归一化计数）
#     classwise_acc = torch.zeros(num_classes).to(device)
#     max_count = max(class_counter.values()) or 1  # 避免除以 0
#     for i in range(num_classes):
#         classwise_acc[i] = class_counter[i] / max_count
#
#     # 计算动态阈值 Te(c)
#     Te = gamma * (classwise_acc / (2. - classwise_acc))
#
#     # 选择伪标签
#     cpl_mask = all_probs.to(device) >= Te[all_preds.to(device)]
#     pseudo_labels = all_preds[cpl_mask.cpu()]
#     pseudo_time = all_time[cpl_mask.cpu()]
#     pseudo_freq = all_freq[cpl_mask.cpu()]
#
#     return pseudo_labels, pseudo_time, pseudo_freq



def augment_data(time_data, freq_data, noise_factor=0.01):
    return (time_data + torch.randn_like(time_data) * noise_factor,
            freq_data + torch.randn_like(freq_data) * noise_factor)


def generate_cpl_pseudo_labels(model, unlabeled_loader, device, num_classes=21, gamma=0.995, max_pseudo=50000):
    model.eval()
    all_preds, all_probs, all_time, all_freq = [], [], [], []
    with torch.no_grad():
        for time_data, freq_data, _, _ in unlabeled_loader:
            # 原始预测
            time_data, freq_data = time_data.to(device), freq_data.to(device)
            output = model(time_data, freq_data)
            probs = F.softmax(output, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)

            # 增强预测
            time_data_aug, freq_data_aug = augment_data(time_data, freq_data)
            output_aug = model(time_data_aug, freq_data_aug)
            probs_aug = F.softmax(output_aug, dim=-1)
            max_probs_aug, preds_aug = torch.max(probs_aug, dim=-1)

            # 一致性检查
            consistent_mask = preds == preds_aug
            all_preds.append(preds[consistent_mask])  # 保持在 GPU
            all_probs.append(max_probs[consistent_mask])
            all_time.append(time_data[consistent_mask])
            all_freq.append(freq_data[consistent_mask])

    all_preds = torch.cat(all_preds)  # GPU
    all_probs = torch.cat(all_probs)  # GPU
    all_time = torch.cat(all_time)  # GPU
    all_freq = torch.cat(all_freq)  # GPU

    # 计算 classwise_acc（需要 CPU 数据给 Counter）
    class_counter = Counter(all_preds.cpu().numpy())
    for i in range(num_classes):
        class_counter[i] = class_counter.get(i, 0)

    classwise_acc = torch.zeros(num_classes).to(device)
    max_count = max(class_counter.values()) or 1
    for i in range(num_classes):
        classwise_acc[i] = class_counter[i] / max_count

    # 计算 Te(c)
    Te = gamma * (classwise_acc / (2. - torch.clamp(classwise_acc, max=1.99)))
    cpl_mask = all_probs >= Te[all_preds]  # 都在 GPU 上

    # 限制伪标签数量
    if cpl_mask.sum() > max_pseudo:
        _, indices = all_probs[cpl_mask].topk(max_pseudo)
        mask = torch.zeros_like(cpl_mask, dtype=torch.bool, device=device)
        mask[indices] = True
        cpl_mask = mask

    # 返回时移到 CPU
    return (all_preds[cpl_mask].cpu(), all_time[cpl_mask].cpu(), all_freq[cpl_mask].cpu())


# 上采样和下采样函数
# def balance_classes(time_data, freq_data, labels, target_count=2500, noise_factor=0.01):
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     label_counts = dict(zip(unique_labels, counts))
#
#     balanced_time = []
#     balanced_freq = []
#     balanced_labels = []
#
#     for label in unique_labels:
#         mask = labels == label
#         class_time = time_data[mask]
#         class_freq = freq_data[mask]
#         class_count = label_counts[label]
#
#         if class_count < target_count:  # 上采样少数类别
#             repeat_factor = int(np.ceil(target_count / class_count))
#             repeated_time = class_time.repeat(repeat_factor, 0)[:target_count]
#             repeated_freq = class_freq.repeat(repeat_factor, 0)[:target_count]
#             repeated_labels = np.full(target_count, label)
#             noise_time = torch.randn_like(repeated_time) * noise_factor
#             noise_freq = torch.randn_like(repeated_freq) * noise_factor
#             balanced_time.append(repeated_time + noise_time)
#             balanced_freq.append(repeated_freq + noise_freq)
#             balanced_labels.append(repeated_labels)
#         elif class_count > target_count:  # 下采样多数类别
#             indices = np.random.choice(class_count, target_count, replace=False)
#             balanced_time.append(class_time[indices])
#             balanced_freq.append(class_freq[indices])
#             balanced_labels.append(np.full(target_count, label))
#         else:
#             balanced_time.append(class_time)
#             balanced_freq.append(class_freq)
#             balanced_labels.append(np.full(class_count, label))
#
#     balanced_time = torch.cat(balanced_time, dim=0)
#     balanced_freq = torch.cat(balanced_freq, dim=0)
#     balanced_labels = np.concatenate(balanced_labels)
#     return balanced_time, balanced_freq, balanced_labels
#


def balance_classes(time_data, freq_data, labels, target_count=2500, noise_factor=0.01):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    balanced_time = []
    balanced_freq = []
    balanced_labels = []

    for label in unique_labels:
        mask = labels == label
        class_time = time_data[mask]
        class_freq = freq_data[mask]
        class_count = label_counts[label]

        # 确保 class_time 是至少 2 维
        if class_time.dim() == 1:
            class_time = class_time.unsqueeze(1)  # (n,) -> (n, 1)
        elif class_time.dim() == 2:
            class_time = class_time.unsqueeze(1)  # (n, 50) -> (n, 1, 50)

        if class_count < target_count:  # 上采样
            repeat_factor = int(np.ceil(target_count / class_count))
            if class_time.dim() == 3:
                repeated_time = class_time.repeat(repeat_factor, 1, 1)[:target_count]
            else:
                repeated_time = class_time.repeat(repeat_factor, 1)[:target_count]
            repeated_freq = class_freq.repeat(repeat_factor, 1, 1)[:target_count]
            repeated_labels = np.full(target_count, label)
            noise_time = torch.randn_like(repeated_time) * noise_factor
            noise_freq = torch.randn_like(repeated_freq) * noise_factor
            balanced_time.append(repeated_time + noise_time)
            balanced_freq.append(repeated_freq + noise_freq)
            balanced_labels.append(repeated_labels)
        elif class_count > target_count:  # 下采样
            indices = np.random.choice(class_count, target_count, replace=False)
            balanced_time.append(class_time[indices])
            balanced_freq.append(class_freq[indices])
            balanced_labels.append(np.full(target_count, label))
        else:
            balanced_time.append(class_time)
            balanced_freq.append(class_freq)
            balanced_labels.append(np.full(class_count, label))

    balanced_time = torch.cat(balanced_time, dim=0)
    balanced_freq = torch.cat(balanced_freq, dim=0)
    balanced_labels = np.concatenate(balanced_labels)
    return balanced_time, balanced_freq, balanced_labels


# 弱增强和强增强
def weak_augment(time_data, freq_data, scale_range=0.1, max_shift=3):
    # scale = 1 + torch.rand(1).item() * scale_range - scale_range / 2  # [0.95, 1.05]
    # shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    # time_shifted = torch.roll(time_data, shifts=shift, dims=2)
    # return time_shifted * scale, freq_data  # 频域不变
    return time_data, freq_data


# def strong_augment(time_data, freq_data, noise_factor=0.05, mask_ratio=0.2, perturb_ratio=0.3):
#     # 强噪声
#     noise_time = torch.randn_like(time_data) * noise_factor
#     noise_freq = torch.randn_like(freq_data) * noise_factor
#
#     # 时间掩码
#     seq_len = time_data.size(2)
#     mask_len = int(seq_len * mask_ratio)
#     start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
#     mask = torch.ones_like(time_data)
#     mask[:, :, start:start + mask_len] = 0
#     time_masked = (time_data + noise_time) * mask
#
#     # 频率扰动
#     mask_freq = torch.rand_like(freq_data) > perturb_ratio
#     freq_perturbed = (freq_data + noise_freq) * mask_freq + noise_freq * (~mask_freq)
#     return time_masked, freq_perturbed

# def strong_augment(time_data, freq_data, noise_factor=0.1, warp_factor=0.3, mask_ratio=0.3):
#     # 时间扭曲
#     seq_len = time_data.size(2)
#     time_steps = torch.linspace(0, seq_len - 1, seq_len).to(time_data.device)
#     warp = torch.rand(1).item() * warp_factor - warp_factor / 2
#     warped_steps = time_steps + warp * torch.sin(time_steps)
#     warped_time = F.interpolate(time_data, size=seq_len, mode='linear', align_corners=False)
#
#     # 强噪声
#     noise_time = torch.randn_like(warped_time) * noise_factor
#     noise_freq = torch.randn_like(freq_data) * noise_factor
#
#     # 频率屏蔽
#     mask_freq = torch.rand_like(freq_data) > mask_ratio
#     freq_masked = freq_data * mask_freq

    return warped_time + noise_time, freq_masked + noise_freq


def strong_augment(time_data, freq_data, device, noise_factor=0.1, warp_factor=0.3, mask_ratio=0.3, drop_ratio=0.2):
    # 时间扭曲
    seq_len = time_data.size(2)
    time_steps = torch.linspace(0, seq_len - 1, seq_len).to(device)
    warp = torch.rand(1).item() * warp_factor - warp_factor / 2
    warped_time = F.interpolate(time_data, size=seq_len, mode='linear', align_corners=False)

    # 强噪声
    noise_time = torch.randn_like(warped_time) * noise_factor
    noise_freq = torch.randn_like(freq_data) * noise_factor

    # 时间掩码
    mask_len = int(seq_len * mask_ratio)
    start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
    mask_time = torch.ones_like(warped_time)
    mask_time[:, :, start:start + mask_len] = 0

    # 频率屏蔽 + 丢弃
    mask_freq = torch.rand_like(freq_data) > mask_ratio
    drop_freq = torch.rand_like(freq_data) > drop_ratio

    time_aug = (warped_time + noise_time) * mask_time
    freq_aug = (freq_data * mask_freq + noise_freq) * drop_freq

    return time_aug, freq_aug


# CPL 阈值生成函数
def generate_cpl_mask(max_probs, pseudo_labels, classwise_acc, p_cutoff=0.95, num_classes=21):
    # 计算类特定阈值
    adaptive_factor = classwise_acc[pseudo_labels] / (2. - classwise_acc[pseudo_labels] + 1e-8)  # 避免除零
    thresholds = p_cutoff * adaptive_factor
    mask = max_probs.ge(thresholds)
    return mask



# KNN + CPL 标签传播函数
# data_embed
#   类型：torch.Tensor
#   形状：(n_samples, embed_dim)，其中 n_samples 是样本数量，embed_dim 是嵌入特征的维度。
#   作用：输入的嵌入特征数据，用于构建 KNN 图。每个样本的特征向量表示其在特征空间中的位置，通常由模型（例如你的并行网络）生成。

# y_label
#   类型：torch.Tensor 或 np.ndarray
#   形状：(n_samples,)，长度与样本数量相同。
#   作用：初始标签向量。有标签样本的真实类别（例如 0 到 20），未标记样本通常初始化为无效值（例如 0 或随机值）。用于标签传播的起点。

# mask_label
#   类型：np.ndarray 或 torch.Tensor
#   形状：(n_samples,)，长度与样本数量相同。
#   作用：掩码数组，标记每个样本是否为有标签数据。
#   0：表示该样本有真实标签（y_label 中对应值为真实类别）。
#   1：表示该样本无标签（需要生成伪标签）。

# topk（默认值：10）
#   类型：int
#   作用：KNN 图中每个样本保留的最近邻数量。控制图的稀疏性，影响标签传播的局部性。
#   值越大，图越稠密，传播范围更广，但噪声可能增加。
#   值越小，图越稀疏，传播更局部，计算更快。
#   示例：topk=5，每个样本连接 5 个最近邻。

# sigma（默认值：0.25）
#   类型：float
#   作用：高斯核的带宽参数，控制相似性权重的衰减速度。
#   sigma 越大，距离较远的样本仍有较大权重，图更平滑。
#   sigma 越小，权重集中于近邻，图更尖锐。
#   示例：sigma=0.25，适用于特征距离较小的场景。

# alpha（默认值：0.99）
#   类型：float，范围 [0, 1]
#   作用：标签传播的步长参数，控制传播矩阵的影响力度。
#   alpha 接近 1，传播依赖更多邻居信息。
#   alpha 接近 0，传播更依赖初始标签。
#   示例：alpha=0.99，强传播，适合未标记数据较多的情况。

# p_cutoff（默认值：0.85）
#   类型：float，范围 [0, 1]
#   作用：Curriculum Pseudo-Labeling (CPL) 的全局置信度阈值，用于筛选伪标签。
#   高于该阈值的伪标签被选中（cpl_mask）。
#   示例：p_cutoff=0.85，要求伪标签置信度较高。
def construct_graph_via_knn_cpl_nearind_gpu(data_embed, y_label, mask_label, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.90, num_real_class=21):
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    emb_all = data_embed / (sigma + eps)
    emb1 = torch.unsqueeze(emb_all, 1)
    emb2 = torch.unsqueeze(emb_all, 0)
    w = ((emb1 - emb2) ** 2).mean(2)
    w = torch.exp(-w / 2)

    topk_values, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
    w = w * mask

    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    y = torch.zeros(y_label.shape[0], num_real_class)
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))
    all_knn_label = torch.argmax(f, 1).cpu().numpy()
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    class_counter = Counter(y_label)
    for i in range(num_real_class):
        class_counter[i] = 0
    for i in range(len(mask_label)):
        if mask_label[i] == 0:
            end_knn_label[i] = y_label[i]
        else:
            class_counter[all_knn_label[i]] += 1

    classwise_acc = torch.zeros((num_real_class,)).to(device)
    for i in range(num_real_class):
        classwise_acc[i] = class_counter[i] / max(class_counter.values()) if max(class_counter.values()) > 0 else 0
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx])))
    cpl_mask[:] = True
    return all_knn_label, end_knn_label, cpl_mask, indices


def construct_graph_via_knn_nearind_gpu(data_embed, y_label, mask_label, device, topk=5, sigma=0.25, alpha=0.99,
                                            p_cutoff=0.90, num_real_class=21):
    eps = np.finfo(float).eps
    n, d = data_embed.shape[0], data_embed.shape[1]
    emb_all = data_embed / (sigma + eps)
    emb1 = torch.unsqueeze(emb_all, 1)
    emb2 = torch.unsqueeze(emb_all, 0)
    w = ((emb1 - emb2) ** 2).mean(2)
    w = torch.exp(-w / 2)

    topk_values, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
    w = w * mask

    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    # 构造初始标签分布
    y = torch.zeros(y_label.shape[0], num_real_class)
    for i in range(len(mask_label)):
        if mask_label[i] == 0:  # 有标签样本
            y[i][int(y_label[i])] = 1
    f = torch.matmul(torch.inverse(torch.eye(n).to(device) - alpha * s + eps), y.to(device))
    end_knn_label = torch.argmax(f, 1).cpu().numpy()

    # 直接基于置信度筛选
    pseudo_label = torch.softmax(f, dim=-1)
    max_probs, _ = torch.max(pseudo_label, dim=-1)
    cpl_mask = max_probs.ge(p_cutoff)  # 只用固定 p_cutoff 筛选

    return end_knn_label, cpl_mask  # 简化返回值



