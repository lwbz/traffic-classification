import datetime
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import queue
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from ts_tfc_ssl.ts_data.dataloader import UCRDataset, IOTDataset
from ts_tfc_ssl.ts_data.preprocessing import normalize_per_series, fill_nan_value
from ts_tfc_ssl.ts_model.loss import sup_contrastive_loss
from ts_tfc_ssl.ts_model.model1111 import ProjectionHead
from ts_tfc_ssl.ts_utils import build_model, set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate, convert_coeff, generate_pseudo_labels, generate_cpl_pseudo_labels, balance_classes, \
    weak_augment, strong_augment

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='Time_Freq', help='encoder backbone, FCN_Time, FCN_Freq, Time_Freq, Transformer_TF')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='IOT21',
                        help='dataset(in ucr)')  # IOT21 TMC21
    parser.add_argument('--dataroot', type=str, default='./IOT_Dataset', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Semi training
    parser.add_argument('--is_pseudo', type=int, default=1, help='0, 1')
    parser.add_argument('--labeled_ratio', type=float, default=0.001, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--pseudo_epochs', type=int, default=30)
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3')
    parser.add_argument('--knn_num_tem', type=int, default=40, help='10, 20, 50')
    parser.add_argument('--knn_num_feq', type=int, default=30, help='10, 20, 50')

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--sup_con_lamda', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--mlp_head', type=bool, default=True, help='head project')
    parser.add_argument('--temperature', type=float, default=50, help='20, 50')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    sum_dataset, sum_target, num_classes = build_dataset(args)  # 特征集 标签 类别数
    args.num_classes = num_classes
    args.seq_len = sum_dataset.shape[1]  # 序列长度

    #  根据数据集的大小动态调整 batch_size，防止因 batch_size 设置过大导致训练无法正常进行。
    #  sum_dataset.shape[0] 是数据集的样本数量
    while sum_dataset.shape[0] * 0.6 < args.batch_size:
        args.batch_size = args.batch_size // 2

    #  如果调整后的 batch_size 的两倍大于数据集中 60% 样本量，说明数据量较小，此时将 args.queue_maxsize 设置为 2。
    if args.batch_size * 2 > sum_dataset.shape[0] * 0.6:
        args.queue_maxsize = 2


    # 调用 build_model(args) 函数，根据 args 参数构建主模型(时域) model 和分类器 classifier
    model, classifier = build_model(args)

    # 构造一个投影头（projection_head），其输入维度为 128，通常用于对比学习（contrastive learning）任务，将模型输出映射到投影空间。
    projection_head = ProjectionHead(input_dim=128)

    # 将主模型、分类器和投影头迁移到指定的设备（如 GPU 或 CPU），以支持加速计算。
    model, classifier = model.to(device), classifier.to(device)
    projection_head = projection_head.to(device)

    # # 调用 build_loss(args) 函数，创建一个损失函数（如交叉熵损失），并迁移到设备。 cross_entropy  |  reconstruction_loss
    loss = build_loss(args).to(device)

    # 使用 state_dict() 保存模型、分类器和投影头的初始参数状态。这对于后续恢复模型参数或初始化新的实例非常有用。
    model_init_state = model.state_dict()
    classifier_init_state = classifier.state_dict()
    projection_head_init_state = projection_head.state_dict()

    # 检查 args.mlp_head 的布尔值，判断是否启用 MLP 投影头（projection_head）
    is_projection_head = args.mlp_head

    # 使用 torch.optim.Adam 优化器，同时优化模型（model）、分类器（classifier）和投影头（projection_head）的参数。
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},
                                  {'params': projection_head.parameters()}],
                                 lr=args.lr)

    # 2025/1/14 start
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hour = int(time_stamp[9:11])
    new_hour = (hour + 8) % 24
    time_stamp = time_stamp[:9] + f"{new_hour:02d}" + time_stamp[11:]
    log_path = os.path.join('./train_log', args.dataset)
    log_path = os.path.join(log_path, args.backbone)
    log_path = os.path.join(log_path, time_stamp)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    state = {k: v for k, v in args._get_kwargs()}

    # 打开文件，准备写入日志
    log_file = open(os.path.join(log_path, 'log.txt'), "a")

    print('Total params: %.2fM\n' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    print('Total params {0:.2f}M\n'.format(sum(p.numel() for p in model.parameters()) / 1000000.0), file=log_file)

    print('=================args=================', file=log_file)
    print('=================args=================')

    i = 1
    for k, v in state.items():
        output = f"{k.center(12)} {str(v).center(30)}"
        output = f"{k.center(12)} {str(v).center(30)}"
        # 打印到控制台
        print(output)
        # 写入日志文件
        print(output, file=log_file)
        i += 1
    print('======================================\n', file=log_file)
    print('======================================\n')
    print('start training on {}'.format(args.dataset), file=log_file)

    # 2025/1/14 end

    print('start training on {}'.format(args.dataset))

    # 调用 get_all_datasets(sum_dataset, sum_target) 函数，将原始数据集（sum_dataset 和 sum_target）划分为：训练集、验证集、测试集
    train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target)

    # losses：一个空列表，用于记录训练过程中的损失值。
    # test_accuracies：一个空列表，用于记录训练过程中测试集上的精度。
    # train_time：初始化训练时间为 0.0，用于累计记录整个训练过程所花费的时间。
    # end_val_epochs：一个空列表，用于记录在验证集上评估的结束轮次。
    losses = []
    val_acc_tem = []  # 2/13
    val_acc_feq = []  # 2/13
    train_loss_tem = []  # 2/20
    train_loss_feq = []  # 2/20
    test_accuracies = []
    train_time = 0.0
    end_val_epochs = []

    # test_accuracies_tem 和 end_val_epochs_tem：记录第一个模型（model）在测试集上的精度以及对应的验证轮次。
    # test_accuracies_feq 和 end_val_epochs_feq：记录第二个模型（model_feq）在测试集上的精度以及对应的验证轮次。
    test_accuracies_tem = []
    end_val_epochs_tem = []
    test_accuracies_feq = []
    end_val_epochs_feq = []

    # 循环的目的是在每一折 (fold) 的训练和评估过程中，对模型和数据进行处理。
    for i, train_dataset in enumerate(train_datasets):
        t = time.time()

        val_loss_fold = []  # 2/13
        val_acc_tem_fold = []  # 2/13
        val_acc_feq_fold = []  # 2/13
        train_loss_tem_fold = []  # 2/20
        train_loss_feq_fold = []  # 2/20

        # 作用：每次进入新的折 (fold) 训练时，将模型、分类器和投影头的状态恢复到初始状态。
        # 目的：确保每折训练和评估的起点一致，不受之前折的训练结果影响。这里涉及两个模型（model 和 model_feq），
        # 分别对应常规输入和频域输入（可能用于频域特征提取）。
        model.load_state_dict(model_init_state)
        classifier.load_state_dict(classifier_init_state)
        projection_head.load_state_dict(projection_head_init_state)

        # print('\n{} fold start training and evaluate'.format(i))
        #
        # # 2025/1/14 start
        # print('\n{} fold start training and evaluate'.format(i), file=log_file)
        # 2025/1/14 end

        # 作用：从每折数据中提取当前折训练、验证和测试集。
        train_target = train_targets[i]

        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        # 调用 build_loss(args) 函数，创建一个损失函数（如交叉熵损失），并迁移到设备。 cross_entropy  |  reconstruction_loss
        # train_target_temp = train_target.astype(np.int64)  # 转换为整数类型
        # class_weights = torch.FloatTensor(1.0 / np.bincount(train_target_temp)).to(device)
        # class_weights = class_weights / class_weights.sum()
        # loss = build_loss(args, class_weights).to(device)

        # 处理数据中的缺失值，将 NaN 值替换为合适的数值。
        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # TODO normalize per series
        # 作用：对每个序列单独归一化，可能是将数据标准化到某个范围（如 [0, 1] 或均值为 0，方差为 1）。
        # 目的：确保训练、验证和测试数据分布一致，避免因特定数据范围差异而影响模型效果。

        train_dataset = normalize_per_series(train_dataset, 1500, 1)
        val_dataset = normalize_per_series(val_dataset, 1500, 1)
        test_dataset = normalize_per_series(test_dataset, 1500, 1)


        # 时域数据转换为张量并增加通道维度
        train_tem = torch.from_numpy(train_dataset).float().unsqueeze(1)  # (n_samples, 1, 50)
        val_tem = torch.from_numpy(val_dataset).float().unsqueeze(1)
        test_tem = torch.from_numpy(test_dataset).float().unsqueeze(1)

        # fft.rfft：
        #   计算训练数据的快速傅里叶变换（FFT），提取频域特征。
        #   dim=-1 指定沿序列长度的维度进行变换。
        train_fft = fft.rfft(torch.from_numpy(train_dataset), dim=-1)  # (n_samples, 26) 复数张量
        train_fft, _ = convert_coeff(train_fft)  # (n_samples, 2, 26)
        val_fft = fft.rfft(torch.from_numpy(val_dataset), dim=-1)
        val_fft, _ = convert_coeff(val_fft)
        test_fft = fft.rfft(torch.from_numpy(test_dataset), dim=-1)
        test_fft, _ = convert_coeff(test_fft)

        # 上采样频域数据到长度 50
        train_freq = F.interpolate(train_fft, size=50, mode='linear', align_corners=False)  # (n_samples, 2, 50)
        val_freq = F.interpolate(val_fft, size=50, mode='linear', align_corners=False)
        test_freq = F.interpolate(test_fft, size=50, mode='linear', align_corners=False)

        # 划分有标签和未标记数据（例如 20% 有标签，80% 未标记）
        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(train_dataset)),
            test_size=1 - args.labeled_ratio,  # 未标记比例
            stratify=train_target  # 按类别分层采样，确保分布一致
        )

        unlabeled_time = train_tem[unlabeled_indices]
        unlabeled_freq = train_freq[unlabeled_indices]
        unlabeled_set = IOTDataset(unlabeled_time.to(device), unlabeled_freq.to(device), is_labeled=False)  # 无标签

        train_tem = train_tem[labeled_indices]
        train_freq = train_freq[labeled_indices]
        train_target = train_target[labeled_indices]

        # # 确保 train_tem 是 3 维
        # if train_tem.dim() == 2:  # (n, 50) -> (n, 1, 50)
        #     train_tem = train_tem.unsqueeze(1)
        #
        # # 平衡类别
        # balanced_time, balanced_freq, balanced_labels = balance_classes(
        #     train_tem, train_freq, train_target, target_count=2500, noise_factor=0.01
        # )

        # # 创建有标签数据集
        # train_set = IOTDataset(
        #     balanced_time.to(device),
        #     balanced_freq.to(device),
        #     torch.from_numpy(balanced_labels).to(device).to(torch.int64),
        #     is_labeled=True
        # )

        # 创建数据集
        # 数据集和加载器
        train_set = IOTDataset(train_tem.to(device), train_freq.to(device),
                               torch.from_numpy(train_target).to(device).to(torch.int64), is_labeled=True)

        val_set = IOTDataset(val_tem.to(device), val_freq.to(device),
                             torch.from_numpy(val_target).to(device).to(torch.int64))
        test_set = IOTDataset(test_tem.to(device), test_freq.to(device),
                              torch.from_numpy(test_target).to(device).to(torch.int64))

        print(f"Labeled samples: {len(train_set)}, Unlabeled samples: {len(unlabeled_set)}")
        print(f"Labeled samples: {len(train_set)}, Unlabeled samples: {len(unlabeled_set)}", file=log_file)

        # 打印分布
        label_dist = Counter(train_target)
        total_labeled = len(train_target)
        label_percent = {k: v / total_labeled * 100 for k, v in label_dist.items()}
        for cls in sorted(label_dist.keys()):
            print(f"Class {int(cls)}: {label_dist[cls]} samples, {label_percent[cls]:.2f}%")

        # # 获取标签分布
        # labels = train_set.target.cpu().numpy()  # 转换为 numpy 数组
        # label_dist = Counter(labels)
        # total_labeled = len(labels)
        #
        # # 计算占比
        # label_percent = {k: v / total_labeled * 100 for k, v in label_dist.items()}
        #
        # # 打印分布
        # print("Labeled set distribution (Count and %):")
        # for cls in sorted(label_dist.keys()):
        #     print(f"Class {cls}: {label_dist[cls]} samples, {label_percent[cls]:.2f}%")




        # 调整逻辑：
        #   如果有标签数据的数量小于当前批量大小，则批量大小减半。
        #   保证有标签数据的批量大小不低于 16，避免批次过小导致训练不稳定。
        # 目的：
        #   动态调整批量大小以适应数据规模，避免因为数据过少导致批次不可用。
        batch_size_labeled = 128
        while train_dataset.shape[0] * args.labeled_ratio < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if train_dataset.shape[0] < 16:
            batch_size_labeled = 16

        # 构建数据加载器
        # DataLoader 参数：
        #   train_set_labled 和 train_set：分别为有标签和完整训练数据集。
        #   batch_size：动态调整的批量大小（针对有标签数据）或固定的 args.batch_size。
        #   num_workers=0：数据加载的线程数为 0，即主线程加载数据，适用于小规模数据集。
        #   drop_last=False：不舍弃最后一个批次，即使它可能小于指定批量大小。

        train_loader = DataLoader(train_set, batch_size=batch_size_labeled, num_workers=0,
                                          drop_last=False, shuffle=True)

        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        # train_loader = DataLoader(train_set, batch_size=batch_size_labeled, num_workers=0,
        #                           drop_last=False, sampler=RandomSampler(train_set, num_samples=int(0.2 * len(train_set))))
        #
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0, sampler=RandomSampler(val_set, num_samples=int(0.2 * len(val_set))))
        # test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0, sampler=RandomSampler(test_set, num_samples=int(0.2 * len(test_set))))

        # 数据加载器
        # labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, num_workers=0, shuffle=True)
        labeled_loader = DataLoader(train_set, batch_size=batch_size_labeled, num_workers=0, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size_labeled, num_workers=0)

        # 损失和准确率记录：
        #   train_loss 和 train_accuracy：用于记录时间域模型的训练损失和准确率。
        #   train_loss_feq 和 train_accuracy_feq：用于记录频域模型的训练损失和准确率。
        # train_loss = []
        # train_accuracy = []
        # train_loss_feq = []
        # train_accuracy_feq = []

        # 训练步数：
        #   num_steps：通过 epoch 和 batch_size 计算每轮训练的步数（可能需要检查其定义）。
        num_steps = args.epoch // args.batch_size

        # 早停相关变量：
        #   last_loss：初始为正无穷，用于记录上一轮的验证损失。
        #   stop_count：记录连续验证损失不下降的轮数。
        #   increase_count：可能用于记录损失增加的次数。
        last_loss = float('inf')
        stop_count = 0
        increase_count = 0

        # 使用 train_set.__len__() 获取训练数据集的样本数量。
        # 通过整除批量大小 (args.batch_size) 计算每轮训练的步数。
        # 如果数据集样本数量过少，导致 num_steps 计算结果为 0，则将其调整为 1，确保至少有一个训练步数。
        num_steps = train_set.__len__() // args.batch_size
        if num_steps == 0:
            num_steps = num_steps + 1

        # 变量含义：
        #   min_val_loss：记录当前最低的验证损失，用于早停判断。
        #   test_accuracy：记录最佳验证模型对应的测试集准确率。
        #   end_val_epoch：记录最低验证损失对应的训练轮次。
        #   min_val_loss_feq、test_accuracy_feq、end_val_epoch_feq：分别用于记录频域模型的最优验证损失、测试准确率和对应轮次。
        #   test_accuracy_tem、end_val_epoch_tem：可能用于时间域模型的临时结果保存。
        # 作用：
        #   提供模型性能监控和早停的基础。
        min_val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0
        min_val_loss_feq = float('inf')
        test_accuracy_tem = 0
        end_val_epoch_tem = 0
        test_accuracy_feq = 0
        end_val_epoch_feq = 0

        for epoch in range(1, args.warmup_epochs + 1):
            # 目的：通过 stop_count 和 increase_count 实现早停机制，避免模型过拟合或在性能不再提高时浪费计算资源。
            # 条件：
            #   stop_count == 80：表示验证集损失连续80次未改善。
            #   increase_count == 80：可能用于其他条件（如准确率下降或某种指标增加太多次）。
            # 效果：满足任一条件时，提前退出训练循环，并打印提示信息。
            if stop_count == 60 or increase_count == 60:
                print('model convergent at epoch {0}, early stopping. stop_count = {1}, increase_count = {2}'.format(
                    epoch, stop_count, increase_count))

                # 2025/1/14 start
                print('model convergent at epoch {0}, early stopping. stop_count = {1}, increase_count = {2}'.format(
                    epoch, stop_count, increase_count), file=log_file)
                # 2025/1/14 end

                break

            # epoch_train_loss：记录当前轮次的总训练损失。
            # epoch_train_acc：记录当前轮次的训练准确率。
            # num_iterations：记录训练过程中处理的数据批次数量（可能用于后续计算均值或动态调整）。
            epoch_train_loss = 0
            epoch_train_acc = 0
            num_iterations = 0

            epoch_train_loss_tem = 0  # 2/20
            epoch_train_loss_feq = 0  # 2/20

            # 作用：将所有模型及其组件设置为训练模式（train()），启用特定于训练的功能，如 Dropout 和 Batch Normalization。
            # 涉及模块：
            #   model：主模型，负责特征提取。
            #   classifier：主分类器，负责将特征映射到类标签。
            #   projection_head：投影头，可能用于对比学习或额外的辅助任务。
            #   model_feq / classifier_feq / projection_head_feq：针对频域数据的模型、分类器和投影头。
            model.train()
            classifier.train()
            projection_head.train()

            # 作用：检查当前轮次是否小于设定的 "warmup" 阈值，执行特殊的训练逻辑（如更小的学习率或专门的损失函数设计）。
            # 背景：Warmup 阶段通常用来：
            #   稳定模型参数的更新。
            #   避免刚开始训练时大梯度导致的不稳定。
            if epoch < args.warmup_epochs:
                for time_data, freq_data, y, _ in train_loader:  # train_labeled_loader：仅包含带标签的数据，专用于有监督训练。

                    optimizer.zero_grad()

                    # pred_embed 是时域数据的嵌入表示。
                    # 若启用了投影头（is_projection_head 为真），则通过 projection_head 得到投影后的嵌入。
                    pred_embed = model(time_data, freq_data)

                    # print(x.shape)
                    # print(x_feq.shape)
                    # if is_projection_head:
                    #     preject_head_embed = projection_head(pred_embed)

                    # 目的：分别计算时域模型和频域模型的分类损失。
                    # classifier 和 classifier_feq：将嵌入映射到最终类别标签，得到预测值。
                    # loss 和 loss_feq：用于计算预测值和真实标签 y 的分类损失。
                    # pred = classifier(pred_embed)
                    pred = pred_embed

                    step_loss = loss(pred, y)

                    epoch_train_loss_tem += step_loss.item()  # 2/20

                    # 通过 step_loss.backward() 计算梯度。
                    # 使用 optimizer.step() 更新模型参数。
                    step_loss.backward()

                    optimizer.step()

                    # 每处理一个 batch，增加迭代计数器。
                    num_iterations = num_iterations + 1

                train_loss_tem_fold.append(epoch_train_loss_tem / num_iterations)  # 2/20
                train_loss_feq_fold.append(epoch_train_loss_feq / num_iterations)  # 2/20

            # 模型进入评估模式
            # 评估模式：
            #   将模型和相关组件（classifier 和 projection_head）切换到评估模式（eval()）。
            #   禁用 dropout 和 BatchNorm 的运行时行为，以确保验证阶段的一致性。
            model.eval()
            classifier.eval()
            projection_head.eval()

            # 验证集评估
            # 功能：
            #   使用 evaluate 函数在验证集上计算损失（val_loss）和准确率（val_accu_tem）。
            #   evaluate 应是一个接受数据加载器、模型、分类器、损失函数和设备的函数，返回模型的性能指标。
            val_loss, val_accu_tem, _, _, _ = evaluate(val_loader, model, classifier, loss, device, 2)

            val_loss_fold.append(val_loss)  # 2/13

            # 记录最佳验证损失和测试集性能
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch
                # test_loss, test_accuracy_tem, test_recall, test_f1 = evaluate(test_loader, model, classifier, loss, device)
                test_loss, test_accuracy_tem, test_recall, test_f1, conf_matrix = evaluate(test_loader, model, classifier, loss, device, 2)

            test_accuracy = test_accuracy_tem

            val_acc_tem_fold.append(val_accu_tem.item())  # 2/13

            # 实现早停机制
            # 早停条件：
            #   停止计数（stop_count）：
            #     如果验证损失的变化幅度小于 1e-4 且当前 epoch 已超过 warmup 阶段，增加停止计数器。
            #     否则重置计数器。
            #   增加计数（increase_count）：
            #     如果验证损失在 warmup 阶段后增加，增加计数器。
            #     否则重置计数器。
            if (epoch > args.warmup_epochs) and (abs(last_loss - val_loss) <= 1e-4):
                stop_count += 1
            else:
                stop_count = 0

            if (epoch > args.warmup_epochs) and (val_loss > last_loss):
                increase_count += 1
            else:
                increase_count = 0

            last_loss = val_loss

            if epoch % 1 == 0:

                print("\nInitial epoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))
                print(
                    "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                    test_recall[1],
                                                                                                    test_recall[2]))
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],
                                                                                                      test_f1[1],
                                                                                                      test_f1[2]))
                print("\nInitial epoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy), file=log_file)

                print(
                    "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                    test_recall[1],
                                                                                                    test_recall[2]),
                    file=log_file)
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],
                                                                                                      test_f1[1],
                                                                                                      test_f1[2]),
                      file=log_file)

        if args.is_pseudo == 1:  #  使用伪标签
            # 生成伪标签
            # 模型进入评估模式
            # model.eval()
            # classifier.eval()
            # projection_head.eval()

            # pseudo_labels = []
            # pseudo_time_list = []
            # pseudo_freq_list = []
            # confidence_threshold = 0.90
            #
            # with torch.no_grad():
            #     for time_data, freq_data, _ in unlabeled_loader:
            #         output = model(time_data, freq_data)
            #         probs = F.softmax(output, dim=1)
            #         max_probs, preds = torch.max(probs, dim=1)
            #         mask = max_probs > confidence_threshold
            #         pseudo_labels.append(preds[mask].cpu())
            #         pseudo_time_list.append(time_data[mask].cpu())
            #         pseudo_freq_list.append(freq_data[mask].cpu())
            #
            # pseudo_labels = torch.cat(pseudo_labels)
            # pseudo_time = torch.cat(pseudo_time_list)
            # pseudo_freq = torch.cat(pseudo_freq_list)
            # pseudo_set = IOTDataset(pseudo_time.to(device), pseudo_freq.to(device),
            #                         pseudo_labels.to(device).to(torch.int64))

            # 混合训练（带伪标签更新）
            # pseudo_labels, pseudo_time, pseudo_freq = generate_pseudo_labels(model, unlabeled_loader, device, 0.97)
            # pseudo_set = IOTDataset(pseudo_time.to(device), pseudo_freq.to(device), pseudo_labels.to(device).to(torch.int64))
            # combined_set = ConcatDataset([train_set, pseudo_set])
            # combined_loader = DataLoader(combined_set, batch_size=batch_size_labeled, num_workers=0)
            #
            # print(f"\nPseudo-labeled samples: {len(pseudo_set)}")
            # print(f"\nPseudo-labeled samples: {len(pseudo_set)}", file=log_file)

            # 混合训练
            # combined_set = ConcatDataset([train_set, pseudo_set])
            # combined_loader = DataLoader(combined_set, batch_size=batch_size_labeled, num_workers=0, shuffle=True)

            pseudo_weight = 0.99
            max_pseudo = len(train_set)

            pseudo_labels, pseudo_time, pseudo_freq = generate_cpl_pseudo_labels(model, unlabeled_loader, device,
                                                                                 args.num_classes,
                                                                                 max_pseudo=max_pseudo)
            # pseudo_labels, pseudo_time, pseudo_freq = generate_pseudo_labels(model, unlabeled_loader, device, 0.99)

            pseudo_set = IOTDataset(pseudo_time.to(device), pseudo_freq.to(device),
                                    pseudo_labels.to(device).to(torch.int64), is_labeled=False)
            combined_set = ConcatDataset([train_set, pseudo_set])
            combined_loader = DataLoader(combined_set, batch_size=batch_size_labeled, shuffle=True)

            print(f"\nPseudo samples: {len(pseudo_set)}")
            print(f"\nPseudo samples: {len(pseudo_set)}", file=log_file)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)  # 降低 lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

            for epoch in range(1, args.pseudo_epochs + 1):
                if epoch % 3 == 0:  # 每 3 次更新伪标签
                    max_pseudo = max_pseudo + int(0.5 * max_pseudo)
                    pseudo_labels, pseudo_time, pseudo_freq = generate_cpl_pseudo_labels(model, unlabeled_loader, device, args.num_classes, max_pseudo=max_pseudo)
                    # pseudo_labels, pseudo_time, pseudo_freq = generate_pseudo_labels(model, unlabeled_loader, device, 0.99)
                    pseudo_set = IOTDataset(pseudo_time.to(device), pseudo_freq.to(device), pseudo_labels.to(device).to(torch.int64), is_labeled=False)
                    combined_set = ConcatDataset([train_set, pseudo_set])
                    combined_loader = DataLoader(combined_set, batch_size=batch_size_labeled, shuffle=True)

                    print(f"\nPseudo samples: {len(pseudo_set)}")
                    print(f"\nPseudo samples: {len(pseudo_set)}", file=log_file)

                    #  有标签数据占比
                    labeled_dist = Counter(train_target)
                    total_labeled = sum(labeled_dist.values())
                    labeled_percent = {int(k): f"{v / total_labeled * 100:.2f}%" for k, v in labeled_dist.items()}

                    #  伪标签数据占比
                    pseudo_dist = Counter(pseudo_labels.numpy())
                    total_pseudo = sum(pseudo_dist.values())
                    pseudo_percent = {k: f"{v / total_pseudo * 100:.2f}%" for k, v in pseudo_dist.items()}

                    # 按 key 排序
                    labeled_percent_sorted = dict(sorted(labeled_percent.items()))
                    pseudo_percent_sorted = dict(sorted(pseudo_percent.items()))

                    # 直接打印
                    print("Labeled distribution (%):", labeled_percent_sorted)
                    print("Pseudo distribution  (%):", pseudo_percent_sorted)

                model.train()
                for time_data, freq_data, target, is_labeled in combined_loader:
                    # time_data, freq_data, target = time_data.to(device), freq_data.to(device), target.to(device)
                    is_labeled = is_labeled.to(device)
                    optimizer.zero_grad()
                    output = model(time_data, freq_data)

                    # # 分割批次：前半部分是有标签数据，后半部分是伪标签数据
                    # batch_size = time_data.size(0)
                    # labeled_size = len(train_set) // (len(train_set) + len(pseudo_set)) * batch_size
                    # labeled_output = output[:labeled_size]
                    # pseudo_output = output[labeled_size:]
                    # labeled_target = target[:labeled_size]
                    # pseudo_target = target[labeled_size:]
                    #
                    # # 计算加权损失
                    # labeled_loss = loss(labeled_output, labeled_target)
                    # pseudo_loss = loss(pseudo_output, pseudo_target)
                    # step_loss = labeled_loss + pseudo_weight * pseudo_loss

                    # 根据 is_labeled 分割损失
                    labeled_mask = is_labeled
                    pseudo_mask = ~is_labeled
                    labeled_output = output[labeled_mask]
                    pseudo_output = output[pseudo_mask]
                    labeled_target = target[labeled_mask]
                    pseudo_target = target[pseudo_mask]

                    labeled_loss = loss(labeled_output, labeled_target) if labeled_output.size(0) > 0 else 0
                    pseudo_loss = loss(pseudo_output, pseudo_target) if pseudo_output.size(0) > 0 else 0
                    step_loss = labeled_loss + pseudo_weight * pseudo_loss

                    # step_loss = loss(output, target)
                    step_loss.backward()
                    optimizer.step()
                # print(f"Pseudo Epoch {epoch}, Loss: {step_loss.item()}")
                # print(f"Pseudo Epoch {epoch}, Loss: {step_loss.item()}", file=log_file)

                # 模型进入评估模式
                # 评估模式：
                #   将模型和相关组件（classifier 和 projection_head）切换到评估模式（eval()）。
                #   禁用 dropout 和 BatchNorm 的运行时行为，以确保验证阶段的一致性。
                model.eval()
                classifier.eval()
                projection_head.eval()

                # 验证集评估
                # 功能：
                #   使用 evaluate 函数在验证集上计算损失（val_loss）和准确率（val_accu_tem）。
                #   evaluate 应是一个接受数据加载器、模型、分类器、损失函数和设备的函数，返回模型的性能指标。
                val_loss, val_accu_tem, _, _, _ = evaluate(val_loader, model, classifier, loss, device, 2)

                val_loss_fold.append(val_loss)  # 2/13

                # 记录最佳验证损失和测试集性能
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    end_val_epoch = epoch
                    # test_loss, test_accuracy_tem, test_recall, test_f1 = evaluate(test_loader, model, classifier, loss, device)
                    test_loss, test_accuracy_tem, test_recall, test_f1, conf_matrix = evaluate(test_loader, model,
                                                                                               classifier, loss, device,
                                                                                               2)

                test_accuracy = test_accuracy_tem

                val_acc_tem_fold.append(val_accu_tem.item())  # 2/13

                # 实现早停机制
                # 早停条件：
                #   停止计数（stop_count）：
                #     如果验证损失的变化幅度小于 1e-4 且当前 epoch 已超过 warmup 阶段，增加停止计数器。
                #     否则重置计数器。
                #   增加计数（increase_count）：
                #     如果验证损失在 warmup 阶段后增加，增加计数器。
                #     否则重置计数器。
                if (epoch > args.warmup_epochs) and (abs(last_loss - val_loss) <= 1e-4):
                    stop_count += 1
                else:
                    stop_count = 0

                if (epoch > args.warmup_epochs) and (val_loss > last_loss):
                    increase_count += 1
                else:
                    increase_count = 0

                last_loss = val_loss

                if epoch % 1 == 0:
                    print("\nPseudo epoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))
                    print(
                        "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                        test_recall[1],
                                                                                                        test_recall[2]))
                    print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],
                                                                                                          test_f1[1],
                                                                                                          test_f1[2]))
                    print("\nepoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy), file=log_file)

                    print(
                        "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                        test_recall[1],
                                                                                                        test_recall[2]),
                        file=log_file)
                    print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],
                                                                                                          test_f1[1],
                                                                                                          test_f1[2]),
                          file=log_file)


        losses.append(val_loss_fold)  # 2/13
        val_acc_tem.append(val_acc_tem_fold)  # 2/13
        val_acc_feq.append(val_acc_feq_fold)  # 2/13
        train_loss_tem.append(train_loss_tem_fold)  # 2/20
        train_loss_feq.append(train_loss_feq_fold)  # 2/20

        # 保存性能指标
        # 功能：
        #   记录每个 fold 的测试集性能和最佳验证 epoch：
        #     最终测试准确率：
        #       test_accuracies: 综合时域与频域选择后的测试准确率。
        #       test_accuracies_tem: 时域模型的测试准确率。
        #       test_accuracies_feq: 频域模型的测试准确率。
        #     最佳验证 epoch：
        #       end_val_epochs: 综合模型的最佳验证 epoch。
        #       end_val_epochs_tem: 时域模型的最佳验证 epoch。
        #       end_val_epochs_feq: 频域模型的最佳验证 epoch。
        # 用途：
        #   将每个 fold 的关键性能指标存入对应列表，便于后续汇总分析（例如平均测试准确率）。
        test_accuracies.append(test_accuracy)
        end_val_epochs.append(end_val_epoch)
        test_accuracies_tem.append(test_accuracy_tem)
        end_val_epochs_tem.append(end_val_epoch)
        test_accuracies_feq.append(test_accuracy_feq)
        end_val_epochs_feq.append(end_val_epoch_feq)

        # 记录训练时间
        # 逻辑：
        #   计算当前 fold 的训练耗时：
        #     t 是每个 fold 的训练时间。
        #     train_time 累积所有 fold 的总训练时间。
        # 用途：
        #   通过训练时间了解模型的运行效率。
        #   结合性能指标，评估训练成本与模型性能之间的平衡。
        t = time.time() - t
        train_time += t

        # print('{} fold finish training'.format(i))
        #
        # # 2025/1/14 start
        # print('{} fold finish training'.format(i), file=log_file)
        # 2025/1/14 end

    # 转换性能指标格式
    # 逻辑：
    #   将存储的性能指标列表（如 test_accuracies、end_val_epochs 等）转换为更适合数值操作的格式：
    #     torch.Tensor：
    #       用于存储测试准确率（test_accuracies 等），便于后续使用 PyTorch 操作（如 torch.mean 计算平均值）。
    #     np.array：
    #       用于存储结束的验证 epoch 数组（end_val_epochs 等），便于后续统计和分析。
    # 用途：
    #   格式化后的数据可直接用于数学计算和后续绘图分析。
    test_accuracies = torch.Tensor(test_accuracies)
    end_val_epochs = np.array(end_val_epochs)

    test_accuracies_tem = torch.Tensor(test_accuracies_tem)
    end_val_epochs_tem = np.array(end_val_epochs_tem)

    test_accuracies_feq = torch.Tensor(test_accuracies_feq)
    end_val_epochs_feq = np.array(end_val_epochs_feq)

    print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed)

    print("test_acc     =", round(torch.mean(test_accuracies).item(), 3))
    print(
        "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0], test_recall[1],
                                                                                        test_recall[2]))
    print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
                                                                                          test_f1[2]))
    print('\nDone!')

    print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed,
          file=log_file)

    print("test_acc     =", round(torch.mean(test_accuracies).item(), 3), file=log_file)
    print(
        "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0], test_recall[1],
                                                                                        test_recall[2]), file=log_file)
    print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
                                                                                          test_f1[2]), file=log_file)
    print('\nDone!', file=log_file)

    # 关闭日志文件
    log_file.close()
    # 2025/1/14 end

    with open(os.path.join(log_path, 'val_loss.txt'), 'w') as file:
        for sublist in losses:
            file.write(','.join(map(str, sublist)) + '\n\n')  # 逗号分隔每个元素，并换行

    with open(os.path.join(log_path, 'val_acc_tem.txt'), 'w') as file:
        for sublist in val_acc_tem:
            file.write(','.join(map(str, sublist)) + '\n\n')  # 逗号分隔每个元素，并换行

    with open(os.path.join(log_path, 'val_acc_feq.txt'), 'w') as file:
        for sublist in val_acc_feq:
            file.write(','.join(map(str, sublist)) + '\n\n')  # 逗号分隔每个元素，并换行

    with open(os.path.join(log_path, 'train_loss_tem.txt'), 'w') as file:
        for sublist in train_loss_tem:
            file.write(','.join(map(str, sublist)) + '\n\n')  # 逗号分隔每个元素，并换行

    with open(os.path.join(log_path, 'train_loss_feq.txt'), 'w') as file:
        for sublist in train_loss_feq:
            file.write(','.join(map(str, sublist)) + '\n\n')  # 逗号分隔每个元素，并换行

    with open(os.path.join(log_path, 'confusion_matrix.txt'), 'w') as file:
        for row in conf_matrix:
            file.write("\t".join(map(str, row)) + "\n")

    acc = round(test_accuracies.item(), 3)
    acc = str(acc)
    acc = acc.replace('.', '_')
    old_name = log_path
    new_name = log_path + "_Done_" + acc
    if args.is_pseudo == 1:
        new_name = new_name + "_pseudo_" + str(args.labeled_ratio)
    os.rename(old_name, new_name)
