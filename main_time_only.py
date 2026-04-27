import datetime
import os
import sys
from collections import Counter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from queue import Queue
import time

import numpy as np
import torch
import torch.fft as fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from ts_tfc_ssl.ts_data.dataloader import UCRDataset
from ts_tfc_ssl.ts_data.preprocessing import normalize_per_series, fill_nan_value
from ts_tfc_ssl.ts_model.loss import sup_contrastive_loss
from ts_tfc_ssl.ts_model.model1111 import ProjectionHead
from ts_tfc_ssl.ts_utils import build_model, set_seed, build_dataset, get_all_datasets, \
    construct_graph_via_knn_cpl_nearind_gpu, \
    build_loss, shuffler, evaluate, convert_coeff

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='FCN_Time', help='MambaModel, LSTM, encoder backbone, FCN_Time, FCN_Freq, Time_Freq')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='IOT17',
                        help='dataset(in ucr)')  # IOT21 TMC21 TMC13 IOT17
    parser.add_argument('--dataroot', type=str, default='./IOT_Dataset', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--log_path', type=str, default='./train_log')

    # Semi training
    parser.add_argument('--is_sl', type=int, default=1, help='0, 1')
    parser.add_argument('--is_pseudo', type=int, default=0, help='0, 1')
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs using only labeled data for ssl')
    parser.add_argument('--pseudo_epochs', type=int, default=30)
    parser.add_argument('--queue_maxsize', type=int, default=3, help='2 or 3')
    parser.add_argument('--knn_num_tem', type=int, default=50, help='10, 20, 50')
    parser.add_argument('--sigma', type=float, default=0.5, help='0.25, 0.5')
    parser.add_argument('--alpha', type=float, default=0.95, help='0.95, 0.99')
    parser.add_argument('--p_cutoff', type=float, default=0.9, help='0.85, 0.9, 0.95')
    parser.add_argument('--pseudo_weight', type=float, default=0.50)

    # Contrastive loss
    parser.add_argument('--sup_con_mu', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--sup_con_lamda', type=float, default=0.05, help='0.05 or 0.005')
    parser.add_argument('--mlp_head', type=bool, default=True, help='head project')
    parser.add_argument('--temperature', type=float, default=20, help='20, 50')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='')
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

    class_weights = torch.ones(args.num_classes, device=device)
    # 调用 build_loss(args) 函数，创建一个损失函数（如交叉熵损失），并迁移到设备。 cross_entropy  |  reconstruction_loss
    loss = build_loss(args, class_weights).to(device)

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
    log_path = os.path.join(args.log_path, args.dataset)
    log_path = os.path.join(log_path, args.backbone)
    log_path = os.path.join(log_path, time_stamp)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    state = {k: v for k, v in args._get_kwargs()}

    # 打开文件，准备写入日志
    log_file = open(os.path.join(log_path, 'log.txt'), "a")

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
        sum_dataset, sum_target, args)

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

    best_acc = 0.0
    best_pre = 0.0
    best_rc = 0.0
    best_f1 = 0.0
    # 循环的目的是在每一折 (fold) 的训练和评估过程中，对模型和数据进行处理。
    for i, train_dataset in enumerate(train_datasets):
        t = time.time()

        val_loss_fold = []  # 2/13
        val_acc_tem_fold = []  # 2/13
        val_acc_feq_fold = []  # 2/13
        train_loss_tem_fold = []   # 2/20
        train_loss_feq_fold = []   # 2/20

        # 作用：每次进入新的折 (fold) 训练时，将模型、分类器和投影头的状态恢复到初始状态。
        # 目的：确保每折训练和评估的起点一致，不受之前折的训练结果影响。这里涉及两个模型（model 和 model_feq），
        # 分别对应常规输入和频域输入（可能用于频域特征提取）。
        model.load_state_dict(model_init_state)
        classifier.load_state_dict(classifier_init_state)
        projection_head.load_state_dict(projection_head_init_state)

        # 作用：从每折数据中提取当前折训练、验证和测试集。
        train_target = train_targets[i]

        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        # X_train, _, y_train, _ = train_test_split(train_dataset, train_target, test_size=1 - args.labeled_ratio, random_state=args.random_seed,
        #                                           stratify=train_target)
        #
        # # Initialize and train the Random Forest model
        # rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
        # rf_model.fit(X_train, y_train)
        #
        #
        # def evaluate_model(y_true, y_pred, dataset_name):
        #     accuracy = accuracy_score(y_true, y_pred)
        #     precision = precision_score(y_true, y_pred, average='macro')
        #     recall = recall_score(y_true, y_pred, average='macro')
        #     f1 = f1_score(y_true, y_pred, average='macro')
        #     print(f"{dataset_name} Metrics:")
        #     print(f"Accuracy: {accuracy:.4f}")
        #     print(f"Precision (macro): {precision:.4f}")
        #     print(f"Recall (macro): {recall:.4f}")
        #     print(f"F1 Score (macro): {f1:.4f}\n")
        #
        #
        # # Evaluate on validation set
        # y_val_pred = rf_model.predict(val_dataset)
        # evaluate_model(val_target, y_val_pred, "Validation")
        #
        # # Evaluate on test set
        # y_test_pred = rf_model.predict(test_dataset)
        # evaluate_model(test_target, y_test_pred, "Test")
        #
        # exit(0)





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

        # 划分有标签和未标记数据（例如 20% 有标签，80% 未标记）
        labeled_indices, unlabeled_indices = train_test_split(
                np.arange(len(train_dataset)),
                test_size=1 - args.labeled_ratio,  # 未标记比例
                stratify=train_target,  # 按类别分层采样，确保分布一致
                random_state=args.random_seed
            )

        train_tem_labeled = train_tem[labeled_indices]
        train_target_labeled = train_target[labeled_indices]

        mask_train = np.ones(len(train_tem), dtype=int)
        mask_train[labeled_indices] = 0
        y_train_all = torch.from_numpy(train_target).clone()
        y_train_all[mask_train == 1] = -1  # 未标记数据设为 -1

        true_labels = torch.from_numpy(train_target).clone()  # 验证伪标签是否正确

        # 同步打乱
        indices = np.random.RandomState(args.random_seed).permutation(len(train_tem))  # 随机排列索引
        train_tem = train_tem[indices]
        y_train_all = y_train_all[indices]
        mask_train = mask_train[indices]

        true_labels = true_labels[indices]

        train_set = UCRDataset(train_tem.to(device),
                                   y_train_all.to(device).to(torch.int64))  # 包含所有数据

        labeled_set = UCRDataset(train_tem_labeled.to(device),
                                     torch.from_numpy(train_target_labeled).to(device).to(torch.int64))

        val_set = UCRDataset(val_tem.to(device),
                                 torch.from_numpy(val_target).to(device).to(torch.int64))
        test_set = UCRDataset(test_tem.to(device),
                                  torch.from_numpy(test_target).to(device).to(torch.int64))

        print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")
        print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}",
                  file=log_file)

        batch_size_labeled = 128
        while train_dataset.shape[0] * args.labeled_ratio < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if train_dataset.shape[0] < 16:
            batch_size_labeled = 16

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0,
                                      drop_last=True, shuffle=False)

        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

        # 数据加载器
        train_labeled_loader = DataLoader(labeled_set, batch_size=batch_size_labeled, num_workers=0, drop_last=True,
                                              shuffle=True)


        # if args.is_pseudo == 0:
        #     labeled_indices_train, unlabeled_indices_train = train_test_split(
        #         np.arange(len(train_dataset)),
        #         test_size=1 - args.labeled_ratio,  # 未标记比例
        #         stratify=train_target,  # 按类别分层采样，确保分布一致
        #         random_state=args.random_seed
        #     )
        #
        #     train_dataset = train_dataset[labeled_indices_train]
        #     train_target = train_target[labeled_indices_train]
        #     labeled_indices, unlabeled_indices = train_test_split(
        #         np.arange(len(val_dataset)),
        #         test_size=1 - args.labeled_ratio,  # 未标记比例
        #         stratify=val_target,  # 按类别分层采样，确保分布一致
        #         random_state=args.random_seed
        #     )
        #
        #     val_dataset = val_dataset[labeled_indices]
        #     val_target = val_target[labeled_indices]
        #
        #     labeled_indices, unlabeled_indices = train_test_split(
        #         np.arange(len(test_dataset)),
        #         test_size=1 - args.labeled_ratio,  # 未标记比例
        #         stratify=test_target,  # 按类别分层采样，确保分布一致
        #         random_state=args.random_seed
        #     )
        #
        #     test_dataset = test_dataset[labeled_indices]
        #     test_target = test_target[labeled_indices]
        #
        #     # UCRDataset：
        #     #   自定义的数据集类，封装了数据和标签。
        #     # 数据集构造：
        #     # train_set_labled：仅包含有标签数据，用于有监督学习部分。
        #     train_set_labled = UCRDataset(torch.from_numpy(train_dataset).to(device),
        #                                   torch.from_numpy(train_target).to(device).to(torch.int64))
        #
        #     # val_set 和 test_set：将验证集和测试集的数据和标签转换为张量格式。
        #     val_set = UCRDataset(torch.from_numpy(val_dataset).to(device),
        #                          torch.from_numpy(val_target).to(device).to(torch.int64))
        #     test_set = UCRDataset(torch.from_numpy(test_dataset).to(device),
        #                           torch.from_numpy(test_target).to(device).to(torch.int64))
        #
        #     print(f"Labeled samples: {len(labeled_indices_train)}, Unlabeled samples: {len(unlabeled_indices_train)}")
        #     print(f"Labeled samples: {len(labeled_indices_train)}, Unlabeled samples: {len(unlabeled_indices_train)}",
        #           file=log_file)
        #
        #     batch_size_labeled = 128
        #     while train_dataset.shape[0] < batch_size_labeled:
        #         batch_size_labeled = batch_size_labeled // 2
        #
        #     if train_dataset.shape[0] < 16:
        #         batch_size_labeled = 16
        #
        #     # 构建数据加载器
        #     train_labeled_loader = DataLoader(train_set_labled, batch_size=batch_size_labeled, num_workers=0,
        #                                       drop_last=False)
        #
        #     val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
        #     test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)


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

        if args.is_sl == 1:
            for epoch in range(1, args.warmup_epochs + 1):
                num_iterations = 0

                epoch_train_loss_tem = 0  # 2/20
                epoch_train_loss_feq = 0  # 2/20

                model.train()
                classifier.train()
                projection_head.train()

                total_loss = 0.0
                if epoch <= args.warmup_epochs:
                    for x, y in train_labeled_loader:  # train_labeled_loader：仅包含带标签的数据，专用于有监督训练。
                        if x.shape[0] < 2:
                            continue

                        optimizer.zero_grad()

                        # pred_embed 是时域数据的嵌入表示。
                        # 若启用了投影头（is_projection_head 为真），则通过 projection_head 得到投影后的嵌入。
                        pred_embed = model(x)

                        if is_projection_head:
                            preject_head_embed = projection_head(pred_embed)

                        # 目的：分别计算时域模型和频域模型的分类损失。
                        # classifier 和 classifier_feq：将嵌入映射到最终类别标签，得到预测值。
                        # loss 和 loss_feq：用于计算预测值和真实标签 y 的分类损失。

                        pred = classifier(pred_embed)

                        step_loss = loss(pred, y)

                        # 累积损失
                        total_loss += step_loss.item()
                        num_iterations += 1
                        avg_loss = total_loss / num_iterations  # 计算平均损失

                        batch_sup_contrastive_loss = sup_contrastive_loss(
                            embd_batch=preject_head_embed,
                            labels=y,
                            device=device,
                            temperature=args.temperature,
                            base_temperature=args.temperature)

                        step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                        sys.stdout.write(
                            f"\rAvg_loss: {avg_loss:.4f}"
                        )
                        sys.stdout.flush()

                        epoch_train_loss_tem += step_loss.item()  # 2/20

                        step_loss.backward()

                        optimizer.step()

                        # 每处理一个 batch，增加迭代计数器。
                        num_iterations = num_iterations + 1

                    train_loss_tem_fold.append(epoch_train_loss_tem / num_iterations)  # 2/20
                    train_loss_feq_fold.append(epoch_train_loss_feq / num_iterations)  # 2/20

                model.eval()
                classifier.eval()
                projection_head.eval()

                # 验证集评估
                # 功能：
                #   使用 evaluate 函数在验证集上计算损失（val_loss）和准确率（val_accu_tem）。
                #   evaluate 应是一个接受数据加载器、模型、分类器、损失函数和设备的函数，返回模型的性能指标。
                val_loss, val_accu_tem, _, _, _, _ = evaluate(val_loader, model, classifier, loss, device, 0)

                val_loss_fold.append(val_loss)  # 2/13

                # 记录最佳验证损失和测试集性能
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    end_val_epoch = epoch

                test_loss, test_accuracy_tem, test_precision, test_recall, test_f1, conf_matrix = evaluate(test_loader, model,
                                                                                           classifier, loss, device, 0)

                test_accuracy = test_accuracy_tem

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    best_pre = test_precision[0]
                    best_rc = test_recall[0]
                    best_f1 = test_f1[0]

                    # 保存模型
                    torch.save(model.state_dict(),
                               os.path.join(log_path, f"model_best.pth"))
                    if is_projection_head:
                        torch.save(projection_head.state_dict(),
                                   os.path.join(log_path, f"projection_head_best.pth"))
                    torch.save(classifier.state_dict(),
                               os.path.join(log_path, f"classifier_best.pth"))

                val_acc_tem_fold.append(val_accu_tem.item())  # 2/13

                print("\nepoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))
                print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(
                    test_precision[0], test_precision[1], test_precision[2]))
                print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],test_recall[1],test_recall[2]))
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],test_f1[1],test_f1[2]))

                print("\nepoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy), file=log_file)
                print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(
                    test_precision[0], test_precision[1], test_precision[2]), file=log_file)
                print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],test_recall[1],test_recall[2]),file=log_file)
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],test_f1[1],test_f1[2]),file=log_file)

        if args.is_pseudo == 1:
            # 队列
            queue_train_x = Queue(maxsize=args.queue_maxsize)
            queue_train_y = Queue(maxsize=args.queue_maxsize)
            queue_train_mask = Queue(maxsize=args.queue_maxsize)

            num_iterations = 0
            for epoch in range(1, args.pseudo_epochs + 1):
                model.train()
                num_iterations = 0
                total_correct_pseudo = 0  # 所有高置信伪标签中的正确数量
                total_pseudo = 0  # 所有高置信伪标签总数
                cal_loss_label_num = 0  # 参与计算loss的样本的数量   所有带标签样本 + 高置信度伪标签样本
                total_labeled_loss = 0.0
                avg_labeled_loss = 0.0

                total_pseudo_loss = 0.0
                avg_pseudo_loss = 0.0

                all_pseudo_labels = []  # 收集所有伪标签

                print(f"\nPseudo Epoch {epoch}")
                print(f"\nPseudo Epoch {epoch}", file=log_file)

                for x, y in train_loader:  

                    start_idx = num_iterations * args.batch_size
                    end_idx = min((num_iterations + 1) * args.batch_size, len(train_tem))
                    y_batch = y.to(device)
                    mask_train_batch = torch.tensor(mask_train[start_idx:end_idx], dtype=torch.int, device=device)

                    optimizer.zero_grad()

                    pred_embed = model(x)

                    pred = classifier(pred_embed)

                    if is_projection_head:
                        preject_head_embed = projection_head(pred_embed)

                    mask_cpl_batch = torch.zeros(len(mask_train_batch), dtype=torch.bool, device=device)

                    if not queue_train_x.full():
                        queue_train_x.put(preject_head_embed.detach())
                        queue_train_y.put(y)
                        queue_train_mask.put(mask_train_batch.cpu().numpy())
                    else:
                        train_x_allq = torch.cat(list(queue_train_x.queue) + [preject_head_embed.detach()], dim=0)
                        train_y_allq = torch.cat(list(queue_train_y.queue) + [y], dim=0)
                        train_mask_allq = np.concatenate(
                            list(queue_train_mask.queue) + [mask_train_batch.cpu().numpy()])
                        # 生成伪标签
                        # end_knn_label 最终生成的伪标签
                        # mask_cpl_knn  高置信伪标签的掩码
                        _, end_knn_label, mask_cpl_knn, _ = construct_graph_via_knn_cpl_nearind_gpu(
                            data_embed=train_x_allq,
                            y_label=train_y_allq,
                            mask_label=train_mask_allq,
                            device=device,
                            topk=args.knn_num_tem,
                            sigma=args.sigma,
                            alpha=args.alpha,
                            p_cutoff=args.p_cutoff,
                            num_real_class=args.num_classes
                        )
                        knn_result_label = torch.tensor(end_knn_label, dtype=torch.int64, device=device)
                        y[mask_train_batch == 1] = knn_result_label[-len(y):][mask_train_batch == 1]
                        mask_cpl_batch[mask_train_batch == 1] = mask_cpl_knn[-len(y):][mask_train_batch == 1]

                        # 收集当前批次的高置信度伪标签
                        pseudo_labels = knn_result_label[-len(y):][mask_cpl_batch.cpu().numpy()]
                        all_pseudo_labels.extend(pseudo_labels.tolist())

                        # # 统计伪标签（仅用于监控，不影响训练）
                        batch_indices = np.arange(start_idx, end_idx)  # 当前批次的下标
                        true_labels_batch = true_labels[batch_indices].numpy()  # 当前批次的真实标签  只用于验证，不参与训练
                        pseudo_labels_batch = knn_result_label[-len(y):].cpu().numpy()  # 当前批次的伪标签
                        pseudo_mask_batch = mask_cpl_batch.cpu().numpy()  # 当前批次的伪标签的掩码（高置信度）
                        valid_pseudo = pseudo_labels_batch[pseudo_mask_batch]  # 当前批次的伪标签中值得信任的部分（高置信度伪标签）
                        valid_true = true_labels_batch[pseudo_mask_batch]  # 当前批次 高置信度伪标签样本 的 真实标签

                        # if len(valid_pseudo) > 0:
                        total_correct_pseudo += (valid_pseudo == valid_true).sum()  # 高置信度伪标签中正确的数量。
                        total_pseudo += len(valid_pseudo)

                        # 更新队列
                        queue_train_x.get()
                        queue_train_y.get()
                        queue_train_mask.get()
                        queue_train_x.put(preject_head_embed.detach())
                        queue_train_y.put(y)
                        queue_train_mask.put(mask_train_batch.cpu().numpy())

                    # 选择损失样本  所有的带标签数据 + 高置信度的伪标签数据
                    mask_select_loss = torch.zeros(len(y), dtype=torch.bool, device=device)
                    for m in range(len(mask_train_batch)):
                        if (mask_train_batch[m] == 0 or mask_cpl_batch[m]) and y[m] >= 0:  # 过滤无效标签:  # 有标签
                            mask_select_loss[m] = True

                    cal_loss_label_num += mask_select_loss.sum().item()

                    embed_loss = 0.01 * torch.norm(pred_embed, dim=1).mean()  # 限制范数

                    # 分离有标签和伪标签的掩码
                    mask_labeled = (mask_train_batch == 0) & (y >= 0)  # 有标签数据
                    mask_pseudo = mask_cpl_batch & (mask_train_batch != 0) & (y >= 0)  # 伪标签数据

                    # 计算损失
                    labeled_loss = 0.0
                    pseudo_loss = 0.0
                    if mask_labeled.sum() > 0:
                        labeled_loss = loss(pred[mask_labeled], y[mask_labeled])
                    if mask_pseudo.sum() > 0:
                        pseudo_loss = loss(pred[mask_pseudo], y[mask_pseudo])

                    total_labeled_loss += labeled_loss.item()
                    total_pseudo_loss += pseudo_loss
                    num_iterations += 1

                    # 计算损失
                    # step_loss = labeled_loss + pseudo_weight * pseudo_loss + embed_loss
                    step_loss = labeled_loss + args.pseudo_weight * pseudo_loss

                    batch_sup_contrastive_loss = sup_contrastive_loss(
                        embd_batch=preject_head_embed[mask_train_batch == 0],
                        labels=y[mask_train_batch == 0],
                        device=device,
                        temperature=args.temperature,
                        base_temperature=args.temperature)

                    step_loss = step_loss + batch_sup_contrastive_loss * args.sup_con_mu

                    sys.stdout.write(
                        f"\rlabeled_loss: {labeled_loss:.4f}, pseudo_loss: {pseudo_loss:.4f}, batch_sup_contrastive_loss: {batch_sup_contrastive_loss:.4f}, step_loss: {step_loss:.4f}")
                    sys.stdout.flush()

                    step_loss.backward()

                    optimizer.step()

                avg_labeled_loss = total_labeled_loss / num_iterations
                avg_pseudo_loss = total_pseudo_loss / num_iterations
                print(f"\navg_labeled_loss = {avg_labeled_loss:.4f}, avg_pseudo_loss = {avg_pseudo_loss:.4f}")
                print(f"\navg_labeled_loss = {avg_labeled_loss:.4f}, avg_pseudo_loss = {avg_pseudo_loss:.4f}",
                      file=log_file)
                # 在每轮结束时计算并输出平均伪标签正确率和总数
                if total_pseudo > 0:
                    avg_pseudo_accuracy = total_correct_pseudo / total_pseudo
                    print(
                        f"\nAverage Pseudo-label accuracy: {avg_pseudo_accuracy:.4f}, Total Confident labels: {total_pseudo}")
                    print(
                        f"\nAverage Pseudo-label accuracy: {avg_pseudo_accuracy:.4f}, Total Confident labels: {total_pseudo}",
                        file=log_file)
                    print("Pseudo label distribution:", Counter(all_pseudo_labels))
                    print("Pseudo label distribution:", Counter(all_pseudo_labels), file=log_file)

                print("cal_loss_label_num = {} + {} = {}".format(len(labeled_indices),
                                                                 cal_loss_label_num - len(labeled_indices),
                                                                 cal_loss_label_num))
                print("cal_loss_label_num = {} + {} = {}".format(len(labeled_indices),
                                                                 cal_loss_label_num - len(labeled_indices),
                                                                 cal_loss_label_num), file=log_file)

                model.eval()
                classifier.eval()
                projection_head.eval()

                # 验证集评估
                # 功能：
                #   使用 evaluate 函数在验证集上计算损失（val_loss）和准确率（val_accu_tem）。
                #   evaluate 应是一个接受数据加载器、模型、分类器、损失函数和设备的函数，返回模型的性能指标。
                val_loss, val_accu_tem, _, _, _, _ = evaluate(val_loader, model, classifier, loss, device, 0)

                val_loss_fold.append(val_loss)  # 2/13

                # 记录最佳验证损失和测试集性能
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    end_val_epoch = epoch

                test_loss, test_accuracy_tem, test_precision, test_recall, test_f1, conf_matrix = evaluate(test_loader, model, classifier, loss, device, 0)

                test_accuracy = test_accuracy_tem

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    best_pre = test_precision[0]
                    best_rc = test_recall[0]
                    best_f1 = test_f1[0]

                    # 保存模型
                    torch.save(model.state_dict(),
                               os.path.join(log_path, f"model_best.pth"))
                    if is_projection_head:
                        torch.save(projection_head.state_dict(),
                                   os.path.join(log_path, f"projection_head_best.pth"))
                    torch.save(classifier.state_dict(),
                               os.path.join(log_path, f"classifier_best.pth"))

                val_acc_tem_fold.append(val_accu_tem.item())  # 2/13

                print("\ntest_accuracy : {:.3f}".format(test_accuracy))
                print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(test_precision[0],test_precision[1],test_precision[2]))
                print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],test_recall[1],test_recall[2]))
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],test_f1[1],test_f1[2]))

                print("\ntest_accuracy : {:.3f}".format(test_accuracy), file=log_file)
                print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(test_precision[0], test_precision[1], test_precision[2]), file=log_file)
                print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],test_recall[1],test_recall[2]),file=log_file)
                print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0],test_f1[1],test_f1[2]),file=log_file)


        losses.append(val_loss_fold)    # 2/13
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
    print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed,file=log_file)

    # print("test_acc     =", round(torch.mean(test_accuracies).item(), 3))
    # print(
    #     "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0], test_recall[1],
    #                                                                                     test_recall[2]))
    # print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
    #                                                                                       test_f1[2]))
    # print('\nDone!')
    #
    # print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed,
    #       file=log_file)
    #
    # print("test_acc     =", round(torch.mean(test_accuracies).item(), 3), file=log_file)
    # print(
    #     "macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0], test_recall[1],
    #                                                                                     test_recall[2]), file=log_file)
    # print("macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
    #                                                                                       test_f1[2]), file=log_file)
    # print('\nDone!', file=log_file)

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

    acc = round(best_acc.item(), 3)
    acc = str(acc)
    acc = acc.replace('.', '_')

    pre = round(best_pre, 3)
    pre = str(pre)
    pre = pre.replace('.', '_')

    rc = round(best_rc, 3)
    rc = str(rc)
    rc = rc.replace('.', '_')

    f1 = round(best_f1, 3)
    f1 = str(f1)
    f1 = f1.replace('.', '_')

    # # 保存模型
    torch.save(model.state_dict(), os.path.join(log_path, f"model_last.pth"))
    if is_projection_head:
        torch.save(projection_head.state_dict(),
                   os.path.join(log_path, f"projection_head_last.pth"))
    torch.save(classifier.state_dict(), os.path.join(log_path, f"classifier_last.pth"))

    old_name = log_path
    new_name = log_path + "_" + acc + "_" + pre + "_" + rc + "_" + f1
    if args.is_pseudo == 1:
        new_name = new_name + "_pseudo_knn_"
    else:
        new_name = new_name + "_base_"
    new_name = new_name + "_" + str(args.labeled_ratio) + "_" + str(len(sum_target)) + "_main_time_only"
    os.rename(old_name, new_name)
