import datetime
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ts_tfc_ssl.ts_data.dataloader import UCRDataset, GraphDataset
from ts_tfc_ssl.ts_data.preprocessing import normalize_per_series, fill_nan_value
from ts_tfc_ssl.ts_utils import build_model, set_seed, build_dataset, get_all_datasets, \
    build_loss, evaluate_base, custom_collate_fn, evaluate_graph

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='DAPP', help='DAPP GCN GAT GIN GraphIoT')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='USTCTFC',
                        help='dataset(in ucr)')  # TMC13 IOT17 Medboit USTCTFC
    parser.add_argument('--dataroot', type=str, default='./IOT_Dataset', help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=13, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--log_path', type=str, default='./train_log')
    parser.add_argument('--max_flow_length', type=int, default=20)

    # Semi training
    parser.add_argument('--is_sl', type=int, default=1, help='0, 1')
    parser.add_argument('--labeled_ratio', type=float, default=0.01, help='0.1, 0.2, 0.4')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs using only labeled data for ssl')

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

    # 调用 build_model(args) 函数，根据 args 参数构建主模型(时域) model 和分类器 classifier
    model, _ = build_model(args)

    # 将主模型、分类器和投影头迁移到指定的设备（如 GPU 或 CPU），以支持加速计算。
    model = model.to(device)

    class_weights = torch.ones(args.num_classes, device=device)
    # 调用 build_loss(args) 函数，创建一个损失函数（如交叉熵损失），并迁移到设备。 cross_entropy  |  reconstruction_loss
    loss = build_loss(args, class_weights).to(device)

    # 使用 state_dict() 保存模型、分类器和投影头的初始参数状态。这对于后续恢复模型参数或初始化新的实例非常有用。
    model_init_state = model.state_dict()

    # 使用 torch.optim.Adam 优化器，同时优化模型（model）、分类器（classifier）和投影头（projection_head）的参数。
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)

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
        train_loss_tem_fold = []  # 2/20
        train_loss_feq_fold = []  # 2/20

        # 作用：每次进入新的折 (fold) 训练时，将模型、分类器和投影头的状态恢复到初始状态。
        # 目的：确保每折训练和评估的起点一致，不受之前折的训练结果影响。这里涉及两个模型（model 和 model_feq），
        # 分别对应常规输入和频域输入（可能用于频域特征提取）。
        model.load_state_dict(model_init_state)

        # 作用：从每折数据中提取当前折训练、验证和测试集。
        train_target = train_targets[i]

        val_dataset = val_datasets[i]
        val_target = val_targets[i]

        test_dataset = test_datasets[i]
        test_target = test_targets[i]

        # 处理数据中的缺失值，将 NaN 值替换为合适的数值。
        train_dataset, val_dataset, test_dataset = fill_nan_value(train_dataset, val_dataset, test_dataset)

        # TODO normalize per series
        # 作用：对每个序列单独归一化，可能是将数据标准化到某个范围（如 [0, 1] 或均值为 0，方差为 1）。
        # 目的：确保训练、验证和测试数据分布一致，避免因特定数据范围差异而影响模型效果。

        # train_dataset = normalize_per_series(train_dataset, 1500, 1)
        # val_dataset = normalize_per_series(val_dataset, 1500, 1)
        # test_dataset = normalize_per_series(test_dataset, 1500, 1)

        # 时域数据转换为张量并增加通道维度
        # train_tem = torch.from_numpy(train_dataset).float().unsqueeze(1)  # (n_samples, 1, 50)
        # val_tem = torch.from_numpy(val_dataset).float().unsqueeze(1)
        # test_tem = torch.from_numpy(test_dataset).float().unsqueeze(1)

        # train_tem = torch.from_numpy(train_dataset).float()
        # val_tem = torch.from_numpy(val_dataset).float()
        # test_tem = torch.from_numpy(test_dataset).float()

        # 划分有标签和未标记数据（例如 20% 有标签，80% 未标记）
        labeled_indices, unlabeled_indices = train_test_split(
            np.arange(len(train_dataset)),
            test_size=1 - args.labeled_ratio,  # 未标记比例
            stratify=train_target,  # 按类别分层采样，确保分布一致
            random_state=args.random_seed
        )

        train_tem_labeled = train_dataset[labeled_indices]
        train_target_labeled = train_target[labeled_indices]

        labeled_set = GraphDataset(train_tem_labeled,
                                   torch.from_numpy(train_target_labeled).to(device).to(torch.int64),
                                   args.dataset, 'train', args.labeled_ratio)

        val_set = GraphDataset(val_dataset,
                               torch.from_numpy(val_target).to(device).to(torch.int64),
                               args.dataset, 'val', args.labeled_ratio)

        test_set = GraphDataset(test_dataset,
                                torch.from_numpy(test_target).to(device).to(torch.int64),
                                args.dataset, 'test', args.labeled_ratio)

        print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}")
        print(f"Labeled samples: {len(labeled_indices)}, Unlabeled samples: {len(unlabeled_indices)}",
              file=log_file)

        batch_size_labeled = 16

        while train_dataset.shape[0] * args.labeled_ratio < batch_size_labeled:
            batch_size_labeled = batch_size_labeled // 2

        if train_dataset.shape[0] < 16:
            batch_size_labeled = 16

        # 数据加载器
        train_labeled_loader = DataLoader(labeled_set, batch_size=batch_size_labeled, num_workers=0, drop_last=True,
                                          shuffle=True, collate_fn=lambda batch: custom_collate_fn(batch))
        val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0,
                                collate_fn=lambda batch: custom_collate_fn(batch))
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                                 collate_fn=lambda batch: custom_collate_fn(batch))

        min_val_loss = float('inf')
        test_accuracy = 0
        end_val_epoch = 0
        min_val_loss_feq = float('inf')
        test_accuracy_tem = 0
        end_val_epoch_tem = 0
        test_accuracy_feq = 0
        end_val_epoch_feq = 0

        for epoch in range(1, args.warmup_epochs + 1):
            num_iterations = 0

            epoch_train_loss_tem = 0  # 2/20
            epoch_train_loss_feq = 0  # 2/20

            model.train()

            total_loss = 0.0

            for x, y in train_labeled_loader:  # train_labeled_loader：仅包含带标签的数据，专用于有监督训练。
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                pred = model(x)

                step_loss = loss(pred, y)

                # 累积损失
                total_loss += step_loss.item()
                num_iterations += 1
                avg_loss = total_loss / num_iterations  # 计算平均损失

                sys.stdout.write(
                    f"\rAvg_loss: {avg_loss:.4f}"
                )
                sys.stdout.flush()

                epoch_train_loss_tem += step_loss.item()  # 2/20

                step_loss.backward()

                optimizer.step()

                # 每处理一个 batch，增加迭代计数器。
                num_iterations = num_iterations + 1

            # train_loss_tem_fold.append(epoch_train_loss_tem / num_iterations)  # 2/20
            # train_loss_feq_fold.append(epoch_train_loss_feq / num_iterations)  # 2/20

            model.eval()

            val_loss, val_accu_tem, _, _, _, _ = evaluate_graph(val_loader, model, loss, device)

            val_loss_fold.append(val_loss)  # 2/13

            # 记录最佳验证损失和测试集性能
            if min_val_loss > val_loss:
                min_val_loss = val_loss
                end_val_epoch = epoch

            test_loss, test_accuracy_tem, test_precision, test_recall, test_f1, conf_matrix = evaluate_graph(test_loader,
                                                                                                            model, loss, device)

            test_accuracy = test_accuracy_tem

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best_pre = test_precision[0]
                best_rc = test_recall[0]
                best_f1 = test_f1[0]

                # 保存模型
                torch.save(model.state_dict(),
                           os.path.join(log_path, f"model_best.pth"))

            val_acc_tem_fold.append(val_accu_tem.item())  # 2/13

            print("\nepoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy))
            print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(
                test_precision[0], test_precision[1], test_precision[2]))
            print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                  test_recall[1],
                                                                                                  test_recall[2]))
            print(
                "macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
                                                                                                test_f1[2]))

            print("\nepoch : {},  test_accuracy : {:.3f}".format(epoch, test_accuracy), file=log_file)
            print("macro_pre = {:.3f}, micro_pre = {:.3f}, weighted_pre = {:.3f}".format(
                test_precision[0], test_precision[1], test_precision[2]), file=log_file)
            print("macro_recall = {:.3f}, micro_recall = {:.3f}, weighted_recall = {:.3f}".format(test_recall[0],
                                                                                                  test_recall[1],
                                                                                                  test_recall[2]),
                  file=log_file)
            print(
                "macro_f1     = {:.3f}, micro_f1     = {:.3f}, weighted_f1     = {:.3f}".format(test_f1[0], test_f1[1],
                                                                                                test_f1[2]),
                file=log_file)

        losses.append(val_loss_fold)  # 2/13
        val_acc_tem.append(val_acc_tem_fold)  # 2/13
        val_acc_feq.append(val_acc_feq_fold)  # 2/13
        train_loss_tem.append(train_loss_tem_fold)  # 2/20
        train_loss_feq.append(train_loss_feq_fold)  # 2/20

        test_accuracies.append(test_accuracy)
        end_val_epochs.append(end_val_epoch)
        test_accuracies_tem.append(test_accuracy_tem)
        end_val_epochs_tem.append(end_val_epoch)
        test_accuracies_feq.append(test_accuracy_feq)
        end_val_epochs_feq.append(end_val_epoch_feq)

        t = time.time() - t
        train_time += t

    test_accuracies = torch.Tensor(test_accuracies)
    end_val_epochs = np.array(end_val_epochs)

    test_accuracies_tem = torch.Tensor(test_accuracies_tem)
    end_val_epochs_tem = np.array(end_val_epochs_tem)

    test_accuracies_feq = torch.Tensor(test_accuracies_feq)
    end_val_epochs_feq = np.array(end_val_epochs_feq)

    print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed)
    print("\nTraining end: Training time (seconds) = ", round(train_time, 3), ", seed = ", args.random_seed,
          file=log_file)

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

    old_name = log_path
    new_name = log_path + "_" + acc + "_" + pre + "_" + rc + "_" + f1
    # if args.is_pseudo == 1:
    #     new_name = new_name + "_pseudo_knn_"
    # else:
    #     new_name = new_name + "_base_"
    new_name = new_name + "_" + str(args.labeled_ratio) + "_main_base_graph"
    os.rename(old_name, new_name)
