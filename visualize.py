import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.fft as fft
import sys
from collections import Counter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from ts_tfc_ssl.ts_model.model import ParallelModel, FCN
from ts_tfc_ssl.ts_model.model1111 import ProjectionHead
from ts_tfc_ssl.ts_utils import build_dataset, get_all_datasets, convert_coeff, set_seed
from ts_tfc_ssl.ts_data.preprocessing import normalize_per_series, normalize_freq_data
from ts_tfc_ssl.ts_data.dataloader import IOTDataset, UCRDataset



# 参数设置（需与训练时一致）
class Args:
    batch_size = 1024  # 示例值，替换为训练时的值
    num_classes = 17   # 示例值，替换为你的类别数
    is_projection_head = True  # 是否使用投影头，根据训练时设置
    dataroot = r'D:\code\pycharm_code\My-TFC-main2\ts_tfc_ssl\IOT_Dataset'
    dataset = 'IOT17'
    # dataset = 'TMC13'
    backbone = 'FCN_Time'
    # backbone = 'Time_Freq'
    input_size = 1
    random_seed = 42

args = Args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# device = torch.device("cuda:1")
set_seed(args)
# 加载模型
model = FCN(args.num_classes, args.input_size).to(device)  # 替换为你的模型类
# model = ParallelModel(args.num_classes).to(device)  # 替换为你的模型类
projection_head = ProjectionHead(input_dim=128).to(device) if args.is_projection_head else None  # 替换为你的投影头类
# classifier = Classifier().to(device)  # 替换为你的分类器类

base_dir = r"D:\code\pycharm_code\My-TFC-main2\ts_tfc_ssl\train_log"

# model_file = r"IOT21\Time_Freq\20250325_211655_Done_0_952_base_0.1_335212_main_knn\model_ratio_0.1_acc_0_952.pth"
# projection_head_file = r"IOT21\Time_Freq\20250325_211655_Done_0_952_base_0.1_335212_main_knn\projection_head_ratio_0.1_acc_0_952.pth"

# model_file = r"IOT21\Time_Freq\20250325_165023_Done_0_951_pseudo_knn_0.1_335212_main_knn\model_ratio_0.1_acc_0_951.pth"
# projection_head_file = r"IOT21\Time_Freq\20250325_165023_Done_0_951_pseudo_knn_0.1_335212_main_knn\projection_head_ratio_0.1_acc_0_951.pth"


# model_file = r"TMC13\Time_Freq\20250326_135607_Done_0_975_base_0.1_80688_main_knn\model_ratio_0.1_acc_0_975.pth"
# projection_head_file = r"TMC13\Time_Freq\20250326_135607_Done_0_975_base_0.1_80688_main_knn\projection_head_ratio_0.1_acc_0_975.pth"

# model_file = r"IOT21\FCN_Time\20250402_195735_0_887_0_687_0_722_0_699_pseudo_knn__0.1_335212_main_time_only\model_ratio_0.1_acc_0_887.pth"
# projection_head_file = r"IOT21\FCN_Time\20250402_195735_0_887_0_687_0_722_0_699_pseudo_knn__0.1_335212_main_time_only\projection_head_ratio_0.1_acc_0_887.pth"

# model_file = r"IOT20\Time_Freq\20250409_155241_0_973_0_953_0_953_0_952_main_time_fre_parallel\model_acc_0_973.pth"
# projection_head_file = r"IOT20\Time_Freq\20250409_155241_0_973_0_953_0_953_0_952_main_time_fre_parallel\projection_head_acc_0_973.pth"

# model_file = r"IOT20\Time_Freq\20250409_164833_0_933_0_846_0_828_0_821_pseudo_knn_0.03_329309_main_knn\model_ratio_0.03_acc_0_933.pth"
# projection_head_file = r"IOT20\Time_Freq\20250409_164833_0_933_0_846_0_828_0_821_pseudo_knn_0.03_329309_main_knn\projection_head_ratio_0.03_acc_0_933.pth"

# IOT17
# model_file = r"IOT17\Time_Freq\20250409_233913_0_965_0_953_0_936_0_941_pseudo_knn_0.1_323422_main_knn\model_best.pth"
# projection_head_file = r"IOT17\Time_Freq\20250409_233913_0_965_0_953_0_936_0_941_pseudo_knn_0.1_323422_main_knn\projection_head_best.pth"

model_file = r"IOT17\FCN_Time\20250409_210509_0_887_0_639_0_678_0_656_base__0.1_323422_main_time_only\model_best.pth"
projection_head_file = r"IOT17\FCN_Time\20250409_210509_0_887_0_639_0_678_0_656_base__0.1_323422_main_time_only\projection_head_best.pth"


model_path = os.path.join(base_dir, model_file)
projection_head_path = os.path.join(base_dir, projection_head_file)

model.load_state_dict(torch.load(model_path))
if args.is_projection_head:
    projection_head.load_state_dict(torch.load(projection_head_path))

model.eval()
if projection_head:
    projection_head.eval()

# 数据加载器（需与训练时一致）
sum_dataset, sum_target, num_classes = build_dataset(args)
train_datasets, train_targets, val_datasets, val_targets, test_datasets, test_targets = get_all_datasets(
        sum_dataset, sum_target, args)
test_dataset = test_datasets[0]
test_target = test_targets[0]
test_dataset = normalize_per_series(test_dataset, 1500, 1)
test_tem = torch.from_numpy(test_dataset).float().unsqueeze(1)
test_fft = fft.rfft(torch.from_numpy(test_dataset), dim=-1)
test_fft, _ = convert_coeff(test_fft)
test_fft = normalize_freq_data(test_fft)
test_freq = F.interpolate(test_fft, size=50, mode='linear', align_corners=False)
test_set = IOTDataset(test_tem.to(device), test_freq.to(device),
                              torch.from_numpy(test_target).to(device).to(torch.int64))
test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

test_set_time = UCRDataset(test_tem.to(device),
                                  torch.from_numpy(test_target).to(device).to(torch.int64))
test_loader_time = DataLoader(test_set_time, batch_size=args.batch_size, num_workers=0)

# 可视化函数（仅测试集）
def visualize_test_embeddings(model, projection_head, test_loader, device, args, embed_type="pred_embed", top_n=10):
    test_embeddings = []
    test_labels = []
    i = 0
    # with torch.no_grad():
    #     for time_data, freq_data, y, _ in test_loader:
    #         print(f"第{i}次迭代")
    #         i += 1
    #         time_data = time_data.to(device)
    #         freq_data = freq_data.to(device)
    #         pred, pred_embed = model(time_data, freq_data)
    #         if embed_type == "preject_head_embed" and args.is_projection_head:
    #             embed = projection_head(pred_embed)
    #         else:
    #             embed = pred_embed
    #         test_embeddings.append(embed.cpu().numpy())
    #         test_labels.append(y.cpu().numpy())

    with torch.no_grad():
        for data, target in test_loader_time:
            print(f"第{i}次迭代")
            i += 1
            data = data.to(device)
            pred_embed = model(data)
            if embed_type == "preject_head_embed" and args.is_projection_head:
                embed = projection_head(pred_embed)
            else:
                embed = pred_embed
            test_embeddings.append(embed.cpu().numpy())
            test_labels.append(target.cpu().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # 统计每个类别的样本数
    label_counts = Counter(test_labels)
    print("Label counts:", label_counts)

    # 选出样本数最多的 top_n 个类别
    least_classes = [cls for cls, count in label_counts.most_common()[-top_n:]][::-1]  # 取最后n个并反转
    top_classes = [cls for cls, count in label_counts.most_common(top_n)]
    print(f"Top {top_n} classes:", top_classes)
    print(f"Top {top_n} least_classes:", least_classes)

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=args.random_seed)
    test_tsne = tsne.fit_transform(test_embeddings)

    # 可视化
    plt.figure(figsize=(8, 6))
    for cls in top_classes:
    # for cls in least_classes:
        idx = test_labels == cls
        plt.scatter(test_tsne[idx, 0], test_tsne[idx, 1], label=f"Class {cls} ({label_counts[cls]})", alpha=0.6)
    # plt.title(f"Test Set t-SNE ({embed_type})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = r'D:\code\pycharm_code\My-TFC-main2\ts_tfc_ssl\visualize'
    save_path = os.path.join(save_path, args.dataset)
    save_path = os.path.join(save_path, args.backbone)
    plt.savefig(os.path.join(save_path, f"tsne_test_{embed_type}.png"))
    plt.close()
    print(f"t-SNE visualization saved as tsne_test_{embed_type}.png")

# 执行可视化
visualize_test_embeddings(model, projection_head, test_loader, device, args, embed_type="pred_embed", top_n=10)

if args.is_projection_head:
    visualize_test_embeddings(model, projection_head, test_loader, device, args, embed_type="preject_head_embed", top_n=10)