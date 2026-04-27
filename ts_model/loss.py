import torch
import torch.nn as nn

EPS = 1e-8


# 返回一个用于分类任务的交叉熵损失函数。
def cross_entropy(class_weights):
    loss = nn.CrossEntropyLoss(weight=class_weights)
    return loss


# 返回一个用于重构任务的均方误差（MSE）损失函数。
def reconstruction_loss():
    loss = nn.MSELoss()
    return loss


# 这段代码实现了一个监督对比损失函数（Supervised Contrastive Loss），它被用于对比学习（contrastive learning）任务中，
# 特别是在有标签的情况下。
# 函数参数
# embd_batch：一个包含样本嵌入（embedding）的张量，形状为 (𝑁, 𝑑) 其中 N 是批量大小，d 是嵌入维度。
# labels 样本的类别标签，形状为 (N,)。
def sup_contrastive_loss(embd_batch, labels, device,
                         temperature=0.07, base_temperature=0.07):
    """计算监督对比损失，处理空输入情况"""
    batch_size = embd_batch.size(0)
    if batch_size == 0:  # 无有效样本
        return torch.tensor(0.0, device=device)

    # 1. 计算样本对之间的相似性
    # 通过点积计算每对样本之间的相似性，结果是一个 (N,N) 的矩阵。
    # 将结果除以温度参数 temperature，以调节 logits 的分布。
    anchor_dot_contrast = torch.div(
        torch.matmul(embd_batch, embd_batch.T),
        temperature)

    # 2对 logits 做数值稳定性处理
    # 为了避免数值溢出（numerical overflow），将相似性矩阵的每行减去该行的最大值。
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # 3. 构造类别掩码
    # 将标签展开为 (N,1) 的形状。
    # 使用标签构造一个 (N,N) 的掩码矩阵 mask，表示哪些样本属于同一类别：
    # mask[i,j]=1 表示第 i 个样本和第 j 个样本属于同一类别。
    # 否则 mask[i,j]=0。
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # 4. 构造 logits 掩码
    # logits_mask 是一个对角线为 0，其余元素为 1 的掩码矩阵，用于排除样本与自身的相似性。
    # 将 logits_mask 和 mask 相乘，得到最终的掩码
    logits_mask = torch.scatter(
        torch.ones_like(logits.detach()),
        1,
        torch.arange(embd_batch.shape[0]).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # 5. 计算对数概率
    # 对 logits 取指数，得到 softmax 的分子部分。
    # 计算每个样本的对数概率。
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # 6. 计算正样本的对数似然的均值
    # 使用掩码 mask 提取属于同一类别的样本对。
    # 对正样本对的对数概率求平均。
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # 7. 计算有效 anchor 数量
    # 统计每个 anchor 是否存在正样本对（即 mask 的每行是否非零）。
    num_anchor = 0
    for s in mask.sum(1):
        if s != 0:
            num_anchor = num_anchor + 1

    # 8. 计算最终损失
    # 损失定义为正样本对对数似然的负均值，并根据温度进行归一化。
    # 最终对所有有效 anchor 的损失求平均。
    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.sum(0) / (num_anchor + 1e-12)

    # 返回一个标量 loss，表示监督对比损失的平均值。
    return loss

# 这段代码实现了监督对比学习中的核心损失函数，主要思想如下：
# 利用样本的嵌入计算相似性矩阵。
# 通过标签构造正样本对的掩码。
# 计算正样本对的对数似然，并取负均值作为损失。


