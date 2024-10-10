import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

# 在这个损失函数中，真实数据是 labels，也就是输入给模型的真实样本的标签或数据。对于变分自编码器（VAE）来说，真实数据通常是输入给编码器的原始数据，经过编码器和解码器的处理后，VAE生成与输入数据结构相似的重构数据。

# 具体到这个函数中：
#preds和labels维度相同，都是n_nodes x n_nodes。这两个参数是模型的输出和输入，都是二值化的邻接矩阵，表示图中节点之间的连接关系。
# 	•	labels 是模型的真实标签，可能是输入数据的二值化版本（如二值图像的像素值，0 或 1），与预测的 preds 相对应。
# 	•	preds 是模型生成的预测值或重构值。

# binary_cross_entropy_with_logits 函数会比较 preds 和 labels，计算模型生成的数据与输入的真实数据之间的差异。对于二值化数据，labels 通常是原始输入数据的二值标签。这样，VAE通过最小化重构误差，使得生成的数据尽可能接近这些真实的输入数据。

# 2024年10月8日，批量损失函数的实现：
# 调整损失函数的思路

# 	1.	考虑批量维度：原始的损失函数假设 preds 和 labels 是单个图的  n \times n  矩阵，因此计算时没有考虑到批量维度。在批量输入时，preds 和 labels 的形状变为 (batch\_size, n, n)，因此需要在计算损失时考虑批量维度。
# 	2.	调整 binary_cross_entropy_with_logits：F.binary_cross_entropy_with_logits 支持批量数据，因此可以直接用于计算批量情况下的重构损失，但需要确保 preds 和 labels 的形状匹配。
# 	3.	调整 KL 散度计算：在原始公式中，KL 散度的计算假设输入是单个图，需要将其调整为对**每个被试（每个批量）**计算，然后取平均值。

#深度学习模型里对于批量数据的损失函数会取mean嘛?
# 是的，在深度学习模型中，对于批量数据的损失函数通常会取 mean，原因是这样可以使得梯度更新的规模保持一致，从而让训练过程更稳定。下面详细解释为什么在深度学习中对批量数据的损失函数取平均值是一个常见的做法：

# 1. 为什么对批量数据的损失取 mean？

# 	•	稳定梯度更新：
# 	•	深度学习模型使用的优化器（如 SGD 或 Adam）通常是基于梯度下降的。如果不对批量数据的损失取平均，那么随着批量大小 (batch_size) 的变化，损失的总和也会变化，从而导致梯度的规模不一致。
# 	•	例如，如果批量大小较大，未平均的损失会更大，导致模型参数的更新步长更大，这会使得训练过程变得不稳定。
# 	•	通过对损失取平均，确保不论 batch_size 是多少，每次参数更新的幅度保持在一个合理的范围内。
# 	•	与学习率的关系：
# 	•	学习率决定了模型每次更新参数的步长。未平均的损失会使得步长变得与 batch_size 相关，而通过对损失取平均，学习率的设置就可以和 batch_size 解耦，从而使得模型在不同的 batch_size 下都能稳定地训练。
# 	•	一致的尺度：
# 	•	取平均后，损失函数的数值不再依赖于批量大小，而是与样本的平均误差相关。这使得在不同的批量大小下训练时，损失值的范围是一致的，便于模型在训练过程中的监控和调试。
# 	•	标准做法：
# 	•	绝大多数深度学习库（如 TensorFlow 和 PyTorch）中，默认的损失函数计算方式都是对 batch 维度取平均的。例如，torch.nn.CrossEntropyLoss 和 torch.nn.MSELoss 都是默认对批量数据的损失取平均。
# 	•	当然，如果需要，也可以通过参数设置让这些损失函数对批量数据的损失取 sum，但一般情况下，mean 是更合理的选择。

# 2. 取 mean 的计算方式

# 在批量数据的情况下，假设我们有 batch_size 个样本，每个样本的损失为 loss_i，那么对整个批量的损失取 mean 的计算方式是：

# \[
# \text{batch\_loss} = \frac{1}{\text{batch\size}} \sum{i=1}^{\text{batch\_size}} \text{loss}_i
# \]

# 	•	这样计算的结果是每个样本的平均损失。
# 	•	当反向传播计算梯度时，每个样本对总损失的贡献是相同的，因此不会因为批量大小的不同而导致梯度的尺度发生显著变化。

# 3. 什么时候不取 mean？

# 虽然大多数情况下对批量损失取平均是合理的，但也有一些特殊情况可能不需要对损失取平均，例如：

# 	•	调整梯度规模：在某些情况下，如果希望根据 batch_size 的变化调整梯度更新的步长，可以选择对损失取 sum，但这通常需要调整学习率以适应新的更新尺度。
# 	•	不平衡样本处理：当样本类别不平衡时，有时会对不同样本赋予不同的权重，这种情况下可能需要更精细地控制损失的计算方式，而不仅仅是简单地对批量取平均。
# 	•	自定义损失函数：如果自定义损失函数有特定需求（例如，累积误差的统计分析），也可以选择不取平均。

# 4. 在你的任务中取 mean 的合理性

# 	•	你的任务中有多个被试，每个被试的图被视作一个样本。通过对损失函数取平均，保证了每个被试对总损失的贡献是相同的。
# 	•	这样做可以使得模型在训练过程中不会因为被试的数量（即 batch_size）的不同而导致损失值大幅变化，保持训练的稳定性。
# 	•	对于 binary_cross_entropy_with_logits，默认情况下它是按 mean 进行处理的，这样可以确保损失值与 batch_size 无关。

# 总结

# 	•	在深度学习模型中，对批量数据的损失取平均 (mean) 是非常常见且标准的做法。
# 	•	取 mean 能够确保模型训练过程中的梯度更新稳定，损失值尺度一致，从而有助于选择合适的学习率并提高训练效果。
# 	•	在你的任务中，考虑到模型是对多个被试的数据进行训练，使用 mean 可以确保不同 batch_size 下的训练效果一致，进而提高模型的泛化能力和训练稳定性。

# def loss_function_batches(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
#     """
#     计算批量情况下的损失函数，包括重构损失和 KL 散度。
#     :param preds: 重构后的邻接矩阵 (batch_size, n, n)
#     :param labels: 原始邻接矩阵 (batch_size, n, n)
#     :param mu: 编码器输出的均值 (batch_size, n, hidden_dim)
#     :param logvar: 编码器输出的对数方差 (batch_size, n, hidden_dim)
#     :param n_nodes: 图中节点的数量
#     :param norm: 用于归一化的因子
#     :param pos_weight: 正样本权重，用于平衡二值交叉熵损失
#     :return: 重构损失 + KL 散度的总损失
#     """
#     # 计算重构损失，使用 binary_cross_entropy_with_logits 并考虑 batch_size
#     cost = norm * F.binary_cross_entropy_with_logits(
#         preds, labels, pos_weight=pos_weight, reduction='mean'
#     )

#     # 计算 KL 散度，使用批量维度进行计算
#     # KL 散度公式：0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     # 对每个样本的 KL 散度取平均值
#     KLD = -0.5 / n_nodes * torch.mean(
#         torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=[1, 2])
#     )

#     return cost + KLD


def loss_function_batches(preds, labels, mu, logvar, n_nodes, norms, pos_weights):
    """
    计算批量情况下的损失函数，包括重构损失和 KL 散度。
    :param preds: 重构后的邻接矩阵 (batch_size, n, n)
    :param labels: 原始邻接矩阵 (batch_size, n, n)
    :param mu: 编码器输出的均值 (batch_size, n, hidden_dim)
    :param logvar: 编码器输出的对数方差 (batch_size, n, hidden_dim)
    :param n_nodes: 图中节点的数量
    :param norms: 每个被试的归一化因子列表 (batch_size,)
    :param pos_weights: 每个被试的正样本权重列表 (batch_size,)
    :return: 重构损失 + KL 散度的总损失
    """
    # 将 norms 和 pos_weights 转换为张量
    norms = torch.tensor(norms, dtype=torch.float32, device=preds.device)
    pos_weights = torch.tensor(pos_weights, dtype=torch.float32, device=preds.device)

    # 计算重构损失，使用 binary_cross_entropy_with_logits，并考虑 pos_weight 的广播
    # 使用 reduction='none' 保留每个样本的损失，然后通过 norms 进行加权
    bce_loss = F.binary_cross_entropy_with_logits(
        preds, labels, pos_weight=pos_weights.view(-1, 1, 1), reduction='none'
    )

    # 对每个样本的损失进行加权和平均
    weighted_bce_loss = (bce_loss.mean(dim=(1, 2)) * norms).mean()

    # 计算 KL 散度，使用批量维度进行计算
    # KL 散度公式：0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 对每个样本的 KL 散度取平均值
    KLD = -0.5 / n_nodes * torch.mean(
        torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=[1, 2])
    )

    # 返回总损失，包括重构损失和 KL 散度
    return weighted_bce_loss + KLD