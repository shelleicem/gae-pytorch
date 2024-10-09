from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import sys

# 将特定路径加入到 Python 搜索路径中
sys.path.append('/root/autodl-tmp/.autodl/graph_link_prediction/graph_link_prediction_forfmri/GAE/gae-pytorch')

from gae.modelmulti import GCNModelVAE
from gae.optimizer import loss_function_batches
from gae.utils import load_and_save_data, mask_test_edges, preprocess_graph_batches, get_roc_score_batches

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='myUKB_FC', help='type of dataset.')
parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/.autodl/data.npz", help="path to the dataset")

args = parser.parse_args()

def gae_for(args):
    # 打印当前使用的数据集名称
    print("Using {} dataset".format(args.dataset_str))

    # 加载图数据的邻接矩阵和特征矩阵
    # 假设新的数据加载函数返回的是已划分的 adj_train, adj_val, adj_test, features 张量
    data = np.load(args.data_path)
    adj_train = data['adj_train']  # (train_p, n, n)
    adj_val = data['adj_val']  # (val_p, n, n)
    adj_test = data['adj_test']  # (test_p, n, n)
    features = torch.FloatTensor(data['features'])  # (p, n, d), d=1 in this case
    
    # 获取节点数量和特征维度
    n_nodes = adj_train.shape[1]
    feat_dim = features.shape[2]

    # 这里假设在训练时我们使用的是所有训练被试的邻接矩阵和特征
    # 保存原始的训练邻接矩阵，用于后续的评估
    adj_train_orig = torch.FloatTensor(adj_train)  # (train_p, n, n)

    # 设置训练集邻接矩阵为输入
    adj_norm = preprocess_graph_batches(adj_train_orig)  # 对每个被试的邻接矩阵进行归一化

    # 构造训练时的标签矩阵，包括自连接
    #torch.eye是单位矩阵
    #这段代码功能总结：
    # 	•	为每个样本的邻接矩阵添加自连接（单位矩阵）。
    # 	•	将结果转换为浮点张量类型，以便在后续的计算中使用。
    # 这个过程在图神经网络（如 GCN、GAT 等）中很常见，因为添加自连接有助于节点在消息传递过程中能够获取到自身的信息。
    adj_label = adj_train_orig + torch.eye(n_nodes).unsqueeze(0).repeat(adj_train_orig.shape[0], 1, 1)
    adj_label = torch.FloatTensor(adj_label)  # 转换为 PyTorch 的张量

    # 计算正样本的权重，用于在损失函数中平衡样本
    # 代码功能总结
    # •	pos_weight 用于处理类别不平衡，尤其是在存在边的数量远小于不存在边的数量的情况下，提升正样本（存在边）在损失函数中的权重。
    # •	norm 用于标准化损失值，使得模型在训练时可以更加稳定，不受图大小或样本不平衡的影响。
    # 这些参数通常在图神经网络的损失函数（如二元交叉熵）中使用，以便更好地训练图中稀疏连接的关系。
    pos_weight = float(n_nodes * n_nodes - adj_train_orig.sum()) / adj_train_orig.sum()
    norm = n_nodes * n_nodes / float((n_nodes * n_nodes - adj_train_orig.sum()) * 2)

    # 初始化 GCN-VAE 模型，使用输入特征维度、隐藏层维度和 dropout 参数
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 将训练集的特征和邻接矩阵标签转换为 TensorDataset，以便于 DataLoader 加载
    # 假设 features 的维度是 (p, n, d)，adj_norm 和 adj_label 的维度是 (p, n, n)
    train_features = features[:adj_train.shape[0]]  # 选择训练部分的特征
    # 创建 TensorDataset
    dataset = TensorDataset(train_features, adj_norm, adj_label)
    batch_size = 32  # 定义 batch size，可以根据内存大小进行调整
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 开始训练循环
    for epoch in range(args.epochs):
        t = time.time()
        model.train()

        epoch_loss = 0  # 用于累计每一个epoch总损失
        for batch_idx, (batch_features, batch_adj_norm, batch_adj_label) in enumerate(dataloader):
            optimizer.zero_grad()

            # 前向传播，得到重构的邻接矩阵 (recovered) 和编码器输出的均值 (mu) 及对数方差 (logvar)
            # 在这里，我们输入的是每个批次的 features 和对应的 adj_norm
            recovered, mu, logvar = model(batch_features, batch_adj_norm)

            # 计算变分自编码器的损失，包括重构损失和 KL 散度
            loss = loss_function_batches(preds=recovered, labels=batch_adj_label,
                                     mu=mu, logvar=logvar, n_nodes=batch_adj_label.shape[1],
                                     norm=norm, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            # 累加每个批次的损失
            epoch_loss += loss.item()

        # 打印当前 epoch 的平均训练损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_epoch_loss),
          "time=", "{:.5f}".format(time.time() - t))

        # 验证集评估
        val_roc_scores, val_ap_scores = evaluate_model_on_data(
        model,
        features[adj_train.shape[0]:adj_train.shape[0] + adj_val.shape[0]],
        adj_val)
        # 计算均值
        val_avg_roc_score = np.mean(val_roc_scores)
        val_avg_ap_score = np.mean(val_ap_scores)
        print("val_roc=", "{:.5f}".format(val_avg_roc_score), "val_ap=", "{:.5f}".format(val_avg_ap_score))
    print("Optimization Finished!")

    # 测试集评估
    test_roc_scores, test_ap_scores = evaluate_model_on_data(model, features[ adj_val.shape[0]: adj_val.shape[0]+adj_test.shape[0]] , adj_test)
    # 计算均值
    test_avg_roc_score = np.mean(test_roc_scores)
    test_avg_ap_score = np.mean(test_ap_scores)
    print('Test ROC score: ' + str(test_avg_roc_score))
    print('Test AP score: ' + str(test_avg_ap_score))

def evaluate_model_on_data(model, features, adj_input):
    """
    对给定的输入邻接矩阵 adj_input 进行模型评估。
    :param model: 训练好的模型
    :param features: 输入的特征张量 (batch_size, n, feature_dim)
    :param adj_input: 被 mask 处理后的邻接矩阵，用于模型重建 (batch_size, n, n)
    :return: 每个被试的 ROC AUC 和 AP 分数列表
    """
    model.eval()  # 设置模型为评估模式

    # 如果 adj_input 是 NumPy 数组，将其转换为 PyTorch 张量
    if isinstance(adj_input, np.ndarray):
        adj_input = torch.FloatTensor(adj_input)

    # 对输入邻接矩阵进行归一化处理
    adj_norm = preprocess_graph_batches(adj_input)
    # 构造标签矩阵，添加自连接
    adj_label = adj_input + torch.eye(adj_input.shape[1]).unsqueeze(0).repeat(adj_input.shape[0], 1, 1)

    with torch.no_grad():
        # 前向传播得到重构后的邻接矩阵 (recovered) 和潜在表示 (mu)
        recovered, mu, _ = model(features, adj_norm)
    
    # 使用潜在表示 (mu) 进行评估
    emb = mu.data.numpy()

    # 调用 get_roc_score 函数，自动从 adj_input 中生成正负样本边，并对每个被试计算 ROC 和 AP
    #对于rocscores以及apscores的计算方式，可以步入函数中去了解，有一个文档
    roc_scores, ap_scores = get_roc_score_batches(emb, adj_input.numpy())
    
    # 返回每个被试的 ROC AUC 和 AP 分数列表
    return roc_scores, ap_scores



# def gae_for(args):
#     # 打印当前使用的数据集名称
#     print("Using {} dataset".format(args.dataset_str))

#     # 加载图数据的邻接矩阵 (adj) 和节点特征矩阵 (features)
#     adj, features = load_data(args.dataset_str)
#     n_nodes, feat_dim = features.shape  # 获取节点数和节点特征的维度

#     # 保存原始邻接矩阵（用于后续评估），并去掉对角线上的自连接
#     adj_orig = adj
#     adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
#     adj_orig.eliminate_zeros()  # 移除矩阵中的零值

#     # 将图的边分为训练集、验证集和测试集
#     adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
#     adj = adj_train  # 设置训练时使用的邻接矩阵

#     # 对邻接矩阵进行归一化处理，便于输入到模型中
#     adj_norm = preprocess_graph(adj)
#     # 构造训练时的标签矩阵，包括自连接
#     adj_label = adj_train + sp.eye(adj_train.shape[0])
#     # 将标签矩阵转换为 PyTorch 的浮点型张量
#     adj_label = torch.FloatTensor(adj_label.toarray()) 【维度是n*n】

#     # 计算正样本的权重，用于在损失函数中平衡样本
#     pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
#     # 计算归一化因子，用于损失函数中的加权计算
#     norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

#     # 初始化 GCN-VAE 模型，使用输入特征维度、隐藏层维度和 dropout 参数
#     model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
#     # 使用 Adam 优化器，学习率为 args.lr
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     hidden_emb = None  # 用于存储最终的隐藏层表示
#     # 开始训练循环，训练 epochs 次
#     for epoch in range(args.epochs):
#         t = time.time()  # 记录当前时间，用于计算每轮训练的时间

#         # 设置模型为训练模式
#         model.train()
#         # 清除上一步的梯度
#         optimizer.zero_grad()
#         # 前向传播，得到重构的邻接矩阵 (recovered) 和编码器输出的均值 (mu) 及对数方差 (logvar)
#         recovered, mu, logvar = model(features, adj_norm)
#         # 计算变分自编码器的损失，包括重构损失和 KL 散度
#         loss = loss_function(preds=recovered, labels=adj_label,
#                              mu=mu, logvar=logvar, n_nodes=n_nodes,
#                              norm=norm, pos_weight=pos_weight)
#         # 反向传播，计算梯度
#         loss.backward()
#         # 获取当前的损失值
#         cur_loss = loss.item()
#         # 更新模型参数
#         optimizer.step()

#         # 将隐空间的均值表示转换为 numpy 数组，便于后续评估
#         hidden_emb = mu.data.numpy()
#         # 使用验证集计算当前隐空间表示的 ROC AUC 和 AP，用于模型性能评估
#         roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

#         # 打印当前 epoch 的训练损失、验证集 AP 和训练时间
#         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
#               "val_ap=", "{:.5f}".format(ap_curr),
#               "time=", "{:.5f}".format(time.time() - t)
#               )

#     # 输出训练完成的提示信息
#     print("Optimization Finished!")

#     # 在测试集上评估模型，计算测试集的 ROC AUC 和 AP 分数
#     roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
#     # 打印测试集的 ROC 和 AP 评估结果
#     print('Test ROC score: ' + str(roc_score))
#     print('Test AP score: ' + str(ap_score))

if __name__ == '__main__':
    #load_and_save_data("/root/autodl-tmp/.autodl/Y68p_FC_2_0.npy",)
    gae_for(args)

# 问题1：我的疑问是这个模型输出的结果是什么？是一个重新生成的邻接矩阵嘛？
# 是的，这个模型的输出之一是一个重新生成的邻接矩阵，具体来说，它通过编码-解码过程对图的结构进行重构。模型输出的结果包括以下几个部分：

# 1. 重构的邻接矩阵 (recovered)

# 	•	recovered 是模型的主要输出之一。它表示模型在编码-解码过程中重构出来的邻接矩阵。这个矩阵包含了节点之间的重建关系，并用概率值来表示每对节点之间是否有连接。
# 	•	通过这个重构的邻接矩阵，可以估计图中节点之间的关系，比如预测未观察到的边或修正原有的边。

# 2. 隐空间表示 (mu)

# 	•	另一个重要输出是 mu，即在变分自编码器的编码器部分生成的隐空间表示。
# 	•	mu 是每个节点的低维表示（或嵌入），它包含了原始图的结构和特征信息的压缩表示。这些嵌入通常用于下游任务，例如节点分类、图聚类、边预测等。
# 	•	在代码中，这些嵌入被用来计算 ROC AUC 和 AP 分数，用于验证集和测试集的评估。

# 3. 对结果的应用

# 	•	通过 recovered，我们能够在测试集上评估模型的性能，判断其对图结构重建的准确性。这是通过计算与真实邻接矩阵（如 adj_orig）的差异来完成的。
# 	•	模型在验证集上的表现通过 ROC AUC 和 AP 指标来评估，测试集上的表现也会同样使用这些指标。
# 	•	ROC AUC（Receiver Operating Characteristic Area Under Curve）衡量模型区分正负样本（连接和非连接）的能力。
# 	•	AP（Average Precision）则是模型在排序任务中的表现。

# 总结

# 	•	模型输出的 recovered 是一个近似于原始邻接矩阵的重建矩阵，表示的是预测的节点连接关系。
# 	•	mu 是节点的低维嵌入表示，能够捕捉图中节点之间的关系。
# 	•	这些输出可以用来进一步分析图的结构，或在图分析任务（如链路预测）中使用。

# 所以，严格来说，模型输出的主要是隐空间表示 (mu)，而 recovered 是这个隐空间通过解码器转换后的邻接矩阵近似值。
    

# 问题2:将图的边分为训练集、验证集和测试集是什么意思
# 将图的边分为训练集、验证集和测试集是图机器学习中的一种数据划分方法，它类似于在常规机器学习任务中划分训练集、验证集和测试集。具体而言，这种划分方法主要用于链路预测任务，即预测图中哪些节点之间存在连接（边）。以下是更详细的解释：

# 1. 什么是边的划分？

# 在图神经网络任务中，图由节点和边组成。划分边的主要目的是通过部分已知的边信息来训练模型，并使用剩余部分来验证模型性能和进行测试。划分过程如下：

# 	•	训练集（Training Set）:
# 	•	训练集中包含的边用于训练模型，即用于学习图中节点之间的关系。
# 	•	在训练时，模型只看到训练集中存在的边（即连接关系），并根据这些信息调整自身的参数。
# 	•	训练集中还可能包含一些负样本，即图中不连接的节点对，这样模型可以学习到哪些节点不应该连接。
# 	•	验证集（Validation Set）:
# 	•	验证集中的边用于模型在训练过程中的性能评估和调参。
# 	•	验证集中的边（正样本）是从原图中去掉的一部分已知连接的边，用于评估模型对未见边的预测能力。
# 	•	同时，也会生成与验证集中正样本数量相等的负样本（即不连接的节点对），用于进一步评估模型。
# 	•	通过验证集可以评估模型在未见连接上的预测能力，并可以用来调整模型的超参数。
# 	•	测试集（Test Set）:
# 	•	测试集是为了最终评估模型的表现，即评估模型在完全未见过的边上的预测能力。
# 	•	测试集中的正样本和负样本同样是从原图中剔除的一部分边和生成的不连接节点对。
# 	•	测试集用于衡量模型泛化性能，帮助判断模型是否能准确地预测未见的连接关系。

# 2. 为什么要进行这样的划分？

# 这种划分的主要目的是评估模型的预测能力，尤其是在以下任务中：

# 	•	链路预测（Link Prediction）:
# 	•	目标是预测图中哪些节点对之间应该有连接，或者判断某个给定的节点对是否存在连接。
# 	•	通过将一部分边分配到验证集和测试集，可以模拟真实场景下预测未知边的过程。
# 	•	防止过拟合:
# 	•	如果模型在训练时看到了所有的边，那么它可能会过拟合到训练数据，导致无法在未见的节点对上做出有效预测。
# 	•	通过划分数据集，可以使得模型只通过训练集中的部分信息进行学习，然后通过验证集和测试集检测其泛化能力。

# 3. 具体的划分方法

# 在代码中，通常通过以下步骤进行边的划分：

# 	1.	随机选择一部分边作为训练集:
# 	•	这些边用于模型的训练。
# 	•	剩余的边用于验证和测试。
# 	2.	从剩余边中随机选择一部分作为验证集:
# 	•	确保训练集中不包含这些边。
# 	•	同时生成等数量的负样本（不存在连接的节点对）。
# 	3.	剩下的边作为测试集:
# 	•	同样生成等数量的负样本，用于测试模型的预测能力。

# 4. 实际应用中的例子

# 假设我们有一个社交网络图，节点表示用户，边表示用户之间的好友关系：

# 	•	训练集: 包含图中80%的好友关系，用于训练模型。
# 	•	验证集: 包含10%的好友关系（正样本）以及等量的非好友关系（负样本），用于评估模型在预测好友关系时的表现。
# 	•	测试集: 包含剩下的10%的好友关系（正样本）以及等量的非好友关系（负样本），用于最终评估模型的预测能力。

# 通过这样的划分，我们可以训练一个模型来预测哪些用户之间是好友（链接预测任务）。在模型评估时，我们可以使用验证集和测试集中的未见边，来判断模型能否准确预测用户之间的好友关系。

# 总结

# 将图的边分为训练集、验证集和测试集，是为了模拟预测图中未见连接的过程，并防止模型过拟合。这种划分方式广泛应用于图神经网络的训练中，特别是在链路预测等任务中，帮助模型通过部分连接关系的学习来预测未见的节点对是否有连接。
    
# 3.问题3:但我这个任务应该是训练集用完整的邻接矩阵去生成完整的邻接矩阵，验证集随机修改n条边的连接方式（先只考虑简单情况，从连接到不连接），测试集又是随机修改n条边的连接方式。那我的数据加载方式是不是应该进行修改？
# 在这种任务设置下，由于训练集需要使用完整的邻接矩阵，而验证集和测试集涉及随机修改边的连接方式，因此数据的加载和划分方式需要进行调整。你可以按照以下思路进行修改：

# 1. 训练集的加载

# 	•	使用完整的邻接矩阵进行训练，模型将学习所有节点之间的连接关系。这意味着不需要像之前那样划分训练集的边。
# 	•	在加载数据时，只需加载整个邻接矩阵即可。

# 2. 生成验证集和测试集

# 	•	验证集和测试集不再从原始数据中划分，而是通过随机修改原始邻接矩阵中的连接关系生成。具体步骤如下：
# 	•	复制完整的邻接矩阵。
# 	•	随机选择  n  条连接的边，将它们的连接状态从“连接”修改为“不连接”（即将相应位置的值从1改为0）。
# 	•	这种操作可以对验证集和测试集分别进行，确保二者使用不同的随机选择。
