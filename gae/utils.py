import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import os
from sklearn.metrics import roc_auc_score, average_precision_score

#写于2024年10月10日，用于计算个体化的正样本权重和归一化参数
def calculate_posweights_per_subject(adj_train_orig):
    """
    为每个被试计算 pos_weight 和 norm。

    参数:
    adj_train_orig (numpy.ndarray): 形状为 (p, n, n) 的张量，其中 p 是被试数量，n 是节点数量。

    返回:
    pos_weights (list): 每个被试的 pos_weight 列表。
    norms (list): 每个被试的 norm 列表。
    """
    p = adj_train_orig.shape[0]  # 被试数量
    n_nodes = adj_train_orig.shape[1]

    # 初始化每个被试的 pos_weight 和 norm 的列表
    pos_weights = []
    norms = []

    for i in range(p):
        # 对每个被试单独计算正样本数和负样本数
        positive = adj_train_orig[i].sum()  # 当前被试的正样本数（连接的边数）
        negative = n_nodes * n_nodes - positive  # 当前被试的负样本数（所有连接对数 - 正样本数）

        # 计算当前被试的 pos_weight 和 norm
        pos_weight = float(negative) / positive if positive > 0 else 0.0
        norm = n_nodes * n_nodes / float(negative * 2) if negative > 0 else 0.0

        pos_weights.append(pos_weight)
        norms.append(norm)

    return pos_weights, norms
# 写于2024年10月8日，用于代替原有的 load_data 函数
import numpy as np
import os

def load_and_save_data(file_path, threshold=0.2, train_ratio=0.8, val_ratio=0.1, save_path="/root/autodl-tmp/.autodl/", save_name="data.npz"):
    """
    加载邻接矩阵数据，生成特征张量，并划分训练集、验证集和测试集，保存结果为 .npz 文件。

    :param file_path: str, npy 文件路径，读取形状为 (p, n, n) 的邻接矩阵张量。
    :param threshold: float, 阈值，用于将输入的邻接矩阵张量转换为 0-1 二值矩阵。
    :param train_ratio: float, 训练集占比，默认 80%。
    :param val_ratio: float, 验证集占比，默认 10%。
    :param save_path: str, 保存路径，生成的文件将存储为 .npz 格式。
    :param save_name: str, 保存的文件名，默认 "data.npz"。

    :return: 无，生成的张量保存在指定路径。
    """
    # 加载邻接矩阵张量
    adj_tensor = np.load(file_path)  # 形状为 (p, n, n)
    p, n, _ = adj_tensor.shape
    
    # 将邻接矩阵张量二值化，根据阈值将元素转换为 0 或 1
    adj_tensor = (adj_tensor >= threshold).astype(int)
    
    # 生成节点特征张量，形状为 (p, n, 1)，每个被试的每个节点特征都为 1
    features_tensor = np.ones((p, n, 1), dtype=np.float32)
    
    # 划分训练集、验证集和测试集
    num_train = int(train_ratio * p)
    num_val = int(val_ratio * p)
    num_test = p - num_train - num_val

    train_idx = range(num_train)
    val_idx = range(num_train, num_train + num_val)
    test_idx = range(num_train + num_val, p)
    
    # 获取训练集、验证集、测试集的邻接矩阵张量
    adj_train = adj_tensor[train_idx, :, :]
    adj_val = adj_tensor[val_idx, :, :]
    adj_test = adj_tensor[test_idx, :, :]
    
    # 保存路径如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 保存结果为 .npz 文件
    save_file = os.path.join(save_path, save_name)
    np.savez_compressed(save_file, 
                        adj_train=adj_train, 
                        adj_val=adj_val, 
                        adj_test=adj_test, 
                        features=features_tensor)
    
    print(f"Data saved to {save_file}.")
    print(f"Train set: {adj_train.shape}, Validation set: {adj_val.shape}, Test set: {adj_test.shape}")
    print(f"Features tensor shape: {features_tensor.shape}")

# 示例调用：
# load_and_save_data("data/adj_tensor.npy", threshold=0.2, save_path="output/", save_name="my_data.npz")


def process_data(file_path, threshold=0.2, output_dir="data", dataset_name="custom"):
    # 1. 加载 (p, n, n) 张量，其中 p 是被试的数量，n 是节点数
    data = np.load(file_path)
    p, n, _ = data.shape

    # 2. 初始化特征矩阵和图结构
    # 使用全 1 矩阵作为特征矩阵，每个节点的特征都为 1
    x = sp.lil_matrix(np.ones((n, n)))
    allx = x.copy()  # allx 和 x 相同，表示训练集和部分验证集的特征

    # 初始化图结构为一个空字典
    graph = {}

    # 3. 对每个被试的 (n, n) 相关矩阵应用阈值，生成二值邻接矩阵
    for subject in range(p):
        # 取出第 subject 个被试的 (n, n) 相关矩阵
        corr_matrix = data[subject]

        # 应用阈值，将相关系数大于等于阈值的位置标记为 1，其余为 0
        binary_matrix = (corr_matrix >= threshold).astype(int)

        # 将二值矩阵转化为邻接表形式
        graph[subject] = {i: list(np.where(binary_matrix[i])[0]) for i in range(n)}

    # 4. 创建测试集的索引文件
    test_idx_reorder = list(range(n))  # 假设所有节点都是测试集节点

    # 将测试集索引排序
    test_idx_range = np.sort(test_idx_reorder)

    # 5. 保存文件到指定路径，格式与所需的加载函数匹配
    output_files = {
        'x': x,
        'tx': x,  # tx 和 x 相同，因为我们用相同的特征矩阵表示训练和测试节点
        'allx': allx,
        'graph': graph
    }
    
    # 为每个文件保存数据
    for name, obj in output_files.items():
        with open(f"{output_dir}/ind.{dataset_name}.{name}", 'wb') as f:
            pkl.dump(obj, f)

    # 保存测试集索引文件
    with open(f"{output_dir}/ind.{dataset_name}.test.index", 'w') as f:
        for idx in test_idx_reorder:
            f.write(f"{idx}\n")

    print(f"Data processed and saved to '{output_dir}' for dataset '{dataset_name}'.")

# 调用示例
#process_data("/root/autodl-tmp/.autodl/Y68p_FC_2_0.npy", threshold=0.2)

def load_data(dataset):
    # 加载数据：x, tx, allx, graph（分别表示训练集、测试集、完整的特征集和图结构）
    names = ['x', 'tx', 'allx', 'graph']
    objects = []  # 用于存储加载的对象

    # 依次加载 x, tx, allx, graph 文件
    for i in range(len(names)):
        '''
        解决 Python 2 和 Python 3 在 pickle 序列化/反序列化 numpy 数组时的兼容性问题
        参考：https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)  # 创建一个 _Unpickler 对象
            u.encoding = 'latin1'   # 设置解码方式为 'latin1' 以解决兼容性问题
            cur_data = u.load()      # 加载数据
            objects.append(cur_data) # 将加载的数据存入 objects 列表

    # 解包 objects 列表，分别赋值给 x, tx, allx, graph
    x, tx, allx, graph = tuple(objects)
    
    # 调用 parse_index_file 函数，加载测试集索引
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    
    # 对测试集的索引进行排序
    test_idx_range = np.sort(test_idx_reorder)

    # 如果数据集是 'citeseer'，需要特殊处理孤立节点
    if dataset == 'citeseer':
        # Citeseer 数据集中有些节点是孤立的（没有连接到图中的其他节点）
        # 因此我们需要找到这些孤立节点，并将它们作为全零向量添加到特征矩阵中
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        
        # 创建一个新的矩阵 tx_extended，其行数等于完整测试集索引的范围长度
        # 列数与原始特征矩阵 x 的列数相同
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        
        # 将原 tx 中的值复制到 tx_extended 中的相应位置
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        
        # 将 tx 替换为 tx_extended
        tx = tx_extended

    # 使用 scipy 的 vstack 函数将 allx 和 tx 垂直堆叠，形成完整的特征矩阵
    features = sp.vstack((allx, tx)).tolil()
    
    # 重新排列特征矩阵，使其顺序与 test_idx_range 对应
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # 将稀疏矩阵转换为稠密矩阵，并将其转换为 PyTorch 的 FloatTensor 类型
    features = torch.FloatTensor(np.array(features.todense()))
    
    # 根据邻接表（graph）构建邻接矩阵 adj
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # 返回邻接矩阵 adj 和特征矩阵 features
    return adj, features

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# 这个函数的确是**输入一个邻接矩阵**，然后生成了一些**与原始邻接矩阵相关的子集**，但并没有直接生成多个完整的邻接矩阵。它主要做的是将**输入的邻接矩阵划分为不同的数据集**形式，具体来说，生成了一个**训练集的邻接矩阵**以及一些**边的列表**，这些边列表表示验证集和测试集的连接关系。

# ### 输出内容的详细解释
# 1. **输入**:
#    - 一个稀疏表示的邻接矩阵 `adj`，它描述了图的连接结构。

# 2. **输出**:
#    - **`adj_train`**: 这是一个新的邻接矩阵，只包含训练集的边（正样本）。它是根据 `train_edges` 构建的矩阵。
#      - 这个矩阵代表的是在模型训练过程中使用的连接关系。
#      - 它是通过去除测试集和验证集中的边，并只保留训练集中边的连接关系构造出来的。

#    - **`train_edges`**: 这是训练集的边的列表，表示哪些节点对之间有连接。
#    - **`val_edges` 和 `val_edges_false`**: 这是验证集中边的正样本和负样本，分别表示哪些节点对之间有连接、哪些节点对之间没有连接。
#    - **`test_edges` 和 `test_edges_false`**: 这是测试集中边的正样本和负样本，类似地，分别表示哪些节点对之间有连接、哪些节点对之间没有连接。

# ### 总结：这个函数生成了哪些数据？
# - 这个函数的主要作用是**将输入的邻接矩阵拆分为不同的边集**，从而得到：
#   - **一个新的邻接矩阵 `adj_train`**，它只包含用于训练的边的连接关系。
#   - **验证集和测试集的边列表**（正样本和负样本），用于模型评估。

# - **它并没有直接生成多个完整的邻接矩阵**，而是通过在原始邻接矩阵中删除或排除部分边来生成一个训练集的邻接矩阵，并通过列表形式记录验证集和测试集的连接。

# 因此，如果你的任务是要生成多个不同的邻接矩阵，那么这个函数的逻辑并不完全满足你的需求。它更适合于那些需要将图划分为训练、验证和测试数据集，并生成部分负样本用于链路预测任务的场景。
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def preprocess_graph_batches(adj_batch):
    """
    对批量邻接矩阵 adj_batch 进行归一化处理，并返回 PyTorch 稠密张量。
    :param adj_batch: 形状为 (batch_size, n, n) 的邻接矩阵张量
    :return: 归一化后的邻接矩阵，形状为 (batch_size, n, n) 的 PyTorch 稠密张量
    """
    batch_size = adj_batch.shape[0]
    n = adj_batch.shape[1]

    adj_normalized_list = []

    for i in range(batch_size):
        # 对第 i 个邻接矩阵进行处理
        adj = sp.coo_matrix(adj_batch[i])
        adj_ = adj + sp.eye(adj.shape[0])  # 添加自连接

        # 计算度矩阵的逆平方根
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())

        # 进行归一化：D^(-1/2) * A * D^(-1/2)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        # 转换为稠密矩阵并添加到列表中
        adj_normalized_dense = torch.FloatTensor(adj_normalized.toarray())
        adj_normalized_list.append(adj_normalized_dense)

    # 将所有的稠密张量堆叠成一个形状为 (batch_size, n, n) 的张量
    adj_normalized_tensor = torch.stack(adj_normalized_list, dim=0)

    return adj_normalized_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_batches(emb, adj_orig):
    """
    计算每个被试的 ROC AUC 和 AP 分数，并返回每个被试的结果。
    :param emb: 批量的潜在表示 (batch_size, n, hidden_dim)
    :param adj_orig: 原始邻接矩阵 (batch_size, n, n)
    :return: 每个被试的 ROC AUC 和 AP 分数列表
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    batch_size = emb.shape[0]
    roc_scores = []
    ap_scores = []

    # 对每个被试的图分别计算 ROC AUC 和 AP
    for i in range(batch_size):
        # 计算第 i 个图的重建邻接矩阵
        adj_rec = np.dot(emb[i], emb[i].T)
        
        # 从原始邻接矩阵中获取正样本和负样本的边
        #edges_pos的最终输出维度是（m,2），每一行表示i存在一个连接的边的位置，包含其行列索引
        edges_pos = np.array(np.where(adj_orig[i] == 1)).T  # 正样本
        edges_neg = np.array(np.where(adj_orig[i] == 0)).T  # 负样本
        
        # 对正样本边进行预测
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))#这里是找到之前有连接的边，然后看预测结果，存在preds里
            pos.append(adj_orig[i, e[0], e[1]]) #i是对应第i个被试的标签
        
        # 对负样本边进行预测
        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[i, e[0], e[1]])
        
        # 合并正负样本的预测值和标签
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        
        # 计算 ROC AUC 和 AP 分数
        #roc_auc_score 是用于计算ROC曲线下面积（AUC，Area Under the Curve）的指标。它常用于二分类任务中，用来评估分类模型的性能。
        #概念解释：
        #•ROC曲线（Receiver Operating Characteristic Curve）：是一条通过改变分类阈值得到的图形，横轴是 假阳性率 (FPR: False Positive Rate)，纵轴是 真阳性率 (TPR: True Positive Rate)。
            #•AUC (Area Under the Curve)：是ROC曲线下方的面积。AUC的取值范围在0和1之间。
            #•AUC=1：表示模型完美分类。
            #•AUC=0.5：表示模型分类没有区分能力（类似随机猜测）。
            #•AUC<0.5：表示模型的分类效果比随机猜测还差。
        #roc_auc_score 的用法：
            #在Python中，roc_auc_score 通常在 scikit-learn 中使用。使用时需要两个主要输入参数：
                #•y_true：真实标签（0或1）。
                #•y_score：模型输出的预测概率（或决策函数的得分）。

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        
        # 将每个被试的结果存储
        roc_scores.append(roc_score)
        ap_scores.append(ap_score)
    
    # 返回每个被试的 ROC AUC 和 AP 分数列表
    return roc_scores, ap_scores

def get_roc_score_batches_accelerated(emb, adj_orig):
    """
    计算每个被试的 ROC AUC 和 AP 分数，并返回每个被试的结果。
    :param emb: 批量的潜在表示 (batch_size, n, hidden_dim)
    :param adj_orig: 原始邻接矩阵 (batch_size, n, n)
    :return: 每个被试的 ROC AUC 和 AP 分数列表
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    batch_size = emb.shape[0]
    roc_scores = []
    ap_scores = []

    # 对每个被试的图分别计算 ROC AUC 和 AP
    for i in range(batch_size):
        # 计算第 i 个图的重建邻接矩阵
        adj_rec = np.dot(emb[i], emb[i].T)
        
        # 获取正样本和随机采样的负样本的边
        edges_pos = np.array(np.where(adj_orig[i] == 1)).T  # 正样本
        edges_neg = np.array(np.where(adj_orig[i] == 0)).T  # 负样本

        # 对正样本和负样本进行批量预测
        pos_indices = tuple(edges_pos.T)
        preds_pos = sigmoid(adj_rec[pos_indices])
        
        num_neg_samples = min(len(edges_pos), len(edges_neg))
        sampled_neg_indices = edges_neg[np.random.choice(len(edges_neg), size=num_neg_samples, replace=False)]
        neg_indices = tuple(sampled_neg_indices.T)
        preds_neg = sigmoid(adj_rec[neg_indices])

        # 合并正负样本的预测值和标签
        preds_all = np.concatenate([preds_pos, preds_neg])
        labels_all = np.concatenate([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        
        # 计算 ROC AUC 和 AP 分数
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        
        # 存储每个被试的结果
        roc_scores.append(roc_score)
        ap_scores.append(ap_score)
    
    return roc_scores, ap_scores

# 问题1:mask_test_edges中验证集和测试集是怎么样的？
# 1. 验证集和测试集的正样本（val_edges 和 test_edges）

# 	•	正样本：验证集和测试集的正样本是从原始邻接矩阵 adj 中存在的边中随机选取的。这些边表示图中实际存在的连接关系。
# 	•	具体流程：
# 	•	将图中的所有边（存在连接的节点对）随机打乱。
# 	•	从中选取一部分边（10%）作为测试集的正样本，再选取一部分边（5%）作为验证集的正样本。
# 	•	这些边分别存储在 test_edges 和 val_edges 中。
# 2. 验证集和测试集的负样本（val_edges_false 和 test_edges_false）

# 	•	负样本：负样本是指图中不存在的边（即图中实际不相连的节点对）。这些负样本用于模型在验证和测试时进行对比，以评估模型在未见的边上的预测能力。
# 	•	具体流程：
# 	•	随机选择一对节点 (idx_i, idx_j)，确保它们在原始图中没有连接。
# 	•	生成与正样本数量相同的负样本，这样验证集和测试集中的正负样本数量是平衡的。
# 	•	这些不存在的边分别存储在 test_edges_false 和 val_edges_false 中。
# 3. 验证集和测试集的用途

# 	•	验证集：
# 	•	正样本 (val_edges)：表示图中实际存在的边，模型需要预测这些边是否存在。
# 	•	负样本 (val_edges_false)：表示图中实际不存在的边，模型需要预测这些边是否不存在。
# 	•	验证集用于在模型训练过程中调节超参数和监控模型性能。模型通过在验证集上的预测结果，可以得出当前模型在未见过的边上的表现。
# 	•	测试集：
# 	•	正样本 (test_edges)：表示图中实际存在的边，模型需要在测试阶段预测这些边是否存在。
# 	•	负样本 (test_edges_false)：表示图中实际不存在的边，模型需要在测试阶段预测这些边是否不存在。
# 	•	测试集用于在模型训练结束后，对模型的最终性能进行评估。测试集的结果是对模型泛化能力的真正考验。

# 4. 验证集和测试集的形式

# 	•	验证集和测试集并不是单独的邻接矩阵形式，而是正样本和负样本的边列表形式。
# 	•	正样本是验证集或测试集中从原始邻接矩阵中选出的连接节点对（这些是实际存在的边）。
# 	•	负样本是随机生成的在原始邻接矩阵中不存在的连接节点对（这些是实际不存在的边）。
# 	•	这些正负样本的边列表用于评估模型在特定边上的预测准确度。

# 5. 示例

# 假设输入图有 100 条边，随机选择了 10 条边作为测试集，5 条边作为验证集，那么：

# 	•	test_edges 中会有 10 条实际存在的边，test_edges_false 中会有 10 条实际不存在的边。
# 	•	val_edges 中会有 5 条实际存在的边，val_edges_false 中会有 5 条实际不存在的边。

# 在模型评估时，会通过这些正样本和负样本计算模型的精确率、召回率、ROC AUC、平均精度等指标，以衡量模型在链路预测任务上的表现。

# 总结

# 	•	验证集和测试集不是邻接矩阵的形式，而是通过正样本和负样本的边列表来表示的。
# 	•	它们从输入的邻接矩阵中随机选择部分边作为正样本，并随机生成同等数量的负样本。
# 	•	这种设置用于在验证和测试阶段评估模型的边预测能力，特别是在链路预测任务中，模型需要区分哪些边是存在的（正样本），哪些边是不存在的（负样本）。