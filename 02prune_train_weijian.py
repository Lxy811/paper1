from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import copy
from sklearn import metrics

from maskrcnn_benchmark.data.datasets.visual_genome import load_graphs, load_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors, rel_edge_vectors

import logging
# 设置日志配置
logging.basicConfig(filename='./prune_ckpt/log.txt', level=logging.INFO)


# 计数融合模块，，用于融合两个输入特征
class CountFusion(nn.Module):
    def __init__(self, dim_x, dim_y, output_dim=512):
        # 调用父类的构造函数
        super(CountFusion, self).__init__()
        # 定义线性层，将输入x映射到output_dim维度
        self.dense_x = nn.Linear(dim_x, output_dim)
        # 定义线性层，将输入y映射到output_dim维度
        self.dense_y = nn.Linear(dim_y, output_dim)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # 对输入x进行线性变换
        x1 = self.dense_x(x)
        # 对输入y进行线性变换
        y1 = self.dense_y(y)
        # 对x1和y1的和应用ReLU激活函数
        item1 = self.relu(x1 + y1)
        # 计算x1和y1的差的平方
        item2 = (x1 - y1) * (x1 - y1)
        # 返回最终融合结果
        return item1 - item2

# 关系剪枝模型，用于预测关系是否存在合理
class RelationPrune(nn.Module):
    def __init__(self, obj_embed_vecs, rel_embed_vecs, num_objs=151, num_rels=51, embed_dim=200, hidden_dim=512):
        # 调用父类的构造函数
        super(RelationPrune, self).__init__()
        # 目标类别的数量
        self.num_obj_cls = num_objs
        # 关系类别的数量
        self.num_rel_cls = num_rels
        # 嵌入维度
        self.embed_dim = embed_dim
        # 隐藏层维度
        self.hidden_dim = hidden_dim
        # 主语嵌入层
        self.sbj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # 宾语嵌入层
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # 关系嵌入层
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        # 在不计算梯度的情况下，将预训练的嵌入向量复制到嵌入层中
        with torch.no_grad():
            self.sbj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
        
        # 主语和宾语的计数融合模块
        self.cnt_fusion_so = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        # 主语和关系的计数融合模块
        self.cnt_fusion_sr = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        # 宾语和关系的计数融合模块
        self.cnt_fusion_or = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)
        # 主语的线性层
        self.dense_s = nn.Linear(embed_dim, hidden_dim)
        # 宾语的线性层
        self.dense_o = nn.Linear(embed_dim, hidden_dim)
        # 关系的线性层
        self.dense_r = nn.Linear(embed_dim, hidden_dim)
        # 投影层，用于将特征映射到一维输出
        self.project = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, sbj_token, obj_token, rel_token):
        # 获取主语的嵌入向量
        sbj_embed = self.sbj_embed(sbj_token)
        # 获取宾语的嵌入向量
        obj_embed = self.sbj_embed(obj_token)
        # 获取关系的嵌入向量
        rel_embed = self.rel_embed(rel_token)
        # 对主语和宾语的嵌入向量进行计数融合
        fused_so = self.cnt_fusion_so(sbj_embed, obj_embed)
        # 对主语和关系的嵌入向量进行计数融合
        fused_sr = self.cnt_fusion_sr(sbj_embed, rel_embed)
        # 对宾语和关系的嵌入向量进行计数融合
        fused_or = self.cnt_fusion_or(obj_embed, rel_embed)

        # 对主语的嵌入向量进行线性变换
        proj_s = self.dense_s(sbj_embed)
        # 对宾语的嵌入向量进行线性变换
        proj_o = self.dense_o(obj_embed)
        # 对关系的嵌入向量进行线性变换
        proj_r = self.dense_r(rel_embed)

        # 对融合结果和投影结果应用sigmoid激活函数
        fused_so, fused_sr, fused_or = torch.sigmoid(fused_so), torch.sigmoid(fused_sr), torch.sigmoid(fused_or)
        proj_r, proj_o, proj_s = torch.sigmoid(proj_r), torch.sigmoid(proj_o), torch.sigmoid(proj_s)

        # 计算不同的融合结果
        act_sor = fused_so * proj_r
        act_sro = fused_sr * proj_o
        act_ors = fused_or * proj_s

        # 将融合结果在最后一个维度上拼接
        concat = torch.cat((act_sor, act_sro, act_ors), dim=-1)
        # 通过投影层得到预测结果，并应用sigmoid激活函数
        logit = torch.sigmoid(self.project(concat))
        return logit

# 关系数据集类，用于创建关系数据的未见空间
class RelationData(Dataset):
    def __init__(self, seen_triplets, num_objs=151, num_rels=51):
        # 生成目标索引列表
        obj_idx_list = np.arange(1, num_objs)
        # 生成关系索引列表
        rel_idx_list = np.arange(1, num_rels)

        # 生成主语维度的索引
        sbj_dim = np.repeat(obj_idx_list, num_objs - 1)
        # 调整目标索引列表的形状
        un_sqz_obj_idx_list = obj_idx_list.reshape(-1, 1)
        # 调整关系索引列表的形状
        un_sqz_rel_idx_list = rel_idx_list.reshape(-1, 1)
        # 生成宾语维度的索引
        obj_dim = np.repeat(un_sqz_obj_idx_list, num_objs - 1, axis=1).T.reshape(-1)
        # 堆叠主语和宾语的索引
        sbj_obj = np.stack((sbj_dim, obj_dim), axis=0)

        # 重复主语和宾语的索引
        so_dim = np.repeat(sbj_obj, num_rels - 1, axis=1)
        # 生成关系维度的索引
        rel_dim = np.repeat(un_sqz_rel_idx_list, sbj_obj.shape[1], axis=1).T.reshape(1, -1)
        # 拼接生成所有可能的三元组
        sor_list = np.concatenate((so_dim, rel_dim), axis=0).T
        # 将三元组转换为torch张量
        self.compose_space = torch.tensor(sor_list, dtype=torch.long)
        # 生成标签
        self.labels = self.gen_label(seen_triplets)

    def gen_label(self, seen_triplets):
        labels = []
        # 遍历所有三元组，显示进度条
        for i in tqdm.tqdm(range(self.compose_space.shape[0])):
            item = self.compose_space[i, :]
            tpt = (int(item[0]), int(item[1]), int(item[2]))
            # 如果三元组在已见三元组集合中，标签为1，否则为0
            if tpt in seen_triplets:
                labels.append(1)
            else:
                labels.append(0)
        # 将标签转换为torch张量
        labels = torch.tensor(labels, dtype=torch.float)
        return labels

    def __getitem__(self, index):
        item = self.compose_space[index, :]
        y = self.labels[index]
        return item[0], item[1], item[2], y

    def __len__(self):
        return self.compose_space.shape[0]

# 测试模型的函数
def run_test(model, dataloader, ep):
    # 将模型设置为评估模式
    model.eval()
    y_preds, y_labels = [], []
    # 初始化预测结果矩阵
    matrix = torch.zeros((151, 151, 51))
    for token_s, token_o, token_r, y_batch in dataloader:
        # 将数据移动到GPU上
        token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()
        # 在不计算梯度的情况下进行预测
        with torch.no_grad():
            y_pred = model(token_s, token_o, token_r)
        y_preds.append(y_pred)
        y_labels.append(y_batch)

        # 将数据移回CPU
        token_s, token_o, token_r = token_s.cpu(), token_o.cpu(), token_r.cpu()
        y_pred = y_pred.cpu()
        # 将预测结果存储到矩阵中
        matrix[token_s, token_o, token_r] = y_pred.reshape(-1)

    # 将预测结果和真实标签拼接成数组
    y_preds = torch.cat(y_preds, dim=0).detach().cpu().numpy()
    y_labels = torch.cat(y_labels, dim=0).numpy()
    # 复制预测结果
    y_preds_prob = copy.deepcopy(y_preds)
    y_preds_hard = copy.deepcopy(y_preds)
    # 将预测概率大于0.5的设置为1，小于0.5的设置为0
    y_preds_hard[y_preds > 0.5] = 1
    y_preds_hard[y_preds < 0.5] = 0
    y_preds_hard = y_preds_hard.astype(int)
    # 计算召回率
    recall = metrics.recall_score(y_labels, y_preds_hard)
    # 计算精确率
    precision = metrics.precision_score(y_labels, y_preds_hard)
    # 计算ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_preds_prob)
    # 计算AUC值
    auc = metrics.auc(fpr, tpr)
    # 记录日志
    logging.info("ep: {}, recall: {:.3f}, precision: {:.3f}, auc: {:.3f}".format(ep, recall, precision, auc))
    print("ep: {}, recall: {:.3f}, precision: {:.3f}, auc: {:.3f}".format(ep, recall, precision, auc))
    # 将模型设置为训练模式
    model.train()
    return matrix, auc

# 训练模型的函数
def train(model, dataloader, test_loader, epoch, lr, save_dir, clip_num=5, pi=0.0):
    # 使用Adam优化器
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    # 定义二元交叉熵损失函数
    criterion = nn.BCELoss()
    # 将模型移动到GPU上
    model = model.cuda()
    best_auc = 0.0
    for ep in range(epoch):
        for step, (token_s, token_o, token_r, y_batch) in enumerate(dataloader):
            # 梯度清零
            opt.zero_grad()
            # 将数据移动到GPU上
            token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()
            y_batch = y_batch.cuda()
            # 进行预测
            y_pred = model(token_s, token_o, token_r)
            # 正样本掩码
            pos_mask = y_batch == 1
            # 负样本掩码
            neg_mask = y_batch == 0
            # 正样本伪标签
            pf_label = y_batch[pos_mask]
            pf_label[pf_label == 1] = 0
            if torch.sum(pos_mask) != 0:
                # 正样本的真实标签损失
                pt_loss = criterion(y_pred[pos_mask].reshape(-1), y_batch[pos_mask])
                # 正样本的伪标签损失
                pf_loss = criterion(y_pred[pos_mask].reshape(-1), pf_label)
            else:
                pt_loss = pf_loss = 0
            if torch.sum(neg_mask) != 0:
                # 负样本的损失
                ng_loss = criterion(y_pred[neg_mask].reshape(-1), y_batch[neg_mask])
            else:
                ng_loss = 0
            # 计算损失项1
            item1 = pi * pt_loss
            # 计算损失项2
            item2 = ng_loss - pi * pf_loss
            item2[item2 < 0] = 0
            # 计算总损失
            loss = item1 + item2
            # 反向传播
            loss.backward()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)
            # 更新参数
            opt.step()
            if step % 30 == 0:
                # 记录日志
                logging.info("Ep: {}, step: {}, loss: {:.5f}".format(ep, step, loss.item()))
                print("Ep: {}, step: {}, loss: {:.5f}".format(ep, step, loss.item()))
        # 进行测试
        mat, auc = run_test(model, test_loader, ep)
        if auc > best_auc:
            best_auc = auc
            # 保存预测结果矩阵
            torch.save(mat, os.path.join(save_dir, "prune_mat.pkl"))
            # 保存模型参数
            torch.save(model.state_dict(), os.path.join(save_dir, "prune_model.pth"))

# 映射对象类别到关系列表中的函数
def convert_obj_class(obj_classes, rel):
    for index, (i_gt_class, i_relationships) in enumerate(zip(obj_classes, rel)):
        for index_rel in range(len(i_relationships)):
            # 将关系中的主语和宾语索引映射到实际的类别
            i_relationships[index_rel, 0] = i_gt_class[i_relationships[index_rel, 0]]
            i_relationships[index_rel, 1] = i_gt_class[i_relationships[index_rel, 1]]
        rel[index] = i_relationships
    return rel


def main():
    path = "/media/n702/data1/Lxy/datasets" # vg data path
    zs_fp = "./zeroshot_triplet_new.pytorch"
    roidb_file = os.path.join(path, "vg/VG-SGG-with-attri.h5")
    dict_file = os.path.join(path, "vg/VG-SGG-dicts-with-attri.json")
    glove_dir = os.path.join(path, "glove")
    embed_dim = 200
    batch_size = 8192
    epoch = 20
    save_dir = "./prune_ckpt/"
    zeroshot_triplet = torch.load(zs_fp).long().numpy()
    print("load info......")
    ind_to_classes, ind_to_predicates, ind_to_attributes = load_info(dict_file)
    print("load train......")
    split_mask, gt_boxes, gt_classes, gt_attributes, relationships = load_graphs(
        roidb_file, "train", num_im=-1, num_val_im=5000,
        filter_empty_rels=True, filter_non_overlap=False, zs_file=zs_fp
    )
    print("load test......")
    _, _, test_class, _, test_relations = load_graphs(
        roidb_file, "test", num_im=-1, num_val_im=5000,
        filter_empty_rels=True, filter_non_overlap=False
    )

    seen_relationships = convert_obj_class(gt_classes, relationships)
    test_relations = convert_obj_class(test_class, test_relations)

    seen_triplets = np.concatenate(np.array(seen_relationships), axis=0)
    seen_set = set()
    unseen_set = set()
    for i in range(len(seen_triplets)):
        item = seen_triplets[i]
        seen_set.add((item[0], item[1], item[2]))

    for i in range(len(zeroshot_triplet)):
        item = zeroshot_triplet[i]
        unseen_set.add((item[0], item[1], item[2]))

    obj_embed_vecs = obj_edge_vectors(ind_to_classes, wv_dir=glove_dir, wv_dim=embed_dim)
    rel_embed_vecs = rel_edge_vectors(ind_to_predicates, wv_dir=glove_dir, wv_dim=embed_dim)
    print("load dataset......")
    dataset = RelationData(seen_set)
    test_dataset = RelationData(unseen_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size//64, shuffle=False)
    model = RelationPrune(obj_embed_vecs, rel_embed_vecs)
    train(model, dataloader, test_loader, epoch, lr=0.001, save_dir=save_dir, pi=0.03)

if __name__ == "__main__":
    main()
