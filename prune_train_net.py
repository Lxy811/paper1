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


class CountFusion(nn.Module):
    def __init__(self, dim_x, dim_y, output_dim=512):
        super(CountFusion, self).__init__()
        self.dense_x = nn.Linear(dim_x, output_dim)  # 初始化输入 x 的线性变换层
        self.dense_y = nn.Linear(dim_y, output_dim)  # 初始化输入 y 的线性变换层
        self.relu = nn.ReLU()  # 初始化 ReLU 激活函数

    def forward(self, x, y):
        x1 = self.dense_x(x)  # 对输入 x 进行线性变换
        y1 = self.dense_y(y)  # 对输入 y 进行线性变换
        item1 = self.relu(x1 + y1)  # 相加后应用 ReLU 激活
        item2 = (x1 - y1) * (x1 - y1)  # 计算差值的平方
        return item1 - item2  # 返回融合结果（加法项减去差值平方项）
#关系组合
class RelationPrune(nn.Module):
    def __init__(self, obj_embed_vecs, rel_embed_vecs, num_objs=151, num_rels=51, embed_dim=200, hidden_dim=512):
        super(RelationPrune, self).__init__()
        self.num_obj_cls = num_objs  # 物体类别数
        self.num_rel_cls = num_rels  # 关系类别数
        self.embed_dim = embed_dim  # 嵌入维度
        self.hidden_dim = hidden_dim  # 隐藏维度
        self.sbj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 初始化主体嵌入层
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 初始化客体嵌入层
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)  # 初始化关系嵌入层
        with torch.no_grad():
            self.sbj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)  # 初始化主体嵌入权重
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)  # 初始化客体嵌入权重
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)  # 初始化关系嵌入权重
        
        self.cnt_fusion_so = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)  # 初始化主体-客体融合模块
        self.cnt_fusion_sr = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)  # 初始化主体-关系融合模块
        self.cnt_fusion_or = CountFusion(embed_dim, embed_dim, output_dim=hidden_dim)  # 初始化客体-关系融合模块
        self.dense_s = nn.Linear(embed_dim, hidden_dim)  # 初始化主体线性变换层
        self.dense_o = nn.Linear(embed_dim, hidden_dim)  # 初始化客体线性变换层
        self.dense_r = nn.Linear(embed_dim, hidden_dim)  # 初始化关系线性变换层
        self.project = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim // 2),  # 线性层，将三部分融合特征降维
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(hidden_dim // 2, 1)  # 线性层，输出单一分数
        )  # 初始化投影网络

    def forward(self, sbj_token, obj_token, rel_token):
        sbj_embed = self.sbj_embed(sbj_token)  # 获取主体嵌入
        obj_embed = self.obj_embed(obj_token)  # 获取客体嵌入
        rel_embed = self.rel_embed(rel_token)  # 获取关系嵌入
        fused_so = self.cnt_fusion_so(sbj_embed, obj_embed)  # 融合主体和客体特征
        fused_sr = self.cnt_fusion_sr(sbj_embed, rel_embed)  # 融合主体和关系特征
        fused_or = self.cnt_fusion_or(obj_embed, rel_embed)  # 融合客体和关系特征

        proj_s = self.dense_s(sbj_embed)  # 线性变换主体嵌入
        proj_o = self.dense_o(obj_embed)  # 线性变换客体嵌入
        proj_r = self.dense_r(rel_embed)  # 线性变换关系嵌入

        fused_so, fused_sr, fused_or = torch.sigmoid(fused_so), torch.sigmoid(fused_sr), torch.sigmoid(fused_or)  # 对融合特征应用 sigmoid
        proj_r, proj_o, proj_s = torch.sigmoid(proj_r), torch.sigmoid(proj_o), torch.sigmoid(proj_s)  # 对投影特征应用 sigmoid

        act_sor = fused_so * proj_r  # 主体-客体与关系投影的激活
        act_sro = fused_sr * proj_o  # 主体-关系与客体投影的激活
        act_ors = fused_or * proj_s  # 客体-关系与主体投影的激活

        concat = torch.cat((act_sor, act_sro, act_ors), dim=-1)  # 拼接三部分激活特征
        logit = torch.sigmoid(self.project(concat))  # 投影并应用 sigmoid 得到最终分数
        return logit  # 返回预测分数

class RelationData(Dataset):
    def __init__(self, seen_triplets, num_objs=151, num_rels=51):
        obj_idx_list = np.arange(1, num_objs)  # 生成物体索引列表（1到num_objs-1）
        rel_idx_list = np.arange(1, num_rels)  # 生成关系索引列表（1到num_rels-1）

        sbj_dim = np.repeat(obj_idx_list, num_objs - 1)  # 重复物体索引生成主体维度
        un_sqz_obj_idx_list = obj_idx_list.reshape(-1, 1)  # 重塑物体索引列表
        un_sqz_rel_idx_list = rel_idx_list.reshape(-1, 1)  # 重塑关系索引列表
        obj_dim = np.repeat(un_sqz_obj_idx_list, num_objs - 1, axis=1).T.reshape(-1)  # 生成客体维度
        sbj_obj = np.stack((sbj_dim, obj_dim), axis=0)  # 堆叠主体和客体索引

        so_dim = np.repeat(sbj_obj, num_rels - 1, axis=1)  # 重复主体-客体对以匹配关系数量
        rel_dim = np.repeat(un_sqz_rel_idx_list, sbj_obj.shape[1], axis=1).T.reshape(1, -1)  # 生成关系维度
        sor_list = np.concatenate((so_dim, rel_dim), axis=0).T  # 拼接生成三元组列表
        self.compose_space = torch.tensor(sor_list, dtype=torch.long)  # 转换为张量表示组合空间
        self.labels = self.gen_label(seen_triplets)  # 生成标签

    def gen_label(self, seen_triplets):
        labels = []
        for i in tqdm.tqdm(range(self.compose_space.shape[0])):
            item = self.compose_space[i, :]  # 获取当前三元组
            tpt = (int(item[0]), int(item[1]), int(item[2]))  # 转换为三元组元组
            if tpt in seen_triplets:
                labels.append(1)  # 已见三元组标记为1
            else:
                labels.append(0)  # 未见三元组标记为0
        labels = torch.tensor(labels, dtype=torch.float)  # 转换为浮点张量
        return labels  # 返回标签张量

    def __getitem__(self, index):
        item = self.compose_space[index, :]  # 获取指定索引的三元组
        y = self.labels[index]  # 获取对应标签
        return item[0], item[1], item[2], y  # 返回主体、客体、关系和标签

    def __len__(self):
        return self.compose_space.shape[0]  # 返回数据集大小

def run_test(model, dataloader, ep):
    model.eval()  # 设置模型为评估模式
    y_preds, y_labels = [], []  # 初始化预测和标签列表
    matrix = torch.zeros((151, 151, 51))  # 初始化三维矩阵存储预测分数
    for token_s, token_o, token_r, y_batch in dataloader:
        token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()  # 将输入移动到 GPU
        with torch.no_grad():
            y_pred = model(token_s, token_o, token_r)  # 前向传播获取预测
        y_preds.append(y_pred)  # 保存预测
        y_labels.append(y_batch)  # 保存真实标签

        token_s, token_o, token_r = token_s.cpu(), token_o.cpu(), token_r.cpu()  # 将输入移回 CPU
        y_pred = y_pred.cpu()  # 将预测移回 CPU
        matrix[token_s, token_o, token_r] = y_pred.reshape(-1)  # 填充预测分数到矩阵

    y_preds = torch.cat(y_preds, dim=0).detach().cpu().numpy()  # 拼接预测并转换为 numpy
    y_labels = torch.cat(y_labels, dim=0).numpy()  # 拼接标签并转换为 numpy
    y_preds_prob = copy.deepcopy(y_preds)  # 复制预测概率
    y_preds_hard = copy.deepcopy(y_preds)  # 复制预测用于硬分类
    y_preds_hard[y_preds > 0.5] = 1  # 阈值大于0.5设为1
    y_preds_hard[y_preds < 0.5] = 0  # 阈值小于0.5设为0
    y_preds_hard = y_preds_hard.astype(int)  # 转换为整数
    recall = metrics.recall_score(y_labels, y_preds_hard)  # 计算召回率
    precision = metrics.precision_score(y_labels, y_preds_hard)  # 计算精确率
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_preds_prob)  # 计算 ROC 曲线
    auc = metrics.auc(fpr, tpr)  # 计算 AUC
    logging.info("ep: {}, recall: {:.3f}, precision: {:.3f}, auc: {:.3f}".format(ep, recall, precision, auc))  # 记录评估指标
    print("ep: {}, recall: {:.3f}, precision: {:.3f}, auc: {:.3f}".format(ep, recall, precision, auc))  # 打印评估指标
    model.train()  # 恢复模型为训练模式
    return matrix, auc  # 返回预测矩阵和 AUC

def train(model, dataloader, test_loader, epoch, lr, save_dir, clip_num=5, pi=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # 初始化 Adam 优化器
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录
    criterion = nn.BCELoss()  # 初始化二元交叉熵损失
    model = model.cuda()  # 将模型移动到 GPU
    best_auc = 0.0  # 初始化最佳 AUC
    for ep in range(epoch):
        for step, (token_s, token_o, token_r, y_batch) in enumerate(dataloader):
            opt.zero_grad()  # 清零梯度
            token_s, token_o, token_r = token_s.cuda(), token_o.cuda(), token_r.cuda()  # 将输入移动到 GPU
            y_batch = y_batch.cuda()  # 将标签移动到 GPU
            y_pred = model(token_s, token_o, token_r)  # 前向传播获取预测
            pos_mask = y_batch == 1  # 正样本掩码
            neg_mask = y_batch == 0  # 负样本掩码
            pf_label = y_batch[pos_mask]  # 获取正样本标签
            pf_label[pf_label == 1] = 0  # 将正样本标签置为0（用于伪负损失）
            if torch.sum(pos_mask) != 0:
                pt_loss = criterion(y_pred[pos_mask].reshape(-1), y_batch[pos_mask])  # 正样本损失
                pf_loss = criterion(y_pred[pos_mask].reshape(-1), pf_label)  # 伪负样本损失
            else:
                pt_loss = pf_loss = 0  # 无正样本时损失为0
            if torch.sum(neg_mask) != 0:
                ng_loss = criterion(y_pred[neg_mask].reshape(-1), y_batch[neg_mask])  # 负样本损失
            else:
                ng_loss = 0  # 无负样本时损失为0
            item1 = pi * pt_loss  # 正样本损失加权
            item2 = ng_loss - pi * pf_loss  # 负样本损失减去伪负损失
            item2[item2 < 0] = 0  # 确保损失非负
            loss = item1 + item2  # 总损失
            loss.backward()  # 反向传播
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_num)  # 梯度裁剪
            opt.step()  # 更新参数
            if step % 30 == 0:
                # 输出到日志
                logging.info("Ep: {}, step: {}, loss: {:.5f}".format(ep, step, loss.item()))  # 记录训练损失
                print("Ep: {}, step: {}, loss: {:.5f}".format(ep, step, loss.item()))  # 打印训练损失
        mat, auc = run_test(model, test_loader, ep)  # 在测试集上评估
        if auc > best_auc:
            best_auc = auc  # 更新最佳 AUC
            #生成语义约束向量---引导零样本的生成
            torch.save(mat, os.path.join(save_dir, "prune_mat.pkl"))  # 保存预测矩阵
            torch.save(model.state_dict(), os.path.join(save_dir, "prune_model.pth"))  # 保存模型权重

# utils 映射到对象类别列表中对应的类别，并将映射后的结果赋值回关系列表中
def convert_obj_class(obj_classes, rel):
    for index, (i_gt_class, i_relationships) in enumerate(zip(obj_classes, rel)):
        for index_rel in range(len(i_relationships)):
            i_relationships[index_rel, 0] = i_gt_class[i_relationships[index_rel, 0]]  # 映射主体类别
            i_relationships[index_rel, 1] = i_gt_class[i_relationships[index_rel, 1]]  # 映射客体类别
        rel[index] = i_relationships  # 更新关系列表
    return rel  # 返回映射后的关系列表


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
    #获取关系标签 
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
