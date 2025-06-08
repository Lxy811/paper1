import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
import copy
from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_posformer import FusionPosTransRelContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from .utils_motifs import to_onehot, obj_edge_vectors, nms_overlaps, encode_box_info
from maskrcnn_benchmark.data import get_dataset_statistics
from math import pi

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_union, cat_boxlist_nofield
from .utils_relation import layer_init

from SHA_GCL_extra.kl_divergence import KL_divergence
from .model_Hybrid_Attention import SHA_Context
from .model_Cross_Attention import CA_Context
from SHA_GCL_extra.utils_funcion import FrequencyBias_GCL
from SHA_GCL_extra.extra_function_utils import generate_num_stage_vector, generate_sample_rate_vector, \
    generate_current_sequence_for_bias, get_current_predicate_idx
from SHA_GCL_extra.group_chosen_function import get_group_splits
import random

#关系推理
@registry.ROI_RELATION_PREDICTOR.register("LxyPredictor1")
class LxyPredictor1(nn.Module):
    def __init__(self, config, in_channels):
        super(LxyPredictor1, self).__init__()
        self.cfg = config
        # 是否开启属性预测
        self.attribute_on = config.MODEL.ATTRIBUTE_ON

        # 加载参数
        # 目标类别数量
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        # 属性类别数量
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        # 关系类别数量
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # 根据配置确定模型的运行模式
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode ='sgcls'
        else:
            self.mode ='sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        # 是否使用视觉信息进行预测
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        # 是否使用偏差进行预测
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        # 剪枝率
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        # 超参数lambda
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # 加载类别字典
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels

        # 根据统计信息加载alpha矩阵
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        # 创建零样本指示矩阵
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
        # 加载剪枝矩阵
        self.prune_mat = torch.load(path).to(self.alpha_mat.device)
        values = self.prune_mat[self.prune_mat != 0]

        sort_values = torch.sort(values)[0]
        length = values.shape[0]
        r_length = int((length - 1) * self.prune_rate)
        if r_length >= len(sort_values):
            self.prune_rate_v = sort_values[-1] + 1
        else:
            self.prune_rate_v = sort_values[r_length]

        self.test_indicator = copy.deepcopy(self.zs_indicator)
        self.test_indicator[(self.alpha_mat == 0) & (self.prune_mat < self.prune_rate_v)] = -1
        self.zs_indicator[(self.alpha_mat == 0) & (self.prune_mat < self.prune_rate_v)] = 0

        self.none_prune_idx = torch.zeros(self.zs_indicator.shape, device=self.zs_indicator.device)
        self.none_prune_idx[(self.alpha_mat == 0) & (self.prune_mat >= self.prune_rate_v)] = 1
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)

        # 混合注意力关系特征模块构建
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # 后解码部分
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        # 后嵌入线性层
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        # 后拼接线性层
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # 关系压缩线性层
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # 上下文压缩线性层
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # 初始化层参数
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            # 上维度线性层
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# 获取模型配置 ############
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        # 零标签填充模式
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        
        ############################################################

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        # 根据预测结果获取不剪枝的索引
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        # 根据预测结果获取零样本测试计算的指示矩阵
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        # 根据预测结果计算权重
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        # 根据预测结果计算已见过的权重
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        # 通过上下文层获取对象分布、对象预测和边上下文特征
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # 后解码操作
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        # 获取关系对数量和对象数量列表
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # 根据对象数量分割头部特征、尾部特征和对象预测
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # 从对象级别特征转换为成对关系级别特征
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # 如果使用视觉信息
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # 生成不剪枝索引和零样本测试计算指标
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        # 如果是训练模式
        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            # 计算关系分布
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            # 获取对象标签列表
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            # 计算权重校准值
            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # 将权重加到关系分布上
            bias_rel_dists = rel_dists + bias_mcal
            # 计算偏置后的关系概率
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            # 计算未见过的关系概率之和
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)

            # 计算未见样本增强损失
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            # 将另一个权重加到关系分布上
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            # 生成非零标签掩码和零标签掩码
            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            # 计算背景关系损失（如果有零标签）
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            # 计算前景关系损失（如果有非零标签）
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            # 计算平均的细粒度偏差缓解损失
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            # 处理关系标签，根据标签选择矩阵
            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):
                rel_tar = rel_labels[i].item()
                if rel_tar == 0:
                    if self.zero_label_padding_mode == 'rand_insert':
                        random_idx = random.randint(0, num_groups - 1)
                        cur_chosen_matrix[random_idx].append(i)
                    elif self.zero_label_padding_mode == 'rand_choose' or self.zero_label_padding_mode == 'all_include':
                        if self.zero_label_padding_mode == 'rand_choose':
                            rand_zeros = random.random()
                        else:
                            rand_zeros = 1.0
                        if rand_zeros >= 0.4:
                            for zix in range(len(cur_chosen_matrix)):
                                cur_chosen_matrix[zix].append(i)
                else:
                    rel_idx = self.incre_idx_list[rel_tar]
                    random_num = random.random()
                    for j in range(num_groups):
                        act_idx = num_groups - j
                        threshold_cur = self.sample_rate_matrix[act_idx - 1][rel_tar]
                        if random_num <= threshold_cur or act_idx < rel_idx:
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

        # 如果是测试模式
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            rel_dists = rel_dists + 2 * zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}

        # 根据对象数量和关系对数量分割对象分布和关系分布
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses


    from collections import defaultdict, Counter
    def re_weight(tpt_list):
        # 使用Counter统计三元组列表中每个三元组的出现次数
        tpt_cnt = Counter(tpt_list)
        # 创建一个全零的张量，用于存储权重矩阵
        alpha_mat = torch.zeros((151, 151, 51)).float()
        # 对统计结果按照出现次数和三元组本身进行排序
        sorted_rel = sorted(tpt_cnt.items(), key=lambda x: (x[1], x[0]))
        # 提取出现次数列表
        freq = np.array([x[1] for x in sorted_rel])
        # 创建三元组到索引的映射字典
        tpt2id = {x[0]: i for i, x in enumerate(sorted_rel)}
        # 获取最大出现次数
        max_freq = max(freq)
        # 计算归一化的对数频率
        log_max_norm = np.log(max_freq / freq)
        # 计算每个三元组的权重值
        alpha_v = log_max_norm * (sum(freq) / sum(log_max_norm * freq))
        # 将权重值赋值到权重矩阵中
        for k, v in tpt_cnt.items():
            alpha_mat[k[0], k[1], k[2]] = alpha_v[tpt2id[k]]
        return alpha_mat
    
    
    # 计算密度归一化的函数
    def edge_losses(rel_dists, rel_labels, loss_type='dnorm', idx_fg=None, idx_bg=None,
                    return_idx=False, loss_weights=(1, 1, 1), sfx=''):
        # 用于存储额外损失的字典
        add_losses = {}

        # 将关系距离张量在第0维上拼接
        rel_dists = cat(rel_dists, dim=0)
        # 将关系标签张量在第0维上拼接
        rel_labels = cat(rel_labels, dim=0)

        # 计算交叉熵损失，不进行归约（得到每个边的损失）
        loss = F.cross_entropy(rel_dists, rel_labels.long(), reduction='none')

        # 如果前景索引为空，则计算前景索引（标签大于0的索引）
        if idx_fg is None:
            idx_fg = torch.nonzero(rel_labels > 0).data.view(-1)

        # 如果背景索引为空，则计算背景索引（标签等于0的索引）
        if idx_bg is None:
            idx_bg = torch.nonzero(rel_labels == 0).data.view(-1)

        # 前景边的数量、背景边的数量、总边的数量
        M_FG, M_BG, M = len(idx_fg), len(idx_bg), len(rel_dists)
        # 断言总边数和总标签数相等
        assert M == len(rel_labels), (M, len(rel_labels))

        # 损失权重参数
        alpha, beta, gamma = loss_weights

        # 如果损失类型为'baseline'
        if loss_type == 'baseline':
            # 断言alpha和beta都为1，否则提示错误
            assert alpha == beta == 1, ('wrong loss is used, use dnorm or dnorm-fgbg', alpha, beta)
            # 对所有边应用相同的权重（除以M以在后面计算平均值）
            loss = gamma * (loss / M)
            # 计算关系损失（对所有前景和背景边的损失求和）
            add_losses['rel_loss' + sfx] = loss.sum()

        # 如果损失类型为'dnorm'或'dnorm-fgbg'
        elif loss_type in ['dnorm', 'dnorm-fgbg']:
            # 创建一个全为1的边权重张量，与关系距离张量在同一设备上
            edge_weights = torch.ones(M).to(rel_dists)

            # 计算前景（已标注）边的权重
            if M_FG > 0:
                edge_weights[idx_fg] = float(alpha) / M_FG

            # 根据损失类型计算背景（未标注）边的权重
            if loss_type == 'dnorm':
                if M_BG > 0 and M_FG > 0:
                    edge_weights[idx_bg] = float(beta) / M_FG
            else:
                if M_BG > 0:
                    edge_weights[idx_bg] = float(beta) / M_BG

            # 将边权重与损失相乘，并乘以gamma
            loss = gamma * loss * torch.autograd.Variable(edge_weights)
            # 计算关系损失（对所有边的加权损失求和）
            add_losses['rel_loss' + sfx] = loss.sum()
            # add_losses['loss_fg' + sfx] = loss[idx_fg].sum()
            # add_losses['loss_bg' + sfx] = loss[idx_bg].sum()
        else:
            # 如果损失类型未实现，则抛出未实现错误
            raise NotImplementedError(loss_type)

        # 如果需要返回索引，则返回额外损失字典、前景索引和背景索引
        if return_idx:
            return add_losses, idx_fg, idx_bg
        # 否则只返回额外损失字典
        else:
            return add_losses


    # 计算节点损失的函数
    def node_losses(rm_obj_dists, rm_obj_labels, sfx=''):
        # 将移除对象的距离张量在第0维上拼接
        rm_obj_dists = cat(rm_obj_dists, dim=0)
        # 将移除对象的标签张量在第0维上拼接
        rm_obj_labels = cat(rm_obj_labels, dim=0)
        # 计算对象损失（使用交叉熵损失）并返回包含损失的字典
        return {'obj_loss' + sfx: F.cross_entropy(rm_obj_dists, rm_obj_labels.long())}