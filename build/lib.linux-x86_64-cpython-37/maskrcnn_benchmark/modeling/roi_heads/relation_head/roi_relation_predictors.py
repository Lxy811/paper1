# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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


@registry.ROI_RELATION_PREDICTOR.register("LxyPredictor1")
class LxyPredictor1(nn.Module):
    def __init__(self, config, in_channels):
        super(LxyPredictor1, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            ###########################################
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            rel_dists = rel_dists + 2*zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
        
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

####(1,0,1) H+SP  怎么是全有的这个结果不是101,我右重新修改了,这是111
@registry.ROI_RELATION_PREDICTOR.register("XPredictor101")
class XPredictor101(nn.Module):
    def __init__(self, config, in_channels):
        super(XPredictor101, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists + 2*zs_test_cal
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
####(1,1,0) H+CL
@registry.ROI_RELATION_PREDICTOR.register("Predictor110")
class Predictor110(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor110, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            ###########################################
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
   
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            # rel_dists = rel_dists + 2*zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
        
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("LxyPredictor110")
class LxyPredictor110(nn.Module):
    def __init__(self, config, in_channels):
        super(LxyPredictor110, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("NewLxyPredictor110")
class NewLxyPredictor110(nn.Module):
    def __init__(self, config, in_channels):
        super(NewLxyPredictor110, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists + zs_test_cal
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses


####(1,1,0)  包括GCL
@registry.ROI_RELATION_PREDICTOR.register("ReLxyPredictor110")
class ReLxyPredictor110(nn.Module):
    def __init__(self, config, in_channels):
        super(ReLxyPredictor110, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
 
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all



####(0,0,0)
@registry.ROI_RELATION_PREDICTOR.register("Predictor000")
class Predictor000(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor000, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, rel_pair_idxs, union_features,
                                                            logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        obj_preds = obj_preds.split(num_objs, dim=0)

        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_pred = cat(pair_preds, dim=0)
        ctx_gate = self.post_cat(edge_ctx)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(edge_ctx)
        
       
        add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

####(1,0,0) 效果不好的话，基线模型改改参数
@registry.ROI_RELATION_PREDICTOR.register("Predictor100")
class Predictor100(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor100, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
       
        add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
####(1,0,1)
@registry.ROI_RELATION_PREDICTOR.register("RePredictor101")
class RePredictor101(nn.Module):
    def __init__(self, config, in_channels):
        super(RePredictor101, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] =0
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] =0
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            ###########################################
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
   
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            rel_dists = rel_dists + 2*zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
        
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all


@registry.ROI_RELATION_PREDICTOR.register("NPredictor101")
class NPredictor101(nn.Module):
    def __init__(self, config, in_channels):
        super(NPredictor101, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        
        rel_dists = rel_dists + 2*zs_test_cal
        add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("LxyPredictorN")
class LxyPredictorN(nn.Module):
    def __init__(self, config, in_channels):
        super(LxyPredictorN, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        ####xxxxxtorch.Size([197, 4096]   torch.Size([2684, 4096]) 
        """
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([2818, 4096])
        torch.Size([194, 512])
        这里的形状一直变动的，因为图片检测到的对象关系是变化的
        """
        # print(ctx_gate.shape)
        # print(union_features.shape)
        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists + zs_test_cal
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
####(1,1,0)
@registry.ROI_RELATION_PREDICTOR.register("Predictor011")
class Predictor011(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor011, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
        
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        # print(rel_dists.shape)
        
        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5
        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape)
            bias_rel_dists = rel_dists + bias_mcal
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists + 2*zs_test_cal
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

####(0,1,0)
@registry.ROI_RELATION_PREDICTOR.register("Predictor010")
class Predictor010(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor010, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
        
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        # print(rel_dists.shape)
        
        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5
        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape)
            bias_rel_dists = rel_dists + bias_mcal
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("RePredictor010")
class RePredictor010(nn.Module):
    def __init__(self, config, in_channels):
        super(RePredictor010, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, rel_pair_idxs, union_features,
                                                            logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        obj_preds = obj_preds.split(num_objs, dim=0)

        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_pred = cat(pair_preds, dim=0)
        ctx_gate = self.post_cat(edge_ctx)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(edge_ctx)
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5
        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            bias_rel_dists = rel_dists + bias_mcal
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("Predictor001")
class Predictor001(nn.Module):
    def __init__(self, config, in_channels):
        super(Predictor001, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, rel_pair_idxs, union_features,
                                                            logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        obj_preds = obj_preds.split(num_objs, dim=0)

        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_pred = cat(pair_preds, dim=0)
        ctx_gate = self.post_cat(edge_ctx)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(edge_ctx)
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5
        if not self.training:
            rel_dists = rel_dists+ zs_test_cal
        add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

###############################################################################################
##############################################################################################
@registry.ROI_RELATION_PREDICTOR.register("PredictorLoss100")
class PredictorLoss000(nn.Module):
    def __init__(self, config, in_channels):
        super(PredictorLoss000, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = 0.0
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = 0.0
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            ###########################################
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
   
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            rel_dists = rel_dists + 2*zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
        
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all

@registry.ROI_RELATION_PREDICTOR.register("PredictorLoss001")
class PredictorLoss001(nn.Module):
    def __init__(self, config, in_channels):
        super(PredictorLoss001, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels


        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        ###self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)


        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        ################# get model configs############
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()
        ############################################################
    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)   #形状同edge_ctx.shape【189，512】O【2672，4096】R

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]     # 假设num_objs = [4, 5, 3] 
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)  #那么第一个张量的形状为 (4, 512)，第二个张量的形状为 (5, 512)。。。。。
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))  #head_rep[idx[0]] 和 tail_rep[idx[1]] 分别表示从 head_rep 和 tail_rep 中选择对应索引位置的特征向量。然后，使用 torch.cat() 函数将这两个特征向量沿着特征维度（dim=-1）进行拼接，得到 concatenated_rep
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
#######################################################################

        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        # print(visual_rep.shape) #torch.Size([3126, 4096])
        # print(edge_ctx.shape) #torch.Size([242, 512])
        # print(pair_preds.shape)

        
        # print(prod_rep.shape)  #torch.Size([3410, 1024]) 3410是检测到的对象？
        # print(rel_dists.shape) #torch.Size([3410, 51])  150种object和50种predicate
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5

        if self.training:
            add_losses = {}
            pair_gt_preds = []
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            # print(bias_mcal.shape) #torch.Size([3410, 51])  torch.Size([3126, 51])  
            # print(bias_mce.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
            bias_rel_dists = rel_dists + bias_mcal
            # print(bias_rel_dists.shape) #torch.Size([3410, 51]) torch.Size([3126, 51])
           
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = 0.0
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] =rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)

            ###########################################
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss
   
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            rel_dists = rel_dists + 2*zs_test_cal
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses
        
    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all


@registry.ROI_RELATION_PREDICTOR.register("TransLike_GCL")
class TransLike_GCL(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_GCL, self).__init__()
         # load parameters
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.union_single_not_match:
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features

        if self.training:
            if not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
                add_losses['obj_loss'] = loss_refine_obj

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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

            return None, None, add_losses
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
            rel_dists = rel_dists.split(num_rels, dim=0)
            obj_dists = obj_dists.split(num_objs, dim=0)

            return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all



@registry.ROI_RELATION_PREDICTOR.register("TransLike_GCL01")
class TransLike_GCL01(nn.Module):
    def __init__(self, config, in_channels):
        super(TransLike_GCL01, self).__init__()
        # load parameters
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.in_channels = in_channels
        # module construct
        if config.GLOBAL_SETTING.BASIC_ENCODER == 'Self-Attention':
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Cross-Attention':
            self.context_layer = CA_Context(config, obj_classes, rel_classes, in_channels)
        elif config.GLOBAL_SETTING.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim*2, self.num_rel_cls)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # get model configs
        self.Knowledge_Transfer_Mode = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_TRANSFER_MODE
        self.no_relation_restrain = config.GLOBAL_SETTING.GCL_SETTING.NO_RELATION_RESTRAIN
        self.zero_label_padding_mode = config.GLOBAL_SETTING.GCL_SETTING.ZERO_LABEL_PADDING_MODE
        self.knowledge_loss_coefficient = config.GLOBAL_SETTING.GCL_SETTING.KNOWLEDGE_LOSS_COEFFICIENT
        # generate the auxiliary lists
        self.group_split_mode = config.GLOBAL_SETTING.GCL_SETTING.GROUP_SPLIT_MODE
        num_of_group_element_list, predicate_stage_count = get_group_splits(config.GLOBAL_SETTING.DATASET_CHOICE, self.group_split_mode)
        self.max_group_element_number_list = generate_num_stage_vector(num_of_group_element_list)  #看上面的函数
        self.incre_idx_list, self.max_elemnt_list, self.group_matrix, self.kd_matrix = get_current_predicate_idx(
            num_of_group_element_list, 0.1, config.GLOBAL_SETTING.DATASET_CHOICE)
        self.sample_rate_matrix = generate_sample_rate_vector(config.GLOBAL_SETTING.DATASET_CHOICE, self.max_group_element_number_list)
        self.bias_for_group_split = generate_current_sequence_for_bias(num_of_group_element_list, config.GLOBAL_SETTING.DATASET_CHOICE)

        self.num_groups = len(self.max_elemnt_list)
        self.rel_compress_all, self.ctx_compress_all = self.generate_muti_networks(self.num_groups)
        self.CE_loss = nn.CrossEntropyLoss()

        if self.use_bias:
            self.freq_bias_all = self.generate_multi_bias(config, statistics, self.num_groups)

        if self.Knowledge_Transfer_Mode != 'None':
            self.NLL_Loss = nn.NLLLoss()
            self.pre_group_matrix = torch.tensor(self.group_matrix, dtype=torch.int64).cuda()
            self.pre_kd_matrix = torch.tensor(self.kd_matrix, dtype=torch.float16).cuda()
            self.criterion_loss = nn.CrossEntropyLoss()

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        add_losses = {}
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # post decode
        edge_rep = self.post_emb(edge_ctx)

        #如果 edge_rep 的形状是 (batch_size, hidden_dim * 2)，那么调整形状后的 edge_rep 将变成 (batch_size, 2, hidden_dim)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.union_single_not_match:
            visual_rep = ctx_gate * self.up_dim(union_features)
        else:
            visual_rep = ctx_gate * union_features
        # print(visual_rep.shape) #torch.Size([1600, 4096])
        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
        # print(prod_rep.shape) #torch.Size([1600, 1024])
        # print(rel_dists.shape) #torch.Size([1600, 51])
        if self.training:
            # if not self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            #     fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            #     loss_refine_obj = self.criterion_loss(obj_dists, fg_labels.long())
            #     add_losses['obj_loss'] = loss_refine_obj

            rel_labels = cat(rel_labels, dim=0)
            max_label = max(rel_labels)

            num_groups = self.incre_idx_list[max_label.item()]
            if num_groups == 0:
                num_groups = max(self.incre_idx_list)
            cur_chosen_matrix = []

            for i in range(num_groups):
                cur_chosen_matrix.append([])

            for i in range(len(rel_labels)):   #这段代码的作用是根据输入的关系标签和一些参数设置，填充一个矩阵
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
                            # print('%d-%d-%d-%.2f-%.2f'%(i, rel_idx, act_idx, random_num, threshold_cur))
                            for k in range(act_idx):
                                cur_chosen_matrix[k].append(i)
                            break

            for i in range(num_groups):
                if max_label == 0:
                    group_visual = visual_rep
                    group_input = prod_rep
                    group_label = rel_labels
                    group_pairs = pair_pred
                else:
                    group_visual = visual_rep[cur_chosen_matrix[i]]
                    group_input = prod_rep[cur_chosen_matrix[i]]
                    group_label = rel_labels[cur_chosen_matrix[i]]
                    group_pairs = pair_pred[cur_chosen_matrix[i]]

                '''count Cross Entropy Loss'''
                jdx = i
                rel_compress_now = self.rel_compress_all[jdx]
                ctx_compress_now = self.ctx_compress_all[jdx]
                group_output_now = rel_compress_now(group_visual) + ctx_compress_now(group_input)
                if self.use_bias:
                    rel_bias_now = self.freq_bias_all[jdx]
                    group_output_now = group_output_now + rel_bias_now.index_with_labels(group_pairs.long())
                    # print("group_output_now.shape")
                    # print(group_output_now.shape)  # torch.Size([528, 5]) + torch.Size([538, 11])
                # actual_label_piece: if label is out of range, then filter it to ensure the training can continue
                actual_label_now = self.pre_group_matrix[jdx][group_label]
                add_losses['%d_CE_loss' % (jdx + 1)] = self.CE_loss(group_output_now, actual_label_now)

                if self.Knowledge_Transfer_Mode == 'KL_logit_Neighbor':
                    if i > 0:
                        '''count knowledge transfer loss'''
                        jbef = i - 1
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)  #应该进行广播机制了吧
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        add_losses['%d%d_kl_loss' % (jbef + 1, jdx + 1)] = kd_loss_final

                elif self.Knowledge_Transfer_Mode == 'KL_logit_TopDown':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        # print("group_output_bef.shape")
                        # print(group_output_bef.shape)  #torch.Size([538, 5])

                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=False)
                            kd_loss_vecify = kd_loss_matrix * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix = KL_divergence(group_output_bef[:, 1:], group_output_now[:, 1:max_vector],
                                                           reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * kd_loss_matrix
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

                elif self.Knowledge_Transfer_Mode == 'KL_logit_BiDirection':
                    layer_total_loss = 0
                    for jbef in range(i):
                        rel_compress_bef = self.rel_compress_all[jbef]
                        ctx_compress_bef = self.ctx_compress_all[jbef]
                        group_output_bef = rel_compress_bef(group_visual) + ctx_compress_bef(group_input)
                        if self.use_bias:
                            rel_bias_bef = self.freq_bias_all[jbef]
                            group_output_bef = group_output_bef + rel_bias_bef.index_with_labels(group_pairs.long())
                        max_vector = self.max_elemnt_list[jbef] + 1

                        if self.no_relation_restrain:
                            kd_choice_vector = self.pre_kd_matrix[jbef][group_label]
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=False)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=False)
                            kd_loss_vecify = (kd_loss_matrix_td + kd_loss_matrix_bu) * kd_choice_vector
                            kd_loss_final = self.knowledge_loss_coefficient * torch.mean(kd_loss_vecify)
                        else:
                            kd_loss_matrix_td = KL_divergence(group_output_bef[:, 1:],
                                                              group_output_now[:, 1:max_vector],
                                                              reduce=True)
                            kd_loss_matrix_bu = KL_divergence(group_output_now[:, 1:max_vector],
                                                              group_output_bef[:, 1:],
                                                              reduce=True)
                            kd_loss_final = self.knowledge_loss_coefficient * (kd_loss_matrix_td + kd_loss_matrix_bu)
                        layer_total_loss += kd_loss_final

                    if i > 0:
                        add_losses['%d_DKS_loss' % (jdx + 1)] = layer_total_loss

            
        else:
            rel_compress_test = self.rel_compress_all[-1]
            ctx_compress_test = self.ctx_compress_all[-1]
            rel_dists = rel_compress_test(visual_rep) + ctx_compress_test(prod_rep)
            if self.use_bias:
                rel_bias_test = self.freq_bias_all[-1]
                rel_dists = rel_dists + rel_bias_test.index_with_labels(pair_pred.long())
        rel_dists = rel_dists.split(num_rels, dim=0)
        obj_dists = obj_dists.split(num_objs, dim=0)

        return obj_dists, rel_dists, add_losses

    def generate_muti_networks(self, num_cls):
        '''generate all the hier-net in the model, need to set mannually if use new hier-class'''
        self.rel_classifer_1 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[0] + 1)
        self.rel_classifer_2 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[1] + 1)
        self.rel_classifer_3 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[2] + 1)
        self.rel_classifer_4 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[3] + 1)
        self.rel_compress_1 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[0] + 1)
        self.rel_compress_2 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[1] + 1)
        self.rel_compress_3 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[2] + 1)
        self.rel_compress_4 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[3] + 1)
        layer_init(self.rel_classifer_1, xavier=True)
        layer_init(self.rel_classifer_2, xavier=True)
        layer_init(self.rel_classifer_3, xavier=True)
        layer_init(self.rel_classifer_4, xavier=True)
        layer_init(self.rel_compress_1, xavier=True)
        layer_init(self.rel_compress_2, xavier=True)
        layer_init(self.rel_compress_3, xavier=True)
        layer_init(self.rel_compress_4, xavier=True)
        if num_cls == 4:
            classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3, self.rel_classifer_4]
            compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3, self.rel_compress_4]
        elif num_cls < 4:
            exit('wrong num in compress_all')
        else:
            self.rel_classifer_5 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_classifer_5, xavier=True)
            self.rel_compress_5 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[4] + 1)
            layer_init(self.rel_compress_5, xavier=True)
            if num_cls == 5:
                classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                 self.rel_classifer_4, self.rel_classifer_5]
                compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                self.rel_compress_4, self.rel_compress_5]
            else:
                self.rel_classifer_6 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_classifer_6, xavier=True)
                self.rel_compress_6 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[5] + 1)
                layer_init(self.rel_compress_6, xavier=True)
                if num_cls == 6:
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6]
                else:
                    self.rel_classifer_7 = nn.Linear(self.pooling_dim, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_classifer_7, xavier=True)
                    self.rel_compress_7 = nn.Linear(self.hidden_dim * 2, self.max_group_element_number_list[6] + 1)
                    layer_init(self.rel_compress_7, xavier=True)
                    classifer_all = [self.rel_classifer_1, self.rel_classifer_2, self.rel_classifer_3,
                                     self.rel_classifer_4, self.rel_classifer_5, self.rel_classifer_6,
                                     self.rel_classifer_7]
                    compress_all = [self.rel_compress_1, self.rel_compress_2, self.rel_compress_3,
                                    self.rel_compress_4, self.rel_compress_5, self.rel_compress_6, self.rel_compress_7]
                    if num_cls > 7:
                        exit('wrong num in compress_all')
        return classifer_all, compress_all

    def generate_multi_bias(self, config, statistics, num_cls):
        self.freq_bias_1 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[0])
        self.freq_bias_2 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[1])
        self.freq_bias_3 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[2])
        self.freq_bias_4 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[3])
        if num_cls < 4:
            exit('wrong num in multi_bias')
        elif num_cls == 4:
            freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4]
        else:
            self.freq_bias_5 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE, predicate_all_list=self.bias_for_group_split[4])
            if num_cls == 5:
                freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3, self.freq_bias_4,
                                 self.freq_bias_5]
            else:
                self.freq_bias_6 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                      predicate_all_list=self.bias_for_group_split[5])
                if num_cls == 6:
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6]
                else:
                    self.freq_bias_7 = FrequencyBias_GCL(config, statistics, config.GLOBAL_SETTING.DATASET_CHOICE,
                                                          predicate_all_list=self.bias_for_group_split[6])
                    freq_bias_all = [self.freq_bias_1, self.freq_bias_2, self.freq_bias_3,
                                     self.freq_bias_4, self.freq_bias_5, self.freq_bias_6, self.freq_bias_7]
                    if num_cls > 7:
                        exit('wrong num in multi_bias')
        return freq_bias_all


@registry.ROI_RELATION_PREDICTOR.register("TCARPredictor")
class TCARPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TCARPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.prune_rate = config.MODEL.ROI_RELATION_HEAD.PRUNE_RATE
        self.lambda_ = config.MODEL.ROI_RELATION_HEAD.LAMBDA_

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = (
            statistics["obj_classes"],
            statistics["rel_classes"],
            statistics["att_classes"],
        )
        if 'alpha_mat' in statistics:
            alpha_mat = statistics['alpha_mat']
        elif 'tpt_weight' in statistics:
            alpha_mat = statistics['tpt_weight']
        self.alpha_mat = alpha_mat.cuda()
        self.zs_indicator = torch.zeros(self.alpha_mat.shape, device=self.alpha_mat.device, dtype=torch.long)
        self.zs_indicator[self.alpha_mat == 0] = 1
        self.zs_indicator[self.alpha_mat != 0] = -1

        path = './prune_ckpt/prune_mat.pkl'
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
        # module construct
        self.context_layer = FusionPosTransRelContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = nn.Linear(self.hidden_dim, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim, self.num_rel_cls)

        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def gen_none_prune_idx(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        none_prune_idx = self.none_prune_idx[pred[:, 0], pred[:, 1], :]
        zs_test_cal = self.test_indicator[pred[:, 0], pred[:, 1], :]
        return none_prune_idx, zs_test_cal

    def self_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        alpha_v[alpha_v == 0] = 1
        return alpha_v

    def self_seen_weight_calibrate(self, pair_preds):
        pred = torch.cat(pair_preds, dim=0)
        alpha_v = -self.alpha_mat[pred[:, 0], pred[:, 1], :]
        return alpha_v

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, rel_pair_idxs, union_features,
                                                            logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        obj_preds = obj_preds.split(num_objs, dim=0)

        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_pred = cat(pair_preds, dim=0)
        ctx_gate = self.post_cat(edge_ctx)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(edge_ctx)
        none_prune_idx, zs_test_cal = self.gen_none_prune_idx(pair_preds)
        eps = 1e-5
        if self.training:
            add_losses = {}
            pair_gt_preds = []
            obj_labels = [proposal.get_field("labels") for proposal in proposals]
            for pair_idx, obj_lbl in zip(rel_pair_idxs, obj_labels):
                pair_gt_preds.append(torch.stack((obj_lbl[pair_idx[:, 0]], obj_lbl[pair_idx[:, 1]]), dim=1))

            bias_mcal = self.self_weight_calibrate(pair_preds)
            bias_mce = self.self_seen_weight_calibrate(pair_preds)
            bias_rel_dists = rel_dists + bias_mcal
            bias_rel_prob = torch.softmax(bias_rel_dists, dim=-1)
            sum_unseen = torch.sum(bias_rel_prob * (none_prune_idx == 1), dim=-1)
            
            if torch.mean(sum_unseen) < eps:
                loss_cal = -torch.log(torch.mean(sum_unseen) + eps)
            else:
                loss_cal = -torch.log(torch.mean(sum_unseen))

            add_losses["loss_cal"] = self.lambda_ * loss_cal
            rel_dists = rel_dists + bias_mce

            ce_criterion = nn.CrossEntropyLoss()
            rel_lbl = cat(rel_labels, dim=0)

            non_zero_mask = rel_lbl != 0
            zero_mask = rel_lbl == 0
            rel_fbg_loss = []
            if rel_lbl[zero_mask].shape[0] != 0:
                bg_rel_loss = ce_criterion(rel_dists[zero_mask], rel_lbl[zero_mask].long())
                rel_fbg_loss.append(3 * bg_rel_loss)
            if rel_lbl[non_zero_mask].shape[0] != 0:
                fg_rel_loss = ce_criterion(rel_dists[non_zero_mask], rel_lbl[non_zero_mask].long())
                rel_fbg_loss.append(fg_rel_loss)
            rel_fbg_loss = sum(rel_fbg_loss) / 4
            add_losses["rel_ce_loss"] = rel_fbg_loss
            rel_dists = to_onehot(rel_lbl, self.num_rel_cls)
        else:
            rel_dists = rel_dists + zs_test_cal
            add_losses = {}
        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, add_losses

@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
