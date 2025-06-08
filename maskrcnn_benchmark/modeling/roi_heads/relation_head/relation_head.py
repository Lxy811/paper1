# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor
from .losses import node_losses, edge_losses

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """
    # 通用关系头类，用于场景图生成中的关系预测

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()  # 克隆配置对象
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)  # 创建联合特征提取器
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)  # 创建物体特征提取器（半输出通道）
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)  # 创建属性特征提取器（半输出通道）
            feat_dim = self.box_feature_extractor.out_channels * 2  # 特征维度为物体和属性特征通道数之和
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)  # 创建物体特征提取器（完整输出通道）
            feat_dim = self.box_feature_extractor.out_channels  # 特征维度为物体特征通道数
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)  # 创建关系预测器
        self.post_processor = make_roi_relation_post_processor(cfg)  # 创建关系后处理器
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)  # 创建关系损失评估器
        self.samp_processor = make_roi_relation_samp_processor(cfg)  # 创建关系采样处理器

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION  # 是否使用联合框特征

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # 前向传播，处理特征图、提议和目标，生成关系预测和损失
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)  # 使用真实框进行关系采样
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)  # 使用检测框进行关系采样
        else:
            rel_labels, rel_binarys = None, None  # 测试模式下无需关系标签和二值关系
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)  # 为测试准备关系对索引

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)  # 提取物体特征

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)  # 提取属性特征
            roi_features = torch.cat((roi_features, att_features), dim=-1)  # 拼接物体和属性特征

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)  # 提取联合框特征
        else:
            union_features = None  # 不使用联合框特征

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        #03--关系特征构建入口
        refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)  # 生成物体和关系预测及附加损失

        # for test
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)  # 测试模式下后处理预测结果
            return roi_features, result, {}  # 返回特征、预测结果和空损失字典

        ####################################################
        #004--初始图节点约束
        obj_labels = [proposal.get_field("labels") for proposal in proposals]  # 获取物体真实标签
        output_losses = node_losses(refine_logits, obj_labels)  # 计算物体分类损失（节点损失）

        e_loss, edges_fg, edges_bg = edge_losses(relation_logits,  # 预测的关系标签（谓词）
                                                 rel_labels,  # 真实关系标签（谓词）
                                                 self.cfg.LOSS,  # 损失类型
                                                 return_idx=True,  # 返回前景和背景索引
                                                 loss_weights=(self.cfg.ALPHA, self.cfg.BETA, self.cfg.GAMMA))  # 损失权重
        add_losses.update(e_loss)  # 更新附加损失
        output_losses.update(add_losses)  # 合并所有损失
    #################################################

        return roi_features, proposals, output_losses  # 返回特征、提议和损失字典


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
