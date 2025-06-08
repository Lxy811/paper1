# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    # 主类，用于实现广义区域卷积神经网络（Generalized R-CNN），当前支持边界框和掩码
    # 包含三个主要部分：
    # - backbone：骨干网络，用于特征提取
    # - rpn：区域建议网络，生成候选区域
    # - heads：处理RPN的特征和建议，计算检测结果或掩码
    """

    def __init__(self, cfg):
        # 初始化父类 nn.Module
        super(GeneralizedRCNN, self).__init__()
        # 克隆配置文件
        self.cfg = cfg.clone()
        # 构建骨干网络，获取图片特征
        self.backbone = build_backbone(cfg)
        # 构建区域建议网络（RPN），使用骨干网络的输出通道数，获取空间特征S
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        # 03--构建ROI头部，处理特征和候选区域，获取视觉特征V用于构建关系特征
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        """
        # 前向传播函数
        # 参数：
        #   images (list[Tensor] 或 ImageList): 需要处理的图像
        #   targets (list[BoxList]): 图像中的真实边界框（可选，用于训练）
        # 返回值：
        #   result (list[BoxList] 或 dict[Tensor]): 模型输出
        #       训练时，返回包含损失的字典
        #       测试时，返回包含额外字段（如分数、标签、掩码）的列表（针对Mask R-CNN模型）
        """
        # 检查训练模式下是否提供了目标
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # 将输入图像转换为 ImageList 格式
        images = to_image_list(images)
        # 通过骨干网络提取特征
        features = self.backbone(images.tensors)
        # 通过RPN生成候选区域和对应的损失-获取空间特征S
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 如果存在ROI头部
        if self.roi_heads:
            # 处理特征、候选区域和目标，生成检测结果和损失
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # 仅RPN模型没有ROI头部
            x = features
            result = proposals
            detector_losses = {}

        # 训练模式下返回损失
        if self.training:
            losses = {}
            # 更新检测器损失
            losses.update(detector_losses)
            # 如果未启用关系模型
            if not self.cfg.MODEL.RELATION_ON:
                # 在关系训练阶段，RPN头部固定，不计算损失
                losses.update(proposal_losses)
            return losses

        # 返回检测结果
        return result
