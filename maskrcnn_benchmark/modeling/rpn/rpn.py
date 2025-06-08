# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x


@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, mid_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """
    # 区域提议网络（RPN）模块，处理主干网络的特征图，生成提议和损失，支持 FPN 和非 FPN 架构

    def __init__(self, cfg, in_channels):
        super(RPNModule, self).__init__()
        self.cfg = cfg.clone()  # 克隆配置对象

        anchor_generator = make_anchor_generator(cfg)  # 创建锚框生成器

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]  # 从注册表获取 RPN 头部
        head = rpn_head(
            cfg, in_channels, cfg.MODEL.RPN.RPN_MID_CHANNEL, anchor_generator.num_anchors_per_location()[0]
        )  # 初始化 RPN 头部，设置输入通道、中间通道和每位置锚框数量

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))  # 创建边界框编码器，设置权重

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)  # 创建训练模式下的边界框选择器
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)  # 创建测试模式下的边界框选择器

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder)  # 创建 RPN 损失评估器

        self.anchor_generator = anchor_generator  # 保存锚框生成器
        self.head = head  # 保存 RPN 头部
        self.box_selector_train = box_selector_train  # 保存训练边界框选择器
        self.box_selector_test = box_selector_test  # 保存测试边界框选择器
        self.loss_evaluator = loss_evaluator  # 保存损失评估器

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # 前向传播，处理图像和特征图，生成 RPN 提议和损失
        objectness, rpn_box_regression = self.head(features)  # 通过 RPN 头部计算目标性得分和边界框回归
        anchors = self.anchor_generator(images, features)  # 生成锚框

        if self.training:
            return self._forward_train(anchors, objectness, rpn_box_regression, targets)  # 训练模式下调用训练前向传播
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)  # 测试模式下调用测试前向传播

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets):
        # 训练模式下的前向传播
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors  # 对于仅 RPN 模型，直接使用锚框作为提议
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )  # 将锚框转换为训练提议，并进行采样
        loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
            anchors, objectness, rpn_box_regression, targets
        )  # 计算目标性损失和边界框回归损失
        losses = {
            "loss_objectness": loss_objectness,  # 目标性损失
            "loss_rpn_box_reg": loss_rpn_box_reg,  # 边界框回归损失
        }
        return boxes, losses  # 返回训练提议和损失字典

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        # 测试模式下的前向传播
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)  # 将锚框转换为测试提议
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]  # 按目标性得分降序排序
            boxes = [box[ind] for box, ind in zip(boxes, inds)]  # 按排序后的索引重新排列提议
        return boxes, {}  # 返回测试提议和空损失字典


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
