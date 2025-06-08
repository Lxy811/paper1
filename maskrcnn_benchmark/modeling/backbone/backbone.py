# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import vgg


@registry.BACKBONES.register("VGG-16")
def build_vgg_fpn_backbone(cfg):
    body = vgg.VGG16(cfg)
    out_channels = cfg.MODEL.VGG.VGG16_OUT_CHANNELS
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    # 构建 ResNet-101 主干网络，结合 FPN 特征金字塔，用于 RetinaNet
    body = resnet.ResNet(cfg)  # 创建 ResNet 主干网络，根据配置初始化
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS  # 获取 ResNet Stage2 输出通道数
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS  # 获取主干网络输出通道数
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels  # 根据是否使用 C5 层确定 P6、P7 输入通道数
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,  # P2 占位符（不使用）
            in_channels_stage2 * 2,  # P3 输入通道数
            in_channels_stage2 * 4,  # P4 输入通道数
            in_channels_stage2 * 8,  # P5 输入通道数
        ],
        out_channels=out_channels,  # FPN 输出通道数
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),  # 使用 Kaiming 初始化卷积层，支持 GroupNorm 和 ReLU
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),  # 添加 P6、P7 层
    )  # 创建 FPN 特征金字塔
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))  # 组合 ResNet 和 FPN 成顺序模型
    model.out_channels = out_channels  # 设置模型输出通道数
    return model  # 返回构建的主干网络


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
