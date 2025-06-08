import torch
from torch.nn.functional import cross_entropy as CE
from maskrcnn_benchmark.modeling.utils import cat
import logging
# 设置日志配置
# logging.basicConfig(filename='./loss/our_fgbg.txt', level=logging.INFO)

def edge_losses(rel_dists, rel_labels, loss_type='dnorm', idx_fg=None, idx_bg=None,
                return_idx=False, loss_weights=(1,1,1), sfx=''):
    '''
    Predicate classification loss. Based on [1].

    [1] B. Knyazev, H. de Vries, C. Cangea, G.W. Taylor, A. Courville, E. Belilovsky.
    Graph Density-Aware Losses for Novel Compositions in Scene Graph Generation. BMVC 2020.
    https://arxiv.org/abs/2005.08230

    :param rel_dists: 关系分布，模型预测的每个关系类别的概率
    :param rel_labels: 真实关系标签
    :param loss_type: 损失类型，默认为 'dnorm'，支持 'baseline' 和 'dnorm-fgbg'
    :param idx_fg: 前景（有标注）关系的索引
    :param idx_bg: 背景（无标注）关系的索引
    :param return_idx: 是否返回前景和背景索引
    :param loss_weights: 损失权重 (alpha, beta, gamma) 用于调整前景、背景和整体损失
    :param sfx: 损失名称后缀，用于区分不同损失
    :return: 包含关系损失的字典，或额外返回前景和背景索引
    '''
    add_losses = {}  # 初始化附加损失字典

    rel_dists = cat(rel_dists, dim=0)  # 拼接所有关系分布
    rel_labels = cat(rel_labels, dim=0)  # 拼接所有关系标签

    loss = CE(rel_dists, rel_labels.long(), reduction='none')  # 计算每个关系的交叉熵损失（不进行归约）

    if idx_fg is None:
        idx_fg = torch.nonzero(rel_labels > 0).data.view(-1)  # 获取前景关系（标签大于0）的索引

    if idx_bg is None:
        idx_bg = torch.nonzero(rel_labels == 0).data.view(-1)  # 获取背景关系（标签等于0）的索引

    M_FG, M_BG, M = len(idx_fg), len(idx_bg), len(rel_dists)  # 计算前景、背景和总关系数量
    assert M == len(rel_labels), (M, len(rel_labels))  # 验证关系数量与标签数量一致

    alpha, beta, gamma = loss_weights  # 解包损失权重

    if loss_type == 'baseline':
        # 基线损失模式
        assert alpha == beta == 1, ('wrong loss is used, use dnorm or dnorm-fgbg', alpha, beta)  # 确保 alpha 和 beta 为 1
        loss = gamma * (loss / M)  # 所有关系平均加权（除以总数量 M）
        add_losses['rel_loss' + sfx] = loss.sum()  # 计算并存储平均损失（前景和背景关系）

    elif loss_type in ['dnorm', 'dnorm-fgbg']:
        # 密度感知损失模式
        edge_weights = torch.ones(M).to(rel_dists)  # 初始化边权重为 1

        # 前景关系的权重
        if M_FG > 0:
            edge_weights[idx_fg] = float(alpha) / M_FG  # 前景关系的权重为 alpha/M_FG（而非基线的 1/M）

        # 背景关系的权重
        if loss_type == 'dnorm':
            # dnorm 模式下，背景权重基于前景数量
            if M_BG > 0 and M_FG > 0:
                edge_weights[idx_bg] = float(beta) / M_FG  # 背景关系的权重为 beta/M_FG（而非基线的 1/M）
        else:
            # dnorm-fgbg 模式下，背景权重基于背景数量
            if M_BG > 0:
                edge_weights[idx_bg] = float(beta) / M_BG  # 背景关系的权重为 beta/M_BG（而非基线的 1/M）

        loss = gamma * loss * torch.autograd.Variable(edge_weights)  # 应用边权重并乘以 gamma
        # logging.info("loss_fg{:.3}".format(loss[idx_fg].sum()))  # 记录前景损失总和
        # logging.info("loss_bg{:.3}".format(loss[idx_bg].sum()))  # 记录背景损失总和
        add_losses['rel_loss' + sfx] = loss.sum()  # 计算并存储总损失
        # add_losses['loss_fg' + sfx] = loss[idx_fg].sum()  # 存储前景损失（注释掉）
        # add_losses['loss_bg' + sfx] = loss[idx_bg].sum()  # 存储背景损失（注释掉）
    else:
        raise NotImplementedError(loss_type)  # 抛出未实现损失类型的错误

    if return_idx:
        return add_losses, idx_fg, idx_bg  # 返回损失字典及前景、背景索引
    else:
        return add_losses  # 仅返回损失字典


def node_losses(rm_obj_dists, rm_obj_labels, sfx=''):
    # 计算物体分类损失
    rm_obj_dists = cat(rm_obj_dists, dim=0)  # 拼接所有物体分布
    rm_obj_labels = cat(rm_obj_labels, dim=0)  # 拼接所有物体标签
    return { 'obj_loss' + sfx: CE(rm_obj_dists, rm_obj_labels.long()) }  # 返回物体分类的交叉熵损失