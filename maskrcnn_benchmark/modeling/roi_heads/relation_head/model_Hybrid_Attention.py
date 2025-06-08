'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info

#自注意力单元
class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Self_Attention_Cell, self).__init__()
        self.cfg = config  # 保存配置对象
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  # 设置默认隐藏维度
        else:
            self.hidden_dim = hidden_dim  # 使用指定隐藏维度
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 获取丢弃率
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 获取注意力头数
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 获取内部维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 获取键维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 获取值维度

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)  # 初始化自注意力编码器

    def forward(self, x, textual_feats=None, num_objs=None):
        assert num_objs is not None  # 确保物体数量不为空
        outp = self.SA_transformer_encoder(x, num_objs)  # 通过自注意力编码器处理输入特征
        return outp  # 返回自注意力输出
#交叉注意力单元
class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        super(Cross_Attention_Cell, self).__init__()
        self.cfg = config  # 保存配置对象
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  # 设置默认隐藏维度
        else:
            self.hidden_dim = hidden_dim  # 使用指定隐藏维度
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 获取丢弃率
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 获取注意力头数
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 获取内部维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 获取键维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 获取值维度

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)  # 初始化交叉注意力编码器

    def forward(self, x, textual_feats, num_objs=None):
        assert num_objs is not None  # 确保物体数量不为空
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)  # 通过交叉注意力编码器处理输入特征
        return outp  # 返回交叉注意力输出
#混合注意力交叉网路
class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    # 单层混合注意力模型，结合自注意力和交叉注意力
    def __init__(self, config):
        super().__init__()
        self.SA_Cell_vis = Self_Attention_Cell(config)  # 初始化视觉特征自注意力单元
        self.SA_Cell_txt = Self_Attention_Cell(config)  # 初始化文本特征自注意力单元
        self.CA_Cell_vis = Cross_Attention_Cell(config)  # 初始化视觉特征交叉注意力单元
        self.CA_Cell_txt = Cross_Attention_Cell(config)  # 初始化文本特征交叉注意力单元

    def forward(self, visual_feats, text_feats, num_objs):
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)  # 文本特征自注意力处理
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)  # 文本特征交叉注意力处理
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)  # 视觉特征自注意力处理
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)  # 视觉特征交叉注意力处理
        textual_output = tsa + tca  # 融合文本自注意力和交叉注意力输出
        visual_output = vsa + vca  # 融合视觉自注意力和交叉注意力输出
        return visual_output, textual_output  # 返回视觉和文本输出

class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    # 混合注意力编码器，包含多层混合注意力
    def __init__(self, config, n_layers):
        super().__init__()
        self.cfg = config  # 保存配置对象
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 获取丢弃率
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD  # 获取注意力头数
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM  # 获取内部维度
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  # 获取隐藏维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM  # 获取键维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM  # 获取值维度
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])  # 初始化多层混合注意力模块

    def forward(self, visual_feats, text_feats, num_objs):
        visual_output = visual_feats  # 初始化视觉输出
        textual_output = text_feats  # 初始化文本输出
        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)  # 逐层处理混合注意力
        visual_output = visual_output + textual_output  # 融合视觉和文本输出
        return visual_output, textual_output  # 返回最终视觉和文本输出

class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config  # 保存配置对象
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'  # 设置预测模式
        else:
            self.mode = 'sgdet'  # 设置检测模式
        self.obj_classes = obj_classes  # 保存物体类别
        self.rel_classes = rel_classes  # 保存关系类别
        self.num_obj_cls = len(obj_classes)  # 物体类别数
        self.num_rel_cls = len(rel_classes)  # 关系类别数
        self.in_channels = in_channels  # 输入通道数
        self.obj_dim = in_channels  # 物体特征维度
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM  # 嵌入维度
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  # 隐藏维度
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES  # 非极大值抑制阈值

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE  # 获取丢弃率
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER  # 物体层数
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER  # 关系层数

        # 语义特征获取L
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # 加载 GloVe 词嵌入
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 初始化物体嵌入层1
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)  # 初始化物体嵌入层2
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)  # 初始化嵌入层1权重
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)  # 初始化嵌入层2权重

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])  # 初始化边界框位置嵌入网络
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)  # 视觉物体特征线性变换
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)  # 文本物体特征线性变换

        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)  # 视觉关系特征线性变换
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)  # 文本关系特征线性变换

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)  # 物体分类输出层

        self.context_obj = SHA_Encoder(config, self.obj_layer)  # 初始化目标编码器
        self.context_edge = SHA_Encoder(config, self.edge_layer)  # 初始化关系编码器

    def forward(self, roi_features, proposals, logger=None):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL  # 是否使用真实物体标签
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None  # 获取物体标签

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)  # 使用真实标签生成嵌入
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()  # 获取预测 logits
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight  # 使用 softmax 加权嵌入

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'  # 确保提议框格式为 xyxy
        pos_embed = self.bbox_embed(encode_box_info(proposals))  # 生成边界框位置嵌入

        # encode objects with transformer
        num_objs = [len(p) for p in proposals]  # 获取每张图像的物体数量
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)  # 拼接视觉特征和位置嵌入
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)  # 线性变换视觉物体特征
        obj_pre_rep_txt = obj_embed  # 获取文本嵌入
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)  # 线性变换文本物体特征
        obj_feats_vis, _ = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)  # 通过物体上下文编码器处理
        obj_feats = obj_feats_vis  # 使用细化目标特征f·

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels  # 使用真实标签作为预测
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)  # 转换为 one-hot 编码
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)  # 拼接视觉特征和物体特征
            edge_pre_rep_txt = self.obj_embed2(obj_labels)  # 目标解码器--获取目标标签预测l
        else:
            obj_dists = self.out_obj(obj_feats)  # 预测物体分布
            use_decoder_nms = self.mode == 'sgdet' and not self.training  # 是否使用解码器 NMS
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]  # 获取每类边界框
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)  # 按类别进行 NMS
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1  # 获取最大概率的物体预测
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)  # 拼接视觉特征和物体特征
            edge_pre_rep_txt = self.obj_embed2(obj_preds)  # 目标解码器--获取目标标签预测l

        # edge context
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)  # 线性变换视觉关系特征
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)  # 线性变换文本关系特征
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs)  # 通过关系编码器处理
        edge_ctx = edge_ctx_vis  # 最终目标特征（关系上下文）

        return obj_dists, obj_preds, edge_ctx  # 返回物体分布、物体预测和关系上下文

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)  # 按物体数量分割分布
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # 计算重叠矩阵（NMS 阈值）
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()  # 计算 softmax 概率
            out_dists_sampled[:, 0] = -1  # 将背景类概率置为负值

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)  # 初始化输出标签

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)  # 找到最大概率的框和类别
                out_label[int(box_ind)] = int(cls_ind)  # 设置标签
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0  # 清除重叠框的概率
                out_dists_sampled[box_ind] = -1.0  # 避免重复采样

            obj_preds.append(out_label.long())  # 添加预测标签
        obj_preds = torch.cat(obj_preds, dim=0)  # 拼接所有预测标签
        return obj_preds  # 返回物体预测



if __name__ == '__main__':
    pass


