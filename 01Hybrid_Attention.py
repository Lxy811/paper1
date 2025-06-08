'''Rectified Identity Cell'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# 导入拼接张量的工具函数
from maskrcnn_benchmark.modeling.utils import cat
# 导入自注意力编码器和交叉注意力编码器
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_co_attention import Self_Attention_Encoder, Cross_Attention_Encoder
# 导入目标和边向量、one-hot编码、NMS重叠计算、框信息编码等工具函数
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors,\
    to_onehot, nms_overlaps, encode_box_info

# 自注意力单元类，继承自nn.Module
class Self_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        # 调用父类的构造函数
        super(Self_Attention_Cell, self).__init__()
        # 保存配置信息
        self.cfg = config
        # 如果未指定隐藏维度，则使用配置文件中的默认值
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        # 保存Dropout率
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        # 保存注意力头的数量
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        # 保存内部维度
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        # 保存键的维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        # 保存值的维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # 定义自注意力编码器，将双向隐藏状态从self.hidden_dim*2映射到self.hidden_dim
        self.SA_transformer_encoder = Self_Attention_Encoder(self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, x, textual_feats=None, num_objs=None):
        # 确保输入的目标数量不为空
        assert num_objs is not None
        # 通过自注意力编码器处理输入
        outp = self.SA_transformer_encoder(x, num_objs)

        return outp

# 交叉注意力单元类，继承自nn.Module
class Cross_Attention_Cell(nn.Module):
    def __init__(self, config, hidden_dim=None):
        # 调用父类的构造函数
        super(Cross_Attention_Cell, self).__init__()
        # 保存配置信息
        self.cfg = config
        # 如果未指定隐藏维度，则使用配置文件中的默认值
        if hidden_dim is None:
            self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        else:
            self.hidden_dim = hidden_dim
        # 保存Dropout率
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        # 保存注意力头的数量
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        # 保存内部维度
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        # 保存键的维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        # 保存值的维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # 定义交叉注意力编码器，将双向隐藏状态从self.hidden_dim*2映射到self.hidden_dim
        self.CA_transformer_encoder = Cross_Attention_Encoder(self.num_head, self.k_dim,
                                self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, x, textual_feats, num_objs=None):
        # 确保输入的目标数量不为空
        assert num_objs is not None
        # 通过交叉注意力编码器处理输入
        outp = self.CA_transformer_encoder(x, textual_feats, num_objs)

        return outp

# 混合注意力处理层类，继承自nn.Module
class Single_Layer_Hybrid_Attention(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 定义视觉特征的自注意力单元
        self.SA_Cell_vis = Self_Attention_Cell(config)
        # 定义文本特征的自注意力单元
        self.SA_Cell_txt = Self_Attention_Cell(config)
        # 定义视觉特征的交叉注意力单元
        self.CA_Cell_vis = Cross_Attention_Cell(config)
        # 定义文本特征的交叉注意力单元
        self.CA_Cell_txt = Cross_Attention_Cell(config)

    def forward(self, visual_feats, text_feats, num_objs):
        # 对文本特征进行自注意力处理
        tsa = self.SA_Cell_txt(text_feats, num_objs=num_objs)
        # 对文本特征进行交叉注意力处理
        tca = self.CA_Cell_txt(text_feats, visual_feats, num_objs=num_objs)
        # 对视觉特征进行自注意力处理
        vsa = self.SA_Cell_vis(visual_feats, num_objs=num_objs)
        # 对视觉特征进行交叉注意力处理
        vca = self.CA_Cell_vis(visual_feats, text_feats, num_objs=num_objs)
        # 文本特征的输出为自注意力和交叉注意力结果之和
        textual_output = tsa + tca
        # 视觉特征的输出为自注意力和交叉注意力结果之和
        visual_output = vsa + vca

        return visual_output, textual_output

#混合注意力网络目标编码器类，继承自nn.Module
class SHA_Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    """
    def __init__(self, config, n_layers):
        # 调用父类的构造函数
        super().__init__()
        # 保存配置信息
        self.cfg = config
        # 保存Dropout率
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        # 保存注意力头的数量
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        # 保存内部维度
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        # 保存隐藏维度
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        # 保存键的维度
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        # 保存值的维度
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        # 定义多个单层混合注意力模块
        self.cross_module = nn.ModuleList([
            Single_Layer_Hybrid_Attention(config)
            for _ in range(n_layers)])

    def forward(self, visual_feats, text_feats, num_objs):
        # 初始化视觉输出为输入的视觉特征
        visual_output = visual_feats
        # 初始化文本输出为输入的文本特征
        textual_output = text_feats

        # 依次通过每个单层混合注意力模块
        for enc_layer in self.cross_module:
            visual_output, textual_output = enc_layer(visual_output, textual_output, num_objs)

        # 视觉输出为视觉输出和文本输出之和
        visual_output = visual_output + textual_output

        return visual_output, textual_output

# 混合注意力网络关系编码器类，继承自nn.Module
class SHA_Context(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        # 调用父类的构造函数
        super().__init__()
        # 保存配置信息
        self.cfg = config
        # 根据配置确定模型模式
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        # 保存目标类别
        self.obj_classes = obj_classes
        # 保存关系类别
        self.rel_classes = rel_classes
        # 保存目标类别的数量
        self.num_obj_cls = len(obj_classes)
        # 保存关系类别的数量
        self.num_rel_cls = len(rel_classes)
        # 保存输入通道数
        self.in_channels = in_channels
        # 保存目标特征的维度
        self.obj_dim = in_channels
        # 保存嵌入维度
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        # 保存隐藏维度
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        # 保存NMS阈值
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # 保存Dropout率
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        # 保存目标层的数量
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        # 保存边层的数量
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # 初始化目标嵌入向量
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        # 定义第一个目标嵌入层
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # 定义第二个目标嵌入层
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        # 加载预训练的嵌入向量
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # 定义边界框嵌入层
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        # 定义视觉目标特征的线性层
        self.lin_obj_visual = nn.Linear(self.in_channels + 128, self.hidden_dim)
        # 定义文本目标特征的线性层
        self.lin_obj_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        # 定义视觉边特征的线性层
        self.lin_edge_visual = nn.Linear(self.hidden_dim + self.in_channels, self.hidden_dim)
        # 定义文本边特征的线性层
        self.lin_edge_textual = nn.Linear(self.embed_dim, self.hidden_dim)

        # 定义输出目标类别的线性层
        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)

        # 定义目标上下文编码器
        self.context_obj = SHA_Encoder(config, self.obj_layer)
        # 定义边上下文编码器
        self.context_edge = SHA_Encoder(config, self.edge_layer)

    def forward(self, roi_features, proposals, logger=None):
        # 判断是否使用真实标签
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        # 获取目标标签
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # 进行标签或logits嵌入
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # 进行空间特征L嵌入
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # 使用Transformer编码目标
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_vis = cat((roi_features, pos_embed), -1)
        obj_pre_rep_vis = self.lin_obj_visual(obj_pre_rep_vis)
        obj_pre_rep_txt = obj_embed
        obj_pre_rep_txt = self.lin_obj_textual(obj_pre_rep_txt)
        obj_feats_vis, _ = self.context_obj(obj_pre_rep_vis, obj_pre_rep_txt, num_objs)
        obj_feats = obj_feats_vis

        # 预测目标分布和目标预测结果
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_labels)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            edge_pre_rep_vis = cat((roi_features, obj_feats), dim=-1)
            edge_pre_rep_txt = self.obj_embed2(obj_preds)

        # 边上下文处理
        edge_pre_rep_vis = self.lin_edge_visual(edge_pre_rep_vis)
        edge_pre_rep_txt = self.lin_edge_textual(edge_pre_rep_txt)
        edge_ctx_vis, _ = self.context_edge(edge_pre_rep_vis, edge_pre_rep_txt, num_objs)
        edge_ctx = edge_ctx_vis

        return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        # 按目标数量分割目标分布
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            # 计算重叠矩阵
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            # 对目标分布进行softmax处理
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            # 将背景类的分数设为-1
            out_dists_sampled[:, 0] = -1

            # 初始化输出标签
            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                # 找到分数最大的框和类别
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                # 记录预测结果
                out_label[int(box_ind)] = int(cls_ind)
                # 将重叠框的分数设为0
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                # 将已处理的框的分数设为-1
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        # 拼接预测结果
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

if __name__ == '__main__':
    pass