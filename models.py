# -*- coding:utf-8 -*-
# 定义了两个深度学习模型，它们用于处理基于BERT的多任务学习，即同时进行意图识别和槽位填充任务。
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

# 一个自定义的PyTorch模型，继承自Hugging Face Transformers库中的 BertPreTrainedModel。这个模型针对序列级别（如意图识别）和标记级别（如槽位填充）的分类任务，实现了多个分类头。
class BertMultiHeadJointClassification(BertPreTrainedModel):
    # 构造函数初始化这个模型，接收BERT的配置对象以及序列标签和标记标签的数量列表。
    def __init__(self, config, seq_label_nums, token_label_nums):
        """
        num_seq_labels & num_token_labels : [head1_label_num, head2_label_num, ..., headn_label_num]
        """
        super().__init__(config)

        self.seq_label_nums = seq_label_nums
        self.token_label_nums = token_label_nums

        self.seq_head_num = len(seq_label_nums)
        self.token_head_num = len(token_label_nums)
        # 定义了BERT模型 self.bert 以处理输入数据。
        self.bert = BertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        # 分别为序列级别和标记级别的任务定义了线性分类器（self.seq_heads 和 self.token_heads），用于预测每个分类任务的输出标签。
        self.seq_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, seq_label_nums[i]) for i in range(self.seq_head_num)]
        )

        self.token_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, token_label_nums[i]) for i in range(self.token_head_num)]
        )
    
    # 它首先通过BERT模型计算出输入的编码表示。
# 使用 dropout 函数减少过拟合。
# 对于每个序列和标记级别的分类任务，使用一个专门的分类头生成预测向量。
# 如果提供了真实的标记和序列标签，使用交叉熵损失函数来计算损失。
# 如果同时提供了槽位标签和意图标签，那么它计算两个损失并相加。
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        seq_labels=None,
        token_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # bert 模型进行训练 得到的结果
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 取出输出内容中的 last_hidden_state 和 pooler_output
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        # dropout 能够减少参数量
        sequence_output = self.dropout(sequence_output)
        # 槽位识别，得到每个槽位的概率值
        token_logits = [self.token_heads[i](sequence_output) for i in range(self.token_head_num)]

        pooled_output = self.dropout(pooled_output)
        # 意图分类，得到每个意图的概率值
        seq_logits = [self.seq_heads[i](pooled_output) for i in range(self.seq_head_num)]

        loss = None
        if token_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 槽位识别的损失函数 交叉熵损失函数
            token_loss_list = [loss_fct(token_logits[i].view(-1, self.token_label_nums[i]), token_labels[i].view(-1)).unsqueeze(0) for i in range(self.token_head_num)]
            loss = torch.cat(token_loss_list).sum()

        if seq_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            seq_loss_list = [loss_fct(seq_logits[i].view(-1, self.seq_label_nums[i]), seq_labels[i].view(-1)).unsqueeze(0) for i in range(self.seq_head_num)]
            seq_loss = torch.cat(seq_loss_list).sum()
            if loss is None:
                loss = seq_loss
            else:
                loss = loss + seq_loss
        # 返回一个字典，包括损失值、序列标签预测、标记标签预测、隐藏状态和注意力权重
        return {
            'loss': loss,
            'seq_logits': seq_logits,
            'token_logits': token_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
            }

# 特化于本案例中的意图和槽位识别任务
class JointBert(BertMultiHeadJointClassification):
    # 指定了一个意图识别分类头和一个槽位填充分类头
    def __init__(self, config, intent_label_num, slot_label_num):
        super().__init__(config, [intent_label_num], [slot_label_num])
    # 方法定义了模型的前向传播逻辑，其中进行如下步骤：
# 接收输入数据和标签，
# 使用BERT模型对输入数据进行编码，
# 将编码后的表示通过dropout层，
# 通过对应的分类头对意图和槽位进行预测，
# 计算损失并返回结果。
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        intent_labels=None,
        slot_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        seq_labels = [intent_labels] if intent_labels is not None else None
        token_labels = [slot_labels] if slot_labels is not None else None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            seq_labels=seq_labels,
            token_labels=token_labels,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
        )
        # 方法则简化了这个过程，实现了针对单个意图和槽位标签集的任务的损失计算和预测结果输出
        return {
            'loss': outputs['loss'],
            'intent_logits': outputs['seq_logits'][0],
            'slot_logits': outputs['token_logits'][0],
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
            }
