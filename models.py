# -*- coding:utf-8 -*-
# 定义了两个深度学习模型，它们用于处理基于BERT的多任务学习，即同时进行意图识别和槽位填充任务。
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

# 一个自定义的 PyTorch 模型，是一个基于BERT的多任务学习模型，能够同时处理多种序列和标记分类任务，通过不同的"头"来识别和预测句子的整体意图以及句子中每个词的具体作用。。
class BertMultiHeadJointClassification(BertPreTrainedModel):
    def __init__(self, config, seq_label_nums, token_label_nums):
        """
        在构造函数内部，我们为机器人安排了学习和测试的配置：
          - config: BERT 语言课程的说明书，告诉机器人大脑该如何配置。
          - seq_label_nums: 意图测试题目的总数（多少个不同的意图）。
          - token_label_nums: 标记测试题目的总数（每个词可能的标记种类）。
        """
        super().__init__(config)
        # 储存意图和标记的数量，也就是测试题目的种类数。
        self.seq_label_nums = seq_label_nums
        self.token_label_nums = token_label_nums
        # 意图任务（即句子级别任务）和标记任务（即词语级别任务）的数量。
        self.seq_head_num = len(seq_label_nums)
        self.token_head_num = len(token_label_nums)

        # 实例化BERT模型，它用于理解输入的语言信息
        self.bert = BertModel(config, add_pooling_layer=True)
        # 配置dropout比率，用于减少过拟合，即避免机器人只记住特定数据的“样子”而不理解其含义。
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 为序列级任务创建多个分类器，用来最终预测意图。
        # 意图识别分类头负责判断一个句子整体要表达的目的是什么。就像你跟助手说：“我饿了”，它的意图识别眼睛就要能明白你的意图是“想吃东西”。
        # 工作原理是，它会看到你的句子，并从中提取关键的信息，然后对比它所学习过的各种可能的意图，判断出最接近的那一个。就好比是一个经验丰富的侦探，通过分析线索来确定你的真正目的。
        # 这是用来处理意图识别任务的部分。
        # 序列头的每一个线性层都被用来预测句子的整体意图或分类。
        # 通常情况下，你可能有多个意图分类，每个seq_head对应一个特定的意图分类任务。一个句子只对应一个整体的意图，所以“序列头”通常用句子级别的特征（例如BERT模型产出的pooled_output）来进行预测。
        self.seq_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, seq_label_nums[i]) for i in range(self.seq_head_num)]
        )
        # 为标记级任务创建多个分类器，用来预测每个词的标记。
        # 槽位填充分类头则是用来识别和理解句子中的具体信息片段，它关注句子的细节。如果你说：“帮我订明天上午的飞往纽约的票”，这只眼睛就要弄清楚“明天上午”是时间信息，“飞往纽约”是目的地信息。
        # 槽位填充分类头的原理是，将句子中的每一个词或词语分类，归入不同的“槽位”，它们代表了不同的信息类别，就像填写表格时把信息放在正确的栏目中一样。
        # 这一部分用于标记级任务，比如槽位填充（slot-filling）。
        # 标记头中的每个线性层用于为输入句子中的每个单词或标记预测一个分类标签，这些标签代表了单词在句中的具体角色或属性。
        # 每个单词都会被分配一个标签，所以模型要为每个单词单独进行预测，使用的是词级别的特征（例如BERT模型产出的sequence_output）
        self.token_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, token_label_nums[i]) for i in range(self.token_head_num)]
        )
        # 两只眼睛协同工作，一只负责理解“你想做什么”，另一只负责理解“具体的关键信息”，共同帮助智能助手更精准地完成你的请求。

    # forward 方法，机器人通过这个方法接收信息并进行处理, 是模型学习和自我测试的地方
    def forward(
        self,
        input_ids=None,           # 输入的句子编号，是我们要分析的数据。
        attention_mask=None,      # 用于标记哪些部分的输入是有意义的，哪些是填充。
        token_type_ids=None,      # 输入的句子分割信息（例如，在问答任务中区分问题和答案）。
        position_ids=None,        # 输入的词汇位置信息。
        head_mask=None,           # 可以用来遮罩BERT中某些头部的参数，对模型的注意力机制有影响。
        inputs_embeds=None,       # 可以直接提供嵌入层，而不是通过模型内部的嵌入层转化input_ids。
        seq_labels=None,          # 真实的意图标签，用于训练时计算损失。
        token_labels=None,        # 真实的词标记标签，用于训练时计算损失。
        output_attentions=None,   # 是否输出注意力权重。
        output_hidden_states=None,# 是否输出BERT隐层状态。
        return_dict=None,         # 是否以字典形式返回输出。
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 这里设定了模型输出格式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 调用bert模型，获取语言的编码输出。
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

        # 提取BERT输出的两个关键信息：针对每个词的编码（sequence_output）和整个句子的总体理解（pooled_output）
        sequence_output = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']

        # 应用dropout，减少过拟合，有助于让模型的理解能力更加健壮和泛化。
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # 使用预设好的测试工具（线性分类器）得出机器人对于意图和每个词标记的预测
        token_logits = [self.token_heads[i](sequence_output) for i in range(self.token_head_num)]
        seq_logits = [self.seq_heads[i](pooled_output) for i in range(self.seq_head_num)]

        loss = None
        # 如果提供了真实的标记和序列标签，使用交叉熵损失函数来计算损失, 并相加
        if token_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 在神经网络中，特别是处理分类问题时常用的一种损失函数。它的作用是衡量模型预测的概率分布与真实情况的概率分布之间的差异，这个损失值越小，说明模型预测的结果越接近真实标签。
            # 对每个头部计算损失并累积。
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
        # 最后，组装所有输出信息，返回给用户查看或用于继续训练
        return {
            'loss': loss,   # 最终累积的损失，用于优化模型。
            'seq_logits': seq_logits,   # 意图预测结果
            'token_logits': token_logits,   # 词标记的预测结果
            'hidden_states': outputs.hidden_states, # BERT的隐层状态
            'attentions': outputs.attentions  # BERT的注意力权重
            }

# 特化于本案例中的意图和槽位识别任务
# 主要简化是通过设定单一的任务头部来减少模型复杂性
class JointBert(BertMultiHeadJointClassification):
    def __init__(self, config, intent_label_num, slot_label_num):
        # 只需要单一的意图和槽位标签数目，因此它只创建了一个意图识别头和一个槽位填充头
        super().__init__(config, [intent_label_num], [slot_label_num])
    
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
        # output_attentions, output_hidden_states 常用于模型分析（如可视化注意力权重）或者高级应用（如特征抽取），但在标准的训练和推理任务中不常用。
        # return_dict 参数用于控制输出格式，但如果默认只需处理损失、意图和槽位的识别结果，这个参数也就不必要了。
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        # 简化为只接受一个意图标签和一个槽位标签，对应于单个的意图和槽位识别任务
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
        # 简化了这个过程，实现了针对单个意图和槽位标签集的任务的损失计算和预测结果输出
        return {
            'loss': outputs['loss'],
            'intent_logits': outputs['seq_logits'][0],
            'slot_logits': outputs['token_logits'][0],
            'hidden_states': outputs['hidden_states'],
            'attentions': outputs['attentions']
        }
