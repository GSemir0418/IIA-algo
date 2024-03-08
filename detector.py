# -*- coding:utf-8 -*-

import torch # 这是一个开源的机器学习库，广泛用于计算机视觉和自然语言处理等领域。
# torch 提供了一个强大的张量计算框架，以及丰富的神经网络层和优化算法，用于构建和训练深度学习模型
import numpy as np # 强大的多维数组对象以及一系列用于操作这些数组的工具 主要用于处理模型的输出，如将概率值转换为槽位和意图的预测标签

from transformers import BertTokenizer # 用于BERT模型的分词器，能够将文本拆分为模型可以处理的令牌（或称“token”，类似于单词或子单词单位）

from models import JointBert # 这是一个用于联合意图和槽位识别的神经网络模型
# 用于管理标签和序列化/反序列化标签字典的模块，可能提供了将标签映射到索引（用于训练和预测过程中）和反向映射的功能。
from labeldict import LabelDict

# 利用了一个预训练好的BERT模型，来理解和预测文本的含义

class JointIntentSlotDetector:
    def __init__(self, model, tokenizer, intent_dict, slot_dict, use_cuda=True):
        self.model = model # 存储着传入的BERT模型，这个模型被训练用来同时识别意图和槽位
        self.tokenizer = tokenizer # 是BERT分词器，用于将输入的文本转换成BERT模型能理解的格式
        self.intent_dict = intent_dict
        self.slot_dict = slot_dict
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu" # 运行时使用的设备，可以是CPU或者当CUDA可用时的GPU，这样可以加速模型的推理计算
        self.model.to(self.device)
        self.model.eval()
    # 用于加载预训练的模型和其他必要的组件
    @classmethod
    def from_pretrained(cls, model_path, tokenizer_path, intent_label_path, slot_label_path, **kwargs):
        intent_dict = LabelDict.load_dict(intent_label_path)
        slot_dict = LabelDict.load_dict(slot_label_path)

        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        model = JointBert.from_pretrained(
            model_path,
            slot_label_num=len(slot_dict),
            intent_label_num=len(intent_dict))

        return cls(model, tokenizer, intent_dict, slot_dict, **kwargs)
    # 处理单句输入，根据预测出的槽位标签序列，来识别文本中的槽位。例如，它会将 B_location 和 I_location 标签与相应的词组相匹配，以找出表示“位置”的词组。
    def _extract_slots_from_labels_for_one_seq(self, input_ids, slot_labels, mask=None):
        results = {}
        unfinished_slots = {}  # dict of {slot_name: slot_value} pairs
        if mask is None:
            mask = [1 for _ in range(len(input_ids))]

        def add_new_slot_value(results, slot_name, slot_value):
            if slot_name == "" or slot_value == "":
                return results
            if slot_name in results:
                results[slot_name].append(slot_value)
            else:
                results[slot_name] = [slot_value]
            return results

        for i, slot_label in enumerate(slot_labels):
            if mask[i] == 0:
                continue
            # 检测槽位的第一字符（B_）开头
            if slot_label[:2] == 'B_':
                slot_name = slot_label[2:]  # 槽位名称 （B_ 后面）
                if slot_name in unfinished_slots:
                    results = add_new_slot_value(results, slot_name, unfinished_slots[slot_name])
                unfinished_slots[slot_name] = self.tokenizer.decode(input_ids[i])
            # 检测槽位的后面字符（I_）开头
            elif slot_label[:2] == 'I_':
                slot_name = slot_label[2:]
                if slot_name in unfinished_slots and len(unfinished_slots[slot_name]) > 0:
                    unfinished_slots[slot_name] += self.tokenizer.decode(input_ids[i])

        for slot_name, slot_value in unfinished_slots.items():
            if len(slot_value) > 0:
                results = add_new_slot_value(results, slot_name, slot_value)

        return results
    
    # 是一个批处理版本，它可以处理多个输入序列。它调用 _extract_slots_from_labels_for_one_seq 来为每个序列提取槽位。
    def _extract_slots_from_labels(self, input_ids, slot_labels, mask=None):
        """
        input_ids : [batch, seq_len]
        slot_labels : [batch, seq_len]
        mask : [batch, seq_len]
        """
        if isinstance(input_ids[0], int):
            return self._extract_slots_from_labels_for_one_seq(input_ids, slot_labels, mask)

        if mask is None:
            mask = [1 for id_seq in input_ids for _ in id_seq]

        return [
            self._extract_slots_from_labels_for_one_seq(
                input_ids[i], slot_labels[i], mask[i]
            )
            for i in range(len(input_ids))
        ]

    # 转换了模型对于槽位可能性的预测为确切的槽位标签。它基于模型输出中每个槽位的最高概率来决定预测标签。
    def _predict_slot_labels(self, slot_probs):
        """
        slot_probs : probability of a batch of tokens into slot labels, [batch, seq_len, slot_label_num], numpy array
        """
        slot_ids = np.argmax(slot_probs, axis=-1)
        return self.slot_dict[slot_ids.tolist()]

    # 类似地转换了意图预测，同样基于最高概率来选择意图标签。
    def _predict_intent_labels(self, intent_probs):
        """
        intent_labels : probability of a batch of intent ids into intent labels, [batch, intent_label_num], numpy array
        """
        intent_ids = np.argmax(intent_probs, axis=-1)
        return self.intent_dict[intent_ids.tolist()]

    # 类的主要功能方法，它接受一系列文本（用户输入的句子），并返回一个包含检测到的意图和槽位的列表。这个过程涉及到对文本进行分词、模型推理、结果转换为标签，并提取相关信息。方法内部先将文本转换（并且可能转为小写），然后将其提供给模型进行预测。返回的信息包括原始文本、预测的意图、以及识别出的槽位值
    def detect(self, text, str_lower_case=True):
        """
        text : list of string, each string is a utterance from user
        """
        list_input = True

        if isinstance(text, str):
            text = [text]
            list_input = False

        if str_lower_case:
            text = [t.lower() for t in text]

        batch_size = len(text)
        # 编码
        inputs = self.tokenizer(text, padding=True)
        # Bert 推理
        with torch.no_grad():
            outputs = self.model(input_ids=torch.tensor(inputs['input_ids']).long().to(self.device))

        intent_logits = outputs['intent_logits']
        slot_logits = outputs['slot_logits']

        intent_probs = torch.softmax(intent_logits, dim=-1).detach().cpu().numpy()
        slot_probs = torch.softmax(slot_logits, dim=-1).detach().cpu().numpy()

        # 得到槽位标注结果
        slot_labels = self._predict_slot_labels(slot_probs)
        # 得到意图识别结果
        intent_labels = self._predict_intent_labels(intent_probs)

        slot_values = self._extract_slots_from_labels(inputs['input_ids'], slot_labels, inputs['attention_mask'])

        outputs = [{'text': text[i], 'intent': intent_labels[i], 'slots': slot_values[i]}
                   for i in range(batch_size)]

        if not list_input:
            return outputs[0]

        return outputs

# 在主运行部分 (if __name__ == '__main__':)，代码设置了模型和分词器的路径、意图和槽位标签文件的路径，并且创建了一个 JointIntentSlotDetector 实例。接着，进入一个无限循环，允许用户多次输入文本，并打印出模型检测到的意图和槽位信息。
if __name__ == '__main__':
    # model_path = '../saved_models/jointbert-SMP2019/model/model_epoch2'
    # tokenizer_path = '../saved_models/jointbert-SMP2019/tokenizer/'
    model_path = 'bert-base-chinese'
    tokenizer_path = 'bert-base-chinese'
    intent_path = 'data/SMP2019/intent_labels.txt'
    slot_path = 'data/SMP2019/slot_labels.txt'
    model = JointIntentSlotDetector.from_pretrained(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        intent_label_path=intent_path,
        slot_label_path=slot_path
    )

    while True:
        text = input("input: ")
        print(model.detect(text))
