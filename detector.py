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

        # 如果没有提供 mask 参数，就创建一个全是1的列表，长度和 input_ids 一样。1代表我们关心这个位置的数据。
        if mask is None:
            mask = [1 for _ in range(len(input_ids))]

        # 把槽位的名称和值加到最终结果中
        def add_new_slot_value(results, slot_name, slot_value):
            # 如果槽位的名称或者值是空的，就什么也不做，返回当前的结果
            if slot_name == "" or slot_value == "":
                return results
            # 槽位的名称已经在最终结果的列表里了，就把这个新的值加到对应的列表里。如果不在，就创建一个新的列表，并加入这个值。
            if slot_name in results:
                results[slot_name].append(slot_value)
            else:
                results[slot_name] = [slot_value]
            return results

        # BERT类模型在面对未知或少见的单词时会破坏单词成最基本的元素。如果模型预测时也是基于这些最小单元（subtokens）的，那么在预测结束后，原始的槽位数据需要被重建。这个重建的过程可能会引入一些原本不存在的分隔符，比如空格和井号。
        
        # 逐一检查 slot_labels 中的每个槽位标签
        for i, slot_label in enumerate(slot_labels):
            # 如果当前位置的 mask 值是0，就跳过这次循环，不处理当前这个位置的数据。
            if mask[i] == 0:
                continue
            # 如果槽位标签的前两个字符是 'B_'，这表示一个槽位的开始
            if slot_label[:2] == 'B_':
                # 取出 'B_' 之后的部分，这部分是槽位的名称
                slot_name = slot_label[2:]
                # 如果这个槽位名称已经在 unfinished_slots 里了，说明我们找到了一个完整的槽位信息，将它加入到 results 中
                if slot_name in unfinished_slots:
                    results = add_new_slot_value(results, slot_name, unfinished_slots[slot_name])
                # 然后将当前的 input_id 转换回文本形式，作为这个槽位的起始值放入 unfinished_slots 中
                unfinished_slots[slot_name] = self.tokenizer.decode(input_ids[i])
            # 如果槽位标签的前两个字符是 'I_'，这表示槽位信息的中间部分
            elif slot_label[:2] == 'I_':
                # 也是取出 'I_' 之后的部分，作为槽位的名称
                slot_name = slot_label[2:]
                # 如果当前的槽位名称已经存在于 unfinished_slots 中，并且它的值不为空，就继续把当前的 input_id 转换的文本加到已有的槽位值的后面
                if slot_name in unfinished_slots and len(unfinished_slots[slot_name]) > 0:
                    unfinished_slots[slot_name] += self.tokenizer.decode(input_ids[i])

        # 最后一个循环是处理 unfinished_slots 中剩余的槽位信息
        for slot_name, slot_value in unfinished_slots.items():
            # 对于每一个未完成的槽位，如果它的值不为空，就把它加入到最终的 results 中
            if len(slot_value) > 0:
                results = add_new_slot_value(results, slot_name, slot_value)
        # 返回 results，里面包含了所有提取出来的槽位名称和它们对应的值
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

    # 根据概率最高的槽位 id 取出对应的槽位内容
    def _predict_slot_labels(self, slot_probs):
        """
        slot_probs : probability of a batch of tokens into slot labels, [batch, seq_len, slot_label_num], numpy array
        """
        slot_ids = np.argmax(slot_probs, axis=-1)
        return self.slot_dict[slot_ids.tolist()]

    # 根据概率最高的意图 id 取出对应的意图内容
    # intent_probs 参数是一个包含概率值的数组，每个概率值都对应一个特定的意图:[[3.1652924e-04 9.9968350e-01]]
    def _predict_intent_labels(self, intent_probs):
        """
        intent_labels : probability of a batch of intent ids into intent labels, [batch, intent_label_num], numpy array
        """
        # 这一行代码使用 np.argmax 函数来找出每一批（batch）概率值中最高的那个。简单来说，就是看看在所有可能的意图中，哪一个是最有可能的。axis=-1 表示我们在数组的最后一个维度上找最大值。
        intent_ids = np.argmax(intent_probs, axis=-1)
        # 最后，这行代码把之前找到的每一个最大概率值对应的编号（ID），转换成现实世界中的意图名称。它用这些编号去查找一个事先建立好的意图字典（intent_dict），得到人类可读的意图标签。
        return self.intent_dict[intent_ids.tolist()]

    # 类的主要功能方法，它接受一系列文本（用户输入的句子），并返回一个包含检测到的意图和槽位的列表
    def detect(self, text, str_lower_case=True):
        """
        text : list of string, each string is a utterance from user
        """
        # def clean_slot_values(slots):
        #     cleaned_slots = {}
        #     for slot_name, slot_values in slots.items():
        #         # 列表推导式，针对槽位值列表中的每个值进行处理
        #         cleaned_values = [''.join(value.split()) for value in slot_values]  # 删除空格
        #         cleaned_values = [value.replace('#', '') for value in cleaned_values]  # 删除井号
        #         cleaned_slots[slot_name] = cleaned_values
        #     return cleaned_slots
        
        # 检查输入是不是一个字符串，如果是，就把它包装成列表。这样不管输入是一个句子还是很多句子，代码都能用同样的方式处理
        list_input = True
        if isinstance(text, str):
            text = [text]
            list_input = False

        # 统一小写处理
        if str_lower_case:
            text = [t.lower() for t in text]

        # 计算了输入文本的数量，也就是批次大小
        batch_size = len(text)
        # 使用分词器（tokenizer）转换文本。这可能包括添加填充（padding），使所有文本的长度一致，这样处理起来更方便
        inputs = self.tokenizer(text, padding=False)
        # Bert 推理
        # 这块使用 PyTorch 库执行一个不需梯度信息的块（主要是出于性能和内存利用的考虑）。它将分词后的文本输入到模型中，做出预测。
        with torch.no_grad():
            outputs = self.model(input_ids=torch.tensor(inputs['input_ids']).long().to(self.device))

        # 获取文本意图的原始分数（logits）和各个槽位的原始分数。
        intent_logits = outputs['intent_logits']
        slot_logits = outputs['slot_logits']

        # 把原始分数（logits）转化为概率，这样我们可以知道每个意图或槽位有多确定
        intent_probs = torch.softmax(intent_logits, dim=-1).detach().cpu().numpy()
        slot_probs = torch.softmax(slot_logits, dim=-1).detach().cpu().numpy()

        # 得到槽位标注结果（只是标注结果）
        # [['[PAD]', '[O]', '[O]', '[O]', '[O]', '[O]', '[O]', 'B_id', 'I_id', '[O]', '[O]', '[O]', '[O]', '[O]', '[PAD]']]
        slot_labels = self._predict_slot_labels(slot_probs)
        # 得到意图识别结果 ['QUERY']
        intent_results = self._predict_intent_labels(intent_probs)

        # 根据预测的槽位标签，提取文本中的实际槽位值
        slot_results = self._extract_slots_from_labels(inputs['input_ids'], slot_labels, inputs['attention_mask'])

        # 创建一个输出列表，包括原始文本、预测的意图以及检测到的槽位
        outputs = [{'text': text[i], 'intent': intent_results[i], 'slots': slot_results[i]}
                   for i in range(batch_size)]
        
        # 如果最初的输入不是列表（只有一个句子），就只返回第一个结果
        if not list_input:
            return outputs[0]
        return outputs

# 主运行部分 
if __name__ == '__main__':
    # model_path = '../saved_models/jointbert-SMP2019/model/model_epoch2'
    # tokenizer_path = '../saved_models/jointbert-SMP2019/tokenizer/'
    # model_path = 'bert-base-chinese'
    # tokenizer_path = 'bert-base-chinese'
    # intent_path = 'data/SMP2019/intent_labels.txt'
    # slot_path = 'data/SMP2019/slot_labels.txt'
    model_path='./save_model/my-model'
    tokenizer_path='./save_model/my-model'
    intent_label_path='./data/op-data/intent_labels.txt'
    slot_label_path='./data/op-data/slot_labels.txt'

    model = JointIntentSlotDetector.from_pretrained(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        intent_label_path=intent_label_path,
        slot_label_path=slot_label_path
    )

    text = '帮我查下id为4001的全部数据'
    print(model.detect(text))
    # while True:
    #     text = input("input: ")
    #     print(model.detect(text))
