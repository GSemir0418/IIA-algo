# -*- coding:utf-8 -*-
# 告诉Python解释器这个文件使用UTF-8编码。

# 使用BERT分词器处理原始数据，并为训练准备好意图和槽位的数据集，这可以用于后续的机器学习模型训练。

import json # 处理JSON文件
from tqdm import tqdm # 在控制台显示进度条

from torch.utils.data import Dataset # 创建数据集的父类
from transformers import BertTokenizer # 文本转换成BERT模型能理解的形式

from labeldict import LabelDict # 处理标签字典

# 为文本中的每个单词或词组赋予一个标签，以表明它是不是一个特定的“槽位”。如果文本中的词和给定的模式匹配，我们就给这个词标上一个特定的标签，比如B_location表示位置的开始，I_location表示位置的内部。如果没有匹配，就标记为普通词[O]。
def get_slot_labels(text, slots, tokenizer):
    """
    text : a string of text
    slots : a dict of {slot_label: [text_pattern1, text_pattern2, ..., text_patternK]}
    """
    text_tokens = tokenizer.tokenize(text)

    slot_labels = []
    i = 0
    while i < len(text_tokens):
        slot_matched = False
        for slot_label, slot_values in slots.items():
            if slot_matched:
                break

            if isinstance(slot_values, str):  # if slots is {slot_label: slot_value}
                slot_values = [slot_values]

            for text_pattern in slot_values:
                pattern_tokens = tokenizer.tokenize(text_pattern)
                if "".join(text_tokens[i: i + len(pattern_tokens)]) == "".join(pattern_tokens):
                    slot_matched = True
                    slot_labels.extend(['B_' + slot_label] + ['I_' + slot_label] * (len(pattern_tokens) - 1))
                    i += len(pattern_tokens)
                    break

        if not slot_matched:
            slot_labels.append('[O]')
            i += 1

    return slot_labels

# 用于存储处理后的数据，包括分词后的输入文本、槽位标签和意图标签
class IntentSlotDataset(Dataset):
    def __init__(self, raw_data, intent_labels, slot_labels, tokenizer):
        # 初始化数据集，包括转换意图和槽位的标签，并将原始数据转换为模型训练时需要的格式
        super().__init__()
        self.intent_label_dict = LabelDict(intent_labels)
        self.slot_label_dict = LabelDict(slot_labels)

        self.intent_label_num = len(self.intent_label_dict)
        self.slot_label_num = len(self.slot_label_dict)

        self.data = []
        for item in tqdm(raw_data):
            # 实体标注 '请帮我打开uc' --》 ['[O]', '[O]', '[O]', '[O]', '[O]', 'B_name']
            slot_labels = get_slot_labels(item['text'], item['slots'], tokenizer)
            # ['[O]', '[O]', '[O]', '[O]', '[O]', 'B_name'] --》 转换为标签
            slot_ids = self.slot_label_dict.encode(['[PAD]'] + slot_labels + ['[PAD]'])
            intent_id = self.intent_label_dict[item['intent']]  # 意图标签
            input_ids = tokenizer.encode(item['text'])  # 文本编码

            assert len(input_ids) == len(slot_ids), "slot label seq has different length than input seq"

            self.data.append({
                "input_ids": input_ids,
                "slot_ids": slot_ids,
                "intent_id": intent_id
            }
            )
        print('Finished processing all data.')

        # 把数据打包成一个批次，并填充，以确保每个批次中的数据具有相同的长度，这对于要在神经网络中使用它们是必要的。
        def batch_collate_fn(batch_data):
            batch_intent_ids = [item['intent_id'] for item in batch_data]
            max_seq_length = max([len(item['input_ids']) for item in batch_data])
            batch_input_ids = [item['input_ids'] + [0] * (max_seq_length - len(item['input_ids'])) for item in batch_data]
            batch_slot_ids = [item['slot_ids'] + [0] * (max_seq_length - len(item['slot_ids'])) for item in batch_data]

            return batch_input_ids, batch_intent_ids, batch_slot_ids

        self.batch_collate_fn = batch_collate_fn

    # 允许你加载储存在文件系统中的训练数据和标签文件
    @classmethod
    def load_from_path(cls, data_content, intent_label_path, slot_label_path, **kwargs):
        """加载数据集"""
        # with open(data_path, 'r', encoding="utf-8") as f:
        #     raw_data = json.load(f)

        with open(intent_label_path, 'r') as f:
            intent_labels = f.read().strip('\n').split('\n')

        with open(slot_label_path, 'r') as f:
            slot_labels = f.read().strip('\n').split('\n')

        return cls(data_content, intent_labels, slot_labels, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
# 它加载了数据并创建了一个 IntentSlotDataset 实例。最后，它打印出数据集中的第一条数据和原始数据的第一条数据，作为检查。
    data_path = '/home/pengyu/workspace/intent-detect-core/data/intent_train_data.json'
    intent_label_path = 'data/intent_labels.txt'
    slot_label_path = 'data/slot_labels.txt'
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = IntentSlotDataset.load_from_path(
        data_path=data_path,
        intent_label_path=intent_label_path,
        slot_label_path=slot_label_path,
        tokenizer=tokenizer,
    )

    print(dataset[0])
    print(dataset.raw_data[0])
