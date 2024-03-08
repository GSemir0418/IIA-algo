# -*- coding:utf-8 -*-

class LabelDict:
    # 初始化这个字典。它接收一个标签列表，并保证每个标签都是唯一的。unk_label 参数指定了未知标签，它用于处理在标签列表之外的情况。
    def __init__(self, labels, unk_label='[UNK]'):
        self.unk_label = unk_label
        if unk_label not in labels:
            self.labels = [unk_label] + labels
        else:
            self.labels = labels

        assert len(self.labels) == len(set(self.labels)), "ERROR: repeated labels appeared!"

    # 使得 LabelDict 对象可以像字典或列表一样被索引。它支持三种类型的索引：
# 如果输入是一个列表，它将迭代这个列表，对每个元素执行索引操作。
# 如果输入是字符串，则返回该字符串对应的索引值；如果字符串不在标签列表中，则返回未知标签的索引。
# 如果输入是一个整数，则返回对应索引的标签字符串。
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]
        elif isinstance(idx, str):
            if idx in self.labels:
                return self.labels.index(idx)
            else:
                return self.labels.index(self.unk_label)
        elif isinstance(idx, int):
            return self.labels[idx]

        print("Warning: unknown indexing type!")
        return None
    # 返回字典中标签的总数
    def __len__(self):
        return len(self.labels)
    # 将字典中所有的标签保存到一个文件中，每个标签一行
    def save_dict(self, save_path):
        with open(save_path, 'w', encoding="utf-8") as f:
            f.write('\n'.join(self.labels))
    # 是对 __getitem__ 方法的封装，它接收一个标签或标签列表并返回相应的索引或索引列表。
    def encode(self, labels):
        return self.__getitem__(labels)
    # 也是对 __getitem__ 方法的封装，但是它是用索引来获取对应的标签。
    def decode(self, labels):
        return self.__getitem__(labels)
    # 一个类方法，用来从文件中加载标签列表。它读取指定路径的文件，获取其中的内容，并将其行分割成标签列表。
    @classmethod
    def load_dict(cls, load_path, **kwargs):
        with open(load_path, 'r', encoding="utf-8") as f:
            labels = f.read().strip('\n').split('\n')

        return cls(labels, **kwargs)
