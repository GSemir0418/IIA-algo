# -*- coding:utf-8 -*-
# 提供了一些实用的功能，主要用于处理文件系统操作和数据处理
import json
import os
import random

# 函数确保工作目录路径存在，如果不存在则创建它
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# 函数负责保存训练好的模型。函数首先通过check_path函数检查给定的保存目录是否存在并创建（如果需要）。然后，它检查模型是否被封装在 DataParallel 或 DistributedDataParallel 中。如果是这样，它将获得原始模型，因为这两个封装器都使用模型的 .module 属性。最后，save_pretrained 方法被用于将模型的状态字典保存到指定目录。
def save_module(model, save_dir):
    check_path(save_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)
    
# 函数用于将数据集分割成训练集和验证集。函数首先从给定路径加载JSON格式的数据集文件。然后，它根据意图标签将数据集中的项分组，在这里意图标签被作为字典的键。通过遍历每个意图的数据项列表，它在每个意图下按照split_rate分割训练和验证数据。在这个过程中，它确保了数据集的分割是意图平衡的，即保证了每种意图的样本都被划分到训练集和验证集中。最后，它打乱了训练集和验证集中的数据项，并返回它们。
def split_data(data_path, split_rate):
    with open(data_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    train_data, val_data = [], []
    intent = {}
    for item in data:
        if item['intent'] not in intent.keys():
            intent[item['intent']] = [item]
        else:
            intent[item['intent']].append(item)

    for key, value in intent.items():
        train = value[:int(len(value) * split_rate)]
        val = value[int(len(value) * split_rate):]
        train_data.extend(train)
        val_data.extend(val)

    random.shuffle(train_data)
    random.shuffle(val_data)
    return train_data, val_data
