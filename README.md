# 基于NLP和意图识别的智能交互助手

**基于自然语言处理和意图识别的智能交互助手，能够理解和响应用户的页面交互指令，并执行相应的操作**

Intelligent Interaction Assistant（IIA)

## 基于 BERT 的中文指令意图和槽位联合识别模块

> 源码仓库：https://github.com/GSemir0418/IIA-algo
>
> BERT 预训练模型下载：https://huggingface.co/google-bert/bert-base-chinese/tree/main
>
> ​	仅需 `README.md`、`config.json`、`pytorch_model_bin`、`tokenizer.json`、`vocab.txt`

### 1 概述

意图识别（Intent Recognition）是自然语言处理（NLP）中的一个重要任务，它旨在确定用户输入的语句中所表达的意图或目的。简单来说，意图识别就是对用户的输入进行语义理解，以便更准确地回答用户的问题或提供相关的服务。

在 NLP 中，意图识别通常被视为一个分类问题，即通过将输入语句分类到预定义的意图类别中来识别其意图。这些类别可以是各种不同的任务、查询、请求等，例如搜索、购买、咨询、命令等。

下面是一个简单的例子来说明意图识别的概念：

> 用户输入： "帮我查下 id 为 4001 的全部数据”
> 意图识别：查询数据，id 字段为 4001

在这个例子中，通过将用户输入的语句分类到“查询”这个意图类别中，系统可以理解用户的意图并为其提供相关的服务

本项目面向公司调度排产软件，实现了一个基于 BERT 的意图（intent）和槽位（slots）联合预测模块。思路参考了 monologg，能够理解用户对工单数据页面的中文自然语言基本交互（CRUD）指令，识别指令意图，提供相关的服务并做出正确响应。

> [monologg/JointBERT: Pytorch implementation of JointBERT: "BERT for Joint Intent Classification and Slot Filling" (github.com)](https://github.com/monologg/JointBERT)
>
> 利用 `[CLS]` token 对应的 last hidden state 去预测整句话的 intent，并利用句子 tokens 的 last hidden states 做序列标注，找出包含 slot values 的 tokens。

#### 1.1 BERT

TODO

#### 1.2 运行环境

- Python 3.11.0

### 2 数据集

#### 2.1 数据集介绍

模型的训练主要依赖于三方面的数据：

1. 训练数据：训练数据以 `json` 格式给出，每条数据包括三个关键词：`text` 表示待检测的文本，`intent` 代表文本的类别标签，`slots` 是文本中包括的所有槽位以及对应的槽值，以字典形式给出。样例如下：

```json
{
    "text": "找出id为123的记录",
    "intent": "QUERY",
    "slots": {
      "id": "123"
    }
},
```

2. 意图标签：以 `txt` 格式给出，每行一个意图，未识别意图以 `[UNK]` 标签表示。样例如下：

```txt
[UNK]
QUERY
UPDATE
...
```

3. 槽位标签：与意图标签类似，以 `txt` 格式给出。包括三个特殊标签： `[PAD]`表示输入序列中的 padding token， `[UNK]` 表示未识别序列标签， `[O]` 表示没有槽位的 token 标签。对于有含义的槽位标签，又分为以 `B` 开头的槽位开始的标签，以及以`I_` 开头的其余槽位标记两种。

```txt
[PAD]
[UNK]
[O]
I_id
B_id
...
```

#### 2.2 数据集优化

以查询场景为例，本项目初步实现查询 id 字段的自然语言意图识别

以下为本项目准备与优化数据集保证的原则：

1. **数据多样性**：即使是同样的意图（如“查找id”）也应包含不同的表达方式、不同的字词排序和结构
2. **不同的槽位值**：为了使模型能够处理各种不同的输入，为每个槽位都应提供多种不同的值。例如，不仅仅是数字，还可能是字母数字组合的ID
3. **不同的槽位类型**：除了 id 字段，考虑是否可能存在其他类型的槽位，比如产量、名称、状态或者自定义的分类
4. **负样本**：包括一些不相关的语句或错位意图的例子。这样做可以帮助模型学会辨别不属于正常意图的句子
5. **噪音数据**：包括一些带有文本错误或语法不正确的句子，以便模型能够处理现实世界中不完美的输入
6. **数据平衡**：确保数据在不同槽位、意图和表达方式之间都是平衡的。不要让某个特定的槽位或表达方式过多，从而造成模型偏差
7. **数据扩增**：如果您觉得手动编写例句太麻烦，可以利用数据扩增工具来自动生成新的训练数据
8. **精确的槽位边界**：确保为槽位值定义的边界是精确的，不要包含任何无关的词语或符号
10. **数据标注一致性**：例如，如果使用 `id` 来表示某种标识，在所有样本中都坚持使用同一概念

#### 2.3 数据集预处理

TODO：data/exact 流程

### 3 模型训练流程

#### 3.1 训练方法

采用增量学习（incremental learning）或分阶段培训。这种方法通过逐步引入不同的概念和复杂性来训练模型，可以帮助模型更集中地学习特定的模式，并在训练初期减少混淆；同时确保在增加新知识和能力的同时，不会损失原有的准确性。此外，这种方式可以帮助定位问题，即如果引入新的数据导致性能下降，可以更容易地识别问题所在并进行相应的修正。

首先实现单意图单字段的训练

```python
python train.py
```

TODO train.py 的代码注释

主要参数如下：

```
python train.py\
       --cuda_devices 0\
       --tokenizer_path "bert-base-chinese"\
       --model_path "bert-base-chinese"\
       --train_data_path "path/to/data/train.json"\
       --intent_label_path "path/to/data/intent_labels.txt"\
       --slot_label_path "path/to/data/slot_labels.txt"\
       --save_dir "/path/to/saved_model/"\
       --batch_size 64\
       --train_epochs 5
```

#### 3.2 意图与槽位预测

训练结束后，我们通过在 `JointIntentSlotDetector` 类中加载训练好的模型进行意图与槽位预测。

```python

```

### 4 模型推理验证

```python
python predict.py
```

```python
from detector import JointIntentSlotDetector
import time

start1_time = time.perf_counter()
model = JointIntentSlotDetector.from_pretrained(
    # model_path='./save_model/bert-base-chinese',
    # tokenizer_path='./save_model/bert-base-chinese',
    # intent_label_path='./data/SMP2019/intent_labels.txt',
    # slot_label_path='./data/SMP2019/slot_labels.txt'
    model_path='./save_model/my-model',
    tokenizer_path='./save_model/my-model',
    intent_label_path='./data/op-data/intent_labels.txt',
    slot_label_path='./data/op-data/slot_labels.txt'
)
start2_time = time.perf_counter()
all_text = ['帮我查下id为4001的全部数据', '查ID是2的数据', '给我id为1001BC2的全部数据', '查ID4102', '急需看一下ID为345的那个记录是啥']
for i in all_text:
    print(model.detect(i))
end_time = time.perf_counter()
time1 = (end_time - start1_time) / 3600
time2 = (end_time - start2_time) / 3600
print("所有检测时间（包括加载模型）：", time1, "s", "除去模型加载时间：", time2, "s",
      "总预测数据量：", len(all_text), "平均预测一条的时间（除去加载模型）：", time2 / len(all_text), "s/条")
```

#### 4.1 推理结果

| text                            | result                                                       | standard                                                     |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 帮我查下id为4001的全部数据      | {'text': '帮我查下id为4001的全部数据', 'intent': 'QUERY', 'slots': {'id': ['4 0 0# # 1']}} | {'text': '帮我查下id为4001的全部数据', 'intent': 'QUERY', 'slots': {'id': ['4001']}} |
| 查id是2的数据                   | {'text': '查id是2的数据', 'intent': 'QUERY', 'slots': {'id': ['2']}} | {'text': '查id是2的数据', 'intent': 'QUERY', 'slots': {'id': ['2']}} |
| 给我id为1001bc2的全部数据       | {'text': '给我id为1001bc2的全部数据', 'intent': 'QUERY', 'slots': {'id': ['1 0 0 1# # b c# # 2']}} | {'text': '给我id为1001bc2的全部数据', 'intent': 'QUERY', 'slots': {'id': ['1001bc2']}} |
| 查id4102                        | {'text': '查id4102', 'intent': 'QUERY', 'slots': {}}         | {'text': '查id4102', 'intent': 'QUERY', 'slots': {}}         |
| 急需看一下id为345的那个记录是啥 | {'text': '急需看一下id为345的那个记录是啥', 'intent': 'QUERY', 'slots': {'id': ['3 4 5']}} | {'text': '急需看一下id为345的那个记录是啥', 'intent': 'QUERY', 'slots': {'id': ['345']}} |

#### 4.2 模型训练结果

| train_loss           | intent_avg | slot_avg |
| -------------------- | ---------- | -------- |
| 0.015221006236970425 | 1.0        | 1..0     |

### 5 基本使用

TODO

项目依赖包版本统计

```bash
pip freeze > requirements.txt
```

Dockerfile

```dockerfile
FROM python:3.11.0-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
# pip 源配置
```

启动项目

```bash
docker build -t iia-algo-model-training-image .
docker run -it --name iia-algo-model-training-container iia-algo-model-training-image
```

## 示例项目

TODO

将模型处理为服务

使用 Nextjs14 开发示例项目