一句话总结：

**基于自然语言处理和意图识别的智能交互助手，能够理解和响应用户的页面交互指令，并执行相应的操作**

Intelligent Interaction Assistant（IIA)

面向排产业务软件



# 意图识别模块

前言

意图识别（Intent Recognition）是自然语言处理（NLP）中的一个重要任务，它旨在确定用户输入的语句中所表达的意图或目的。简单来说，意图识别就是对用户的话语进行语义理解，以便更好地回答用户的问题或提供相关的服务。

在 NLP 中，意图识别通常被视为一个分类问题，即通过将输入语句分类到预定义的意图类别中来识别其意图。这些类别可以是各种不同的任务、查询、请求等，例如搜索、购买、咨询、命令等。

下面是一个简单的例子来说明意图识别的概念：

> 用户输入： "我想订一张从北京到上海的机票。
> 意图识别：预订机票。

在这个例子中，通过将用户输入的语句分类到“预订机票”这个意图类别中，系统可以理解用户的意图并为其提供相关的服务。

意图识别是 NLP 中的一项重要任务，它可以帮助我们更好地理解用户的需求和意图，从而为用户提供更加智能和高效的服务。

在智能对话任务中，意图识别是一种非常重要的技术，它可以帮助系统理解用户的输入，从而提供更加准确和个性化的回答和服务。



意图识别和槽位填充是对话系统中的基础任务。本仓库实现了一个基于 BERT 的意图（intent）和槽位（slots）联合预测模块。想法上实际与JoinBERT类似（GitHub：[BERT for Joint Intent Classification and Slot Filling](https://link.zhihu.com/?target=https%3A//github.com/monologg/JointBERT)），利用 `[CLS]` token 对应的 last hidden state 去预测整句话的 intent，并利用句子 tokens 的 last hidden states 做序列标注，找出包含 slot values 的 tokens。你可以自定义自己的意图和槽位标签，并提供自己的数据，通过下述流程训练自己的模型，并在 `JointIntentSlotDetector` 类中加载训练好的模型直接进行意图和槽值预测。



##### Bert 模型下载

Bert模型下载地址：[https://huggingface.co/google-bert/bert-base-chinese/tree/main](https://link.zhihu.com/?target=https%3A//huggingface.co/bert-base-chinese/tree/main)

## 数据集介绍

训练数据以 json 格式给出，每条数据包括三个关键词：`text` 表示待检测的文本，`intent` 代表文本的类别标签，`slots` 是文本中包括的所有槽位以及对应的槽值，以字典形式给出。

```json
{
    "text": "搜索西红柿的做法。",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {"ingredient": "西红柿"}
}
```

原始数据集：[https://conference.cipsc.org.cn/smp2019/](https://link.zhihu.com/?target=https%3A//conference.cipsc.org.cn/smp2019/)

本项目中在原始数据集中新增了部分数据，用来平衡数据。

### 数据准备



模型的训练主要依赖于三方面的数据：

1. 训练数据：训练数据以json格式给出，每条数据包括三个关键词：`text`表示待检测的文本，`intent`代表文本的类别标签，`slots`是文本中包括的所有槽位以及对应的槽值，以字典形式给出。在`data/`路径下，给出了[SMP2019](https://conference.cipsc.org.cn/smp2019/)数据集作为参考，样例如下：

```
{
    "text": "搜索西红柿的做法。",
    "domain": "cookbook",
    "intent": "QUERY",
    "slots": {"ingredient": "西红柿"}
}
```



1. 意图标签：以txt格式给出，每行一个意图，未识别意图以`[UNK]`标签表示。以SMP2019为例：

```
[UNK]
LAUNCH
QUERY
ROUTE
...
```



1. 槽位标签：与意图标签类似，以txt格式给出。包括三个特殊标签： `[PAD]`表示输入序列中的padding token, `[UNK]`表示未识别序列标签, `[O]`表示没有槽位的token标签。对于有含义的槽位标签，又分为以'B_'开头的槽位开始的标签, 以及以'I_'开头的其余槽位标记两种。

```
[PAD]
[UNK]
[O]
I_ingredient
B_ingredient
...
```

## 模型训练

```python
python train.py
```

### 训练过程



在数据准备完成后，可以使用以下命令进行模型训练，这里我们选择在`bert-base-chinese`预训练模型基础上进行finetune：

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



## 意图与槽位预测



训练结束后，我们通过在`JointIntentSlotDetector`类中加载训练好的模型进行意图与槽位预测。

```
from detector import JointIntentSlotDetector

model = JointIntentSlotDetector.from_pretrained(
    model_path='path/to/saved_model/model',
    tokenizer_path='path/to/saved_model/tokenizer/',
    intent_label_path='path/to/data/intent_labels.txt',
    slot_label_path='path/to/data/slot_labels.txt'
)
print(model.detect('西红柿的做法是什么'))
```

## 模型推理

```python
python predict.py
```



场景：

id code quantity status name 查询

准备用于**查询**数据的意图识别和槽位填充模型的样本集

生成与优化数据集

1. **数据多样性**：请确保您的数据集具有足够的多样性。即使是同样的意图（如“查找id”）也应包含不同的表达方式、不同的字词排序和结构。
2. **不同的槽位值**：为了使您的模型能够处理各种不同的输入，为每个槽位都应提供多种不同的值。例如，不仅仅是数字，还可能是字母数字组合的ID。
3. **不同的槽位类型**：您已经包括了id、code和status。考虑是否可能存在其他类型的槽位，比如时间、位置或者自定义的分类。
4. **负样本**：包括一些不相关的语句或错位意图的例子。这样做可以帮助模型学会辨别不属于“QUERY”意图的句子。
5. **噪音数据**：包括一些带有文本错误或语法不正确的句子，以便模型能够处理现实世界中不完美的输入。
6. **数据平衡**：确保数据在不同槽位、意图和表达方式之间都是平衡的。不要让某个特定的槽位或表达方式过多，从而造成模型偏差。
7. **数据扩增**：如果您觉得手动编写例句太麻烦，可以利用数据扩增工具来自动生成新的训练数据。
8. **精确的槽位边界**：确保为槽位值定义的边界是精确的，不要包含任何无关的词语或符号。
9. **适应性**：如果您的模型要部署在特定的环境中，确保数据集反映了目标环境的用例和语言。
10. **数据标注一致性**：保持数据标注的一致性很重要。例如，如果您使用"id"来表示某种标识，在所有样本中都坚持使用同一概念。





pip3 install tqdm torch transformers

pip3 install scikit-learn

pip3 install seqeval

```bash
pip freeze > requirements.txt
```

数据预处理 + train 脚本



初步测试：

intent_avg: 1.0

slot_avg: 0.97

train_loss: 0.09



增量学习（incremental learning）或分阶段培训。这种方法通过逐步引入不同的概念和复杂性来训练模型，有时这可以帮助模型更集中地学习特定的模式，并在训练初期减少混淆。

确保在增加新知识和能力的同时，不会损失原有的准确性。此外，这种方式可以帮助定位问题，即如果引入新的数据导致性能下降，可以更容易地识别问题所在并进行相应的修正。