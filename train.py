# -*- coding:utf-8 -*-

# 使用了预训练的BERT模型作为基础，对其进行微调以应用于特定的NLU任务
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from datasets import IntentSlotDataset
from models import JointBert
from tools import save_module, split_data

# dev 函数负责评估模型在验证数据集上的性能
# 禁用梯度计算，这样可以减少内存消耗并加速计算。
# 通过模型预测意图和槽位，然后根据模型输出计算准确率。
# 最后，函数返回总的开发集准确率，以及单独的意图准确率和槽位准确率。
def dev(model, val_dataloader, device, slot_dict):
    # 将模型设置为评估模式。
    model.eval()
    intent_acc, slot_acc = 0, 0
    all_true_intent, all_pred_intent = [], []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            input_ids, intent_labels, slot_labels = batch

            outputs = model(
                input_ids=torch.tensor(input_ids).long().to(device),
                intent_labels=torch.tensor(intent_labels).long().to(device),
                slot_labels=torch.tensor(slot_labels).long().to(device)
            )

            intent_probs = torch.softmax(outputs["intent_logits"], dim=-1).detach().cpu().numpy()
            slot_probs = torch.softmax(outputs["slot_logits"], dim=-1).detach().cpu().numpy()
            slot_ids = np.argmax(slot_probs, axis=-1)
            intent_ids = np.argmax(intent_probs, axis=-1)
            slot_ids = slot_ids.tolist()
            intent_ids = intent_ids.tolist()

            slot_ids = [[slot_dict[i] for i in line] for line in slot_ids]
            slot_labels = [[slot_dict[i] for i in line] for line in slot_labels]

            all_true_intent.extend(intent_labels)
            all_pred_intent.extend(intent_ids)

            intent_acc += accuracy_score(intent_labels, intent_ids)
            slot_acc += accuracy_score(slot_labels, slot_ids)

    intent_avg, slot_avg = intent_acc / len(val_dataloader), slot_acc / len(val_dataloader)
    dev_acc = intent_avg + slot_avg
    return dev_acc, intent_avg, slot_avg

# 包含整个训练过程的逻辑，它接收一个包含所有训练参数的对象 args，包括模型路径、数据路径、训练超参数
def train(args):
    # 模型保存位置
    # model_save_dir = args.save_dir + "/" + args.model_path.split("/")[-1]
    model_save_dir = args.save_dir 

    # 读取命令行参数中提供的槽位和意图标签路径，创建字典用以编解码标签
    with open(args.slot_label_path, 'r') as f:
        slot_labels = f.read().strip('\n').split('\n')
    slot_dict = dict(zip(range(len(slot_labels)), slot_labels))

    # -----------set cuda environment-------------
    # 判断使用CPU还是GPU训练
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------load tokenizer 加载 BERT 分词器-----------
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    save_module(tokenizer, model_save_dir)

    # -----------load data 加载数据-----------------
    # 读取训练和验证数据，并将数据封装成适用于训练的格式
    train_data, val_data = split_data(args.train_data_path, args.train_val_data_split)

    train_dataset = IntentSlotDataset.load_from_path(
        data_content=train_data,
        intent_label_path=args.intent_label_path,
        slot_label_path=args.slot_label_path,
        tokenizer=tokenizer
    )

    val_dataset = IntentSlotDataset.load_from_path(
        data_content=val_data,
        intent_label_path=args.intent_label_path,
        slot_label_path=args.slot_label_path,
        tokenizer=tokenizer
    )

    # -----------load model and dataset 加载模型和数据集-----------
    # 初始化BERT模型用于联合意图和槽位填充任务
    model = JointBert.from_pretrained(
        args.model_path,
        slot_label_num=train_dataset.slot_label_num,
        intent_label_num=train_dataset.intent_label_num
    )
    model = model.to(device).train()

    # 准备数据加载器，将数据打乱并批量载入模型
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.batch_collate_fn)

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=val_dataset.batch_collate_fn)

    # -----------calculate training steps-----------
    # 计算训练的总步数
    if args.max_training_steps > 0:
        total_steps = args.max_training_steps
    else:
        total_steps = len(train_dataset) * args.train_epochs // args.gradient_accumulation_steps // args.batch_size

    print('calculated total optimizer update steps : {}'.format(total_steps))

    # -----------prepare optimizer and schedule------------
    # 准备优化器和学习率调度器
    parameter_names_no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 这些参数会被应用常规的权重衰减（由 args.weight_decay 指定）
        {'params': [
            para for para_name, para in model.named_parameters()
            if not any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': args.weight_decay},
        # 这些参数的权重衰减被设置为0
        {'params': [
            para for para_name, para in model.named_parameters()
            if any(nd_name in para_name for nd_name in parameter_names_no_decay)
        ],
            'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # 学习率变化（更新）方式
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    # -----------training-------------
    # 初始化最佳准确率
    max_acc = 0
    # 开始训练循环(epoch 是训练的轮次)
    for epoch in range(args.train_epochs):
        # 初始化总损失: 在开始新的轮次之前，把总损失设置为0，后面会用它来计算这个轮次的平均损失
        total_loss = 0
        # 设置模型为训练模式
        model.train()
        # 遍历数据加载器中的批次
        for step, batch in enumerate(train_dataloader):
            # 处理批次内数据: 从批次中提取出输入的ID、意图标签和槽位标签
            input_ids, intent_labels, slot_labels = batch
            # 模型推理和计算损失: 将数据送入模型进行处理，并返回结果。模型输出一个字典，其中包含了损失，即当前模型的表现
            outputs = model(
                input_ids=torch.tensor(input_ids).long().to(device),
                intent_labels=torch.tensor(intent_labels).long().to(device),
                slot_labels=torch.tensor(slot_labels).long().to(device)
            )
            # 累加损失值
            loss = outputs['loss']
            total_loss += loss.item()
            # 处理梯度累积: 如果你设置梯度累积步骤大于1，则在执行反向传播之前，先将损失除以累积步骤数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 损失反向传播: 反向传播过程中，计算损失关于模型参数的梯度
            loss.backward()
            # 梯度裁剪和更新参数: 检查是否到了更新梯度的步骤。裁剪梯度可以防止训练出现不稳定的情况。调用优化器和学习率调度器来更新参数，并重置梯度
            if step % args.gradient_accumulation_steps == 0:
                # 用于对梯度进行裁剪，以防止在神经网络训练过程中出现梯度爆炸的问题。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        # 计算这个轮次的平均损失: 把总损失除以批次数量来获取这个轮次的平均损失
        train_loss = total_loss / len(train_dataloader)
        # 评估模型在验证集上的性能: 使用之前定义的 dev 函数来计算模型在验证集上的准确率
        dev_acc, intent_avg, slot_avg = dev(model, val_dataloader, device, slot_dict)
        # 保存性能最佳的模型
        flag = False
        if max_acc < dev_acc:
            max_acc = dev_acc
            flag = True
            save_module(model, model_save_dir)
        # 输出训练信息
        print(f"[{epoch}/{args.train_epochs}] train loss: {train_loss}  dev intent_avg: {intent_avg} "
              f"def slot_avg: {slot_avg} save best model: {'*' if flag else ''}")
    # 训练结束后的最后评估: 在所有训练轮次结束后，再次评估模型在验证集上的表现
    dev_acc, intent_avg, slot_avg = dev(model, val_dataloader, device, slot_dict)
    # 输出最终模型的性能和保存位置
    print("last model dev intent_avg: {} def slot_avg: {}".format(intent_avg, slot_avg))
    print("模型保存位置：" + model_save_dir)

# 脚本的入口通过if __name__ == '__main__':判断，来确保只有在脚本被直接运行时，才会执行训练过程。
if __name__ == '__main__':
    # 模型的保存位置、数据文件，以及训练参数等都可以通过命令行参数灵活地设置。参数的设置使用argparse.ArgumentParser来进行，它使得用户可以轻松通过命令行来配置训练的各种细节
    parser = argparse.ArgumentParser()

    # environment parameters
    parser.add_argument("--cuda_devices", type=str, default='0', help='set cuda device numbers')

    # model parameters
    parser.add_argument("--model_path", type=str, default='./bert-base-chinese', help="pretrained model loading path")

    # data parameters
    # parser.add_argument("--train_data_path", type=str, default='data/SMP2019/data.json', help="training data path")
    parser.add_argument("--train_data_path", type=str, default='data/op-data/data.json', help="training data path")
    parser.add_argument("--train_val_data_split", type=float, default=0.8, help="training data and val data split rate")
    # parser.add_argument("--slot_label_path", type=str, default='data/SMP2019/slot_labels.txt', help="slot label path")
    parser.add_argument("--slot_label_path", type=str, default='data/op-data/slot_labels.txt', help="slot label path")
    # parser.add_argument("--intent_label_path", type=str, default='data/SMP2019/intent_labels.txt', help="intent label path")
    parser.add_argument("--intent_label_path", type=str, default='data/op-data/intent_labels.txt', help="intent label path")

    # training parameters
    # parser.add_argument("--save_dir", type=str, default='./save_model', help="directory to save the model")
    parser.add_argument("--save_dir", type=str, default='./save_model/my-model/', help="directory to save the model")
    # 告诉优化器你打算训练多少步。每一步就是一次梯度更新，即调整一次模型的权重。如果这个值设置大于0，那它就是训练的最大步数；如果设置为0，那通常意味着将根据数据量和其他参数自动决定训练步数。
    parser.add_argument("--max_training_steps", type=int, default=0, help='max training step for optimizer(优化器的最大训练步数), if larger than 0')
    # 决定了累积多少个更新步骤后才进行一次权重更新。这在显存不足以容纳很大批量的数据时非常有用。它允许模型累积一定数量的梯度，然后一次性更新，这样可以模拟大批量数据的训练效果。
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of updates steps to accumulate before performing a backward() pass.(执行 backward() 之前累积的更新步数数量)")
    # 一次训练你要喂给模型多少数据。一般来说，批量越大，训练越稳定，但是显存占用也越高
    parser.add_argument("--batch_size", type=int, default=32, help='training data batch size')
    # 这表示你会把整个训练数据集过多少遍。一个epoch意味着每一个数据点都被模型看过一次。
    parser.add_argument("--train_epochs", type=int, default=20, help='training epoch number')
    # 学习率是优化器在调整模型权重时的步长大小。太高可能导致模型学习过快，错过最佳解；太低则可能导致训练过程缓慢甚至停滞。
    parser.add_argument("--learning_rate", type=float, default=5e-5, help='learning rate')
    # 这是Adam优化算法中的一个很小的数值，用来提高数值稳定性。一般默认 1e-8
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="epsilon for Adam optimizer")
    # 这个参数在训练开始时用来逐渐增加学习率，以帮助模型更稳定地训练。
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup step number")
    # 衰减率用来降低模型权重的大小，以防止过拟合。
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay rate")
    # 这是梯度剪切的一个阈值，用来防止在训练过程中出现梯度爆炸的问题。
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="maximum norm for gradients")

    args = parser.parse_args()

    train(args)
