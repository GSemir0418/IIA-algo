# -*- coding:utf-8 -*-

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
all_text = ['帮我查下id为4001的全部数据', '查ID是2的数据', '给我id为1001BC2的全部数据', '查ID4102', '急需看一下ID为345的那个记录是啥',
            '帮我查下编码为4001的全部数据', '查CODE是2的数据', '给我编码为1001BC2的全部数据', '查编码4102', '急需看一下code为345的那个记录是啥']
for i in all_text:
    print(model.detect(i))
end_time = time.perf_counter()
time1 = (end_time - start1_time) / 3600
time2 = (end_time - start2_time) / 3600
print("所有检测时间（包括加载模型）：", time1, "s", "除去模型加载时间：", time2, "s",
      "总预测数据量：", len(all_text), "平均预测一条的时间（除去加载模型）：", time2 / len(all_text), "s/条")
