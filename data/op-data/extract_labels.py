import json
# 用来从一个JSON文件中提取信息，然后创建两个文本文件，一个是保存意图标签的 intent_labels.txt，另一个是保存槽位标签的 slot_labels.txt。

# 这一行是用来判断这个python文件是被直接执行还是被当成模块导入到其他文件中。如果是直接执行，那么下面的代码就会运行。
if __name__ == '__main__':
    # 这行代码打开了名为 data.json 的文件，准备读取里面的内容，指定了文件编码为 utf-8，这样可以正确读取含有特殊字符的文本。
    with open('data.json', 'r', encoding="utf-8") as f:
        # 这行代码读取了打开的文件并将JSON格式的文件内容转化成了Python能理解的数据结构（一般是字典或列表）并存储在变量 data 中
        data = json.load(f)

    # QUERY 1598
    # DELETE 1598
    # UPDATE 1598
    # 代码定义了意图标签 intent_labels 和槽位标签 slot_labels 的初始列表。列表中的 [UNK], [PAD] 和 [O] 分别代表未知标签、填充标签和普通标签。
    intent_labels = ['[UNK]']
    slot_labels = ['[PAD]','[UNK]', '[O]']
    for item in data:
        # 对于每个项目中的 intent 字段，如果它不在 intent_labels 列表中，就把它添加进去
        if item['intent'] not in intent_labels:
            intent_labels.append(item['intent'])
        # 对于项目中的 slots 字典，代码遍历了每个槽位名称和槽位值。如果 'B_'+slot_name（表示槽位的开始）不在 slot_labels 列表中，就把 'I_'+slot_name（表示槽位的内部）和 'B_'+slot_name 两个标签都添加到 slot_labels 列表中。
        for slot_name, slot_value in item['slots'].items():
            if 'B_'+slot_name not in slot_labels:
                slot_labels.extend(['I_'+slot_name, 'B_'+slot_name])
    # 打开（如果不存在则创建）一个文件叫 slot_labels.txt，用写入模式，并把 slot_labels 列表中的元素转化为文本写入文件，每个标签占一行。
    with open('slot_labels.txt', 'w') as f:
        f.write('\n'.join(slot_labels))
    # 打开（如果不存在则创建）一个文件叫 intent_labels.txt，并把 intent_labels 列表中的元素写入文件，每个标签占一行。
    with open('intent_labels.txt', 'w') as f:
        f.write('\n'.join(intent_labels))
