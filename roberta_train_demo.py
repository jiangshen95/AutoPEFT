# 导入必要的库和函数
# 用于加载和处理序列到序列的语言模型
import re
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
# 导入必要的库
from adapters import AdapterTrainer
from transformers import TrainingArguments, EvalPrediction
import numpy as np
from transformers import RobertaConfig
from transformers import RobertaTokenizer
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from adapters import AutoAdapterModel, AdapterArguments, AdapterTrainer, AdapterConfig, ConfigUnion, LoRAConfig, SeqBnConfig, PrefixTuningConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


print("Start Preprocessing...")
# 定义一些参数
model_name_or_path = "roberta-base"
task_name = "rotten_tomatoes"
output_dir = "out/roberta-base-rotten_tomatoes/"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
# dataset = load_dataset("glue", task_name)
dataset = load_dataset("rotten_tomatoes")

# 添加lora和adapter
# 加载预训练的序列到序列语言模型
model = AutoAdapterModel.from_pretrained(model_name_or_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

print_trainable_parameters(model)

# 配置PEFT的参数
lora_config = LoRAConfig(
    r=32,  # 设置LoRA的rank
    alpha=32,  # LoRA的alpha值，决定参数增加的数量
    dropout=0.1,  # LoRA层的dropout比例
    # leave_out=[6, 7, 8, 9, 10, 11],  # 指定需要转换的层 #important
)


bn_config = SeqBnConfig(
    reduction_factor=16,  # 设置瓶颈维度
    # leave_out=[6, 7, 8, 9, 10, 11]  # 指定需要转换的层 #important
)

config_list = [lora_config, bn_config]
peft_config = ConfigUnion(*[config_list[i] for i in range(len(config_list))])


model.add_adapter("rotten_tomatoes", peft_config)
model.train_adapter("rotten_tomatoes")
model.set_active_adapters("rotten_tomatoes")

model.add_classification_head(task_name, num_labels=5)
model = model.to(device)

print_trainable_parameters(model)

# 定义训练和验证的数据加载器
train_dataloader = DataLoader(dataset['train'], batch_size=64, shuffle=True)
valid_dataloader = DataLoader(
    dataset['validation'], batch_size=64, shuffle=False)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)

# 定义损失函数
loss_fn = CrossEntropyLoss()
print("Start training...")


def train_epoch(epoch_num):
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}", unit="batch"):
            inputs = tokenizer(batch['text'], return_tensors='pt',
                               padding=True, truncation=True, max_length=512)
            inputs = {name: tensor.to(device)
                      for name, tensor in inputs.items()}  # 将输入数据移动到GPU上
            batch['label'] = batch['label'].to(device)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, batch['label'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        # 验证
        model.eval()
        total_loss, total_correct, total_count = 0, 0, len(
            dataset['validation'])
        for batch in valid_dataloader:
            with torch.no_grad():
                inputs = tokenizer(
                    batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {name: tensor.to(device) for name,
                          tensor in inputs.items()}  # 将输入数据移动到GPU上
                batch['label'] = batch['label'].to(device)
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, batch['label'])
                total_loss += loss.item()
                total_correct += (outputs.logits.argmax(dim=-1)
                                  == batch['label']).sum().item()
        print(f'Epoch {epoch+1} || Validation loss: {total_loss/total_count:.6f} || Validation accuracy: {total_correct/total_count:.6f}')


def get_trainable_parameters(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(name)
            names.append(name)
    return names


def group_parameters_by_prefix(names, opt='', print_names=False):
    groups = {}
    names = [
        name for name in names if task_name in name and 'head' not in name and opt in name]
    for name in names:
        # 分割参数名，获取前缀
        prefix = name.split(task_name)[0]
        prefix = prefix.replace('query.', '').replace('value.', '')
        # 如果前缀已经在字典中，就将参数名添加到对应的列表中
        if prefix in groups:
            groups[prefix].append(name)
        # 否则，创建一个新的列表
        else:
            groups[prefix] = [name]
    if print_names:
        for prefix, names in groups.items():
            print(f"{prefix}:")
            for name in names:
                print(f"  {name}")
    return groups


def find_group_with_most_small_values(groups, model):
    max_small_values = 0
    max_group = None
    max_names = []
    for group, names in groups.items():
        num_small_values = 0
        for name in names:
            weights = model.state_dict()[name]
            num_small_values += torch.lt(torch.abs(weights),
                                         0.001).sum().item()
        if num_small_values > max_small_values:
            max_small_values = num_small_values
            max_group = group
            max_names = names
    return max_group, max_names, max_small_values


def plot_small_value_ratios(groups, model):
    group_names = []
    small_value_ratios = []
    for group, names in groups.items():
        num_small_values = 0
        total_values = 0
        for name in names:
            weights = model.state_dict()[name]
            num_small_values += torch.lt(torch.abs(weights),
                                         0.001).sum().item()
            total_values += weights.numel()
        small_value_ratios.append(num_small_values / total_values)
        group_names.append(group)

    plt.bar(group_names, small_value_ratios)
    plt.xlabel('Group')
    plt.ylabel('Ratio of small values')
    plt.title('Ratio of weights less than 0.001 in each group')
    plt.show()


def plot_total_parameters(groups, model):
    group_names = []
    total_parameters = []
    for group, names in groups.items():
        total_values = 0
        for name in names:
            weights = model.state_dict()[name]
            total_values += weights.numel()
        total_parameters.append(total_values)
        group_names.append(group)

    plt.bar(group_names, total_parameters)
    plt.xlabel('Group')
    plt.ylabel('Total parameters')
    plt.title('Total parameters in each group')
    plt.show()


def set_weights_to_zero_and_untrainable(group, model):
    for name in group:
        # 获取权重
        weights = model.state_dict()[name]
        # 将权重设置为全 0
        weights.zero_()
        # 将修改后的权重重新赋值给模型中的对应模块

        name = re.sub(r'\.(\d+)', r'[\1]', name)
        exec('model.'+name+'= Parameter(data=weights, requires_grad=False)')


def reinitialize_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1:  # bias
                torch.nn.init.zeros_(param)
            else:  # weights
                torch.nn.init.xavier_uniform_(param)


for i in range(5):
    # 剪枝训练循环
    train_epoch(5)

    # 删除lora
    # names = get_trainable_parameters(model)
    # groups = group_parameters_by_prefix(names, opt='lora', print_names=True)
    # max_group, max_names, max_small_values = find_group_with_most_small_values(
    #     groups, model)
    # print(
    #     f"The group with the most weights less than 0.001 is {max_group} with {max_small_values} such weights.")
    # plot_small_value_ratios(groups, model)
    # # plot_total_parameters(groups, model)
    # set_weights_to_zero_and_untrainable(max_names, model)

    # # 删除adapter
    # names = get_trainable_parameters(model)
    # groups = group_parameters_by_prefix(
    #     names, opt='adapter', print_names=False)
    # max_group, max_names, max_small_values = find_group_with_most_small_values(
    #     groups, model)
    # print(
    #     f"The group with the most weights less than 0.001 is {max_group} with {max_small_values} such weights.")
    # plot_small_value_ratios(groups, model)
    # # plot_total_parameters(groups, model)
    # set_weights_to_zero_and_untrainable(max_names, model)

    # 重新初始化可训练参数
    # TODO
    # reinitialize_trainable_parameters(model)
