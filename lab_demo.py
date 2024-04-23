# 导入必要的库和函数
# 用于加载和处理序列到序列的语言模型
import hiddenlayer as h
import argparse
import logging
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

from pruning_methods import *
# from roberta_train_demo import plot_small_value_ratios


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


# Define global variables
device = None
dataset = None
model = None
tokenizer = None
train_dataloader = None
valid_dataloader = None
optimizer = None
scheduler = None
loss_fn = None
dummy_input = None


def get_dataset(name):
    global dataset, model
    if name in ['axb', 'axg', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']:
        dataset = load_dataset("super_glue", name)
    else:
        dataset = load_dataset(name)
    if dataset['test']:
        # 获取分类信息
        num_labels = dataset["test"].features["label"].num_classes
        # 初始化分类头
        model.add_classification_head(name, num_labels=num_labels)
        model = model.to(device)
    else:
        logger.info("No test dataset available for this task.")


def preprocessing(task_name, configs=None):
    global device, dataset, model, tokenizer, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn
    logger.info("Start Preprocessing...")
    # 定义一些参数
    model_name_or_path = "roberta-base"
    output_dir = "out/"

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 添加lora和adapter
    # 加载预训练的序列到序列语言模型
    model = AutoAdapterModel.from_pretrained(model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print_trainable_parameters(model)
    # 加载数据集
    get_dataset(task_name)

    if configs == None:
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
    else:
        config_list = configs

    peft_config = ConfigUnion(*[config_list[i]
                              for i in range(len(config_list))])

    model.add_adapter(task_name, peft_config)
    model.train_adapter(task_name)
    model.set_active_adapters(task_name)

    model = model.to(device)

    print_trainable_parameters(model)
    if 'train' not in dataset:
        dataset = dataset['test'].train_test_split(test_size=0.5)

    # 定义训练和验证的数据加载器
    train_dataloader = DataLoader(
        dataset['train'], batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(
        dataset['test'], batch_size=16, shuffle=False)

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=1e-7)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)

    # 定义损失函数
    loss_fn = CrossEntropyLoss()
    logger.info("Start training...")


def train_epoch(epoch_num):
    global dummy_input
    # 重置优化器
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * 3)
    # 开始训练
    for epoch in range(epoch_num):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}", unit="batch"):
            text_fields = [
                field for field in dataset['test'].column_names if field not in ['label', 'idx']]
            batch_fields = [batch[field] for field in text_fields]

            inputs = tokenizer(*batch_fields,
                               return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {name: tensor.to(device) for name,
                      tensor in inputs.items()}  # 将输入数据移动到GPU上
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
            dataset['test'])
        for batch in valid_dataloader:
            with torch.no_grad():
                text_fields = [
                    field for field in dataset['test'].column_names if field not in ['label', 'idx']]
                batch_fields = [batch[field] for field in text_fields]

                inputs = tokenizer(*batch_fields,
                                   return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {name: tensor.to(device) for name,
                          tensor in inputs.items()}  # 将输入数据移动到GPU上
                if (dummy_input == None):
                    dummy_input = inputs
                batch['label'] = batch['label'].to(device)
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, batch['label'])
                total_loss += loss.item()
                total_correct += (outputs.logits.argmax(dim=-1)
                                  == batch['label']).sum().item()
        logger.info(
            f'Epoch {epoch+1} || Validation loss: {total_loss/total_count:.6f} || Validation accuracy: {total_correct/total_count:.6f}')


def get_trainable_parameters(model):
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.info(name)
            names.append(name)
    return names


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


def main():
    global model, dataset
    parser = argparse.ArgumentParser(description='Lab demo script')
    parser.add_argument('--run_baseline_lora', action='store_true',
                        help='Run the baseline: lora or adapter')
    parser.add_argument('--run_baseline_adapter', action='store_true',
                        help='Run the baseline: lora or adapter')

    args = parser.parse_args()
    # 创建一个日志记录器
    logger = logging.getLogger('lab')
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建一个文件处理器，将日志写入到文件中
    file_handler = logging.FileHandler('output.log', mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建一个格式器，定义日志的格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到日志记录器中
    logger.addHandler(file_handler)

    task_names = ['axb']
    task_names = ['rotten_tomatoes']
    # task_names = ['axg', 'cb', 'copa',
    #   'multirc', 'record', 'rte', 'wic', 'wsc']
    # 配置PEFT的参数
    lora_config = LoRAConfig(
        r=64,  # 设置LoRA的rank
        alpha=32,  # LoRA的alpha值，决定参数增加的数量
        dropout=0.1,  # LoRA层的dropout比例
        # leave_out=[6, 7, 8, 9, 10, 11],  # 指定需要转换的层 #important
    )

    bn_config = SeqBnConfig(
        reduction_factor=16,  # 设置瓶颈维度
        # leave_out=[6, 7, 8, 9, 10, 11]  # 指定需要转换的层 #important
    )

    configs = [lora_config, bn_config]

    if args.run_baseline_lora:
        for task in task_names:
            preprocessing(task, [lora_config])
            logger.info(f'Task:{task}')
            train_epoch(10)
    elif args.run_baseline_adapter:
        for task in task_names:
            model = None
            dataset = None
            preprocessing(task, [bn_config])
            logger.info(f'Task:{task}')
            train_epoch(10)
    else:
        for task in task_names:
            model = None
            dataset = None
            preprocessing(task, configs)
            torch.cuda.empty_cache()
            logger.info(f'Task:{task}')
            for i in range(1):
                train_epoch(3)
                
                prune_model(model, task_name=task, opts=[
                            'lora'], p_method='values_below_threshold', top_p=1, print_names=True)
                prune_model(model, task_name=task, opts=[
                            'adapter'], p_method='values_below_threshold', top_p=1, print_names=True)
                reinitialize_trainable_parameters(model)


if __name__ == "__main__":
    main()
