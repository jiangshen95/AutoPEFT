from adapters import AutoAdapterModel, AdapterArguments, AdapterTrainer, AdapterConfig, ConfigUnion, LoRAConfig, SeqBnConfig, PrefixTuningConfig
import sys
from torch.utils.data import DataLoader
from datasets import load_dataset
from adapters.trainer import AdapterTrainer
from adapters.composition import Stack
from adapters import AutoAdapterModel
from transformers import RobertaTokenizer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch


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


# 加载模型和分词器
model = AutoAdapterModel.from_pretrained("roberta-base")
model.add_classification_head('rotten_tomatoes', num_labels=2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

lora_config = LoRAConfig(
    r=64,  # 设置LoRA的rank
    alpha=32,  # LoRA的alpha值，决定参数增加的数量
    dropout=0.1,  # LoRA层的dropout比例
    # leave_out=[6, 7, 8, 9, 10, 11],  # 指定需要转换的层 #important
)
# 添加LoRA适配器
model.add_adapter("lora", config=lora_config)
# model.set_active_adapters("lora")
model.train_adapter("lora")

print_trainable_parameters(model)

# 加载数据集
dataset = load_dataset('super_glue', 'axb')  # 请替换为你的数据集
print(dataset)
dataset = dataset['test'].train_test_split(test_size=0.5)
# 对数据集进行预处理


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


def preprocess_function1(examples):
    return tokenizer(
        examples["sentence1"], truncation=True, padding="max_length")


def preprocess_function2(examples):
    return tokenizer(
        examples["sentence2"], truncation=True, padding="max_length")


encoded_dataset_train = dataset['train'].map(preprocess_function1, batched=True)
encoded_dataset_train = dataset['train'].map(preprocess_function1, batched=True)

encoded_dataset_val = dataset['test'].map(preprocess_function2, batched=True)
encoded_dataset_val = dataset['test'].map(preprocess_function2, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = np.sum(preds == labels) / len(labels)
    return {'accuracy': acc}


# 创建Trainer
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset_train,
    eval_dataset=encoded_dataset_val,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
