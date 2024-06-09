import torch
from transformers import LlamaForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
from adapters import (
    AdapterArguments,
    AdapterConfig,
    AdapterTrainer,
    AutoAdapterModel,
    ConfigUnion,
    LoRAConfig,
    PrefixTuningConfig,
    SeqBnConfig,
)

# 加载腐烂番茄数据集
dataset = load_dataset("rotten_tomatoes")

# 分割数据集为训练和验证集
train_dataset = dataset['train']
eval_dataset = dataset['validation']

# 加载Llama3-8b模型和分词器
model_name = "meta-llama/Meta-Llama-3-8B"
model = AutoAdapterModel.from_pretrained(model_name, num_labels=2).half()

# 使用正确的分词器类
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token

# 定义数据处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 应用数据处理函数
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate()
