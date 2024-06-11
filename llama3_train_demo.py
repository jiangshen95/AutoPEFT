import torch
from transformers import LlamaForSequenceClassification, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import TrainerCallback
from adapters import LoRAConfig, AdapterTrainer, AutoAdapterModel
from src.dataset_wrapper import PEFTDataset

class GradientLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone().cpu().numpy()
            self.log_gradients(state.global_step, gradients)

    def log_gradients(self, step, gradients):
        # Replace this with your desired logging mechanism
        # Here, we simply print the gradients
        print(f"Step {step}:")
        for name, grad in gradients.items():
            print(f"Layer: {name}, Gradient: {grad}")

# 加载腐烂番茄数据集
dataset = PEFTDataset(
    'rotten_tomatoes', instructs=True, test_size=0.2).get_dataset()

# 分割数据集为训练和验证集
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(train_dataset[0])
print(eval_dataset)

# 加载Llama3-8b模型和分词器
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token

# 加载模型并更新配置
model = AutoAdapterModel.from_pretrained(model_name)
model.add_classification_head("rotten_tomatoes", num_labels=2)

# 设置分类头的标签数量
model.config.num_labels = 2

# 加载LoRA配置
adapter_config = LoRAConfig(
    r=16,
    alpha=32,
    dropout=0.1
)

# 添加并激活LoRA适配器
model.add_adapter("lora", config=adapter_config)
model.train_adapter("lora")

# 定义数据处理函数
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# 应用数据处理函数
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print(train_dataset[0])

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

model = model.half()
# 定义AdapterTrainer
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GradientLoggingCallback]  # 添加自定义的回调
)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate()
