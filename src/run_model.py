"""The main function that runs the model
It can be used by 2 ways:
1. run 'python run_model.py --lora 64' in the terminal
2. use PEFTModel.run(args/configs) in other python files
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
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
from transformers import RobertaTokenizer, TrainingArguments, AutoTokenizer, TrainerCallback, Trainer

from pruning_methods import get_trainable_parameters, group_parameters_by_prefix
from src.dataset_wrapper import PEFTDataset
from src.peft_search_space import PEFTSearchSpace
from sklearn.metrics import f1_score
from trainer_with_grad import TrainerWithGrad
import random

# 设置Python的随机种子
random.seed(42)

# 设置NumPy的随机种子
np.random.seed(42)

# 设置PyTorch的随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 为所有GPU设置种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


class GradientCaptureCallback(TrainerCallback):

    def __init__(self, model_trainer):
        self.model_trainer = model_trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} gradients:")
        for name, grad in self.model_trainer.gradients.items():
            print(f"{name}: {grad}")


class PEFTModel:
    """Model wrapper for PEFT

    Args:
        configs (dict): a dict of configs.
        trainer (Trainer): a trainer for model fine-tuning.
        dataset (Dataset): the training dataset.
        model_name (str): model name.
        task_name (str): task name.
        model (nn.Module): model to fune-tuning.
        tokenizer (RobertaTokenizer): tokenizer of language model.
    """

    def __init__(self, configs, dataset=None):
        """
        Args:
            configs: a dict of configs
            dataset: a dataset object
        """
        self.trainer = None
        self.model_name = "meta-llama/Meta-Llama-3-8B"
        self.task_name = "mytask"

        self.configs = configs
        self.dataset = dataset
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.instructs = 0
        self.epochs = 1
        self.gradients = {}

        # calculate the number of labels
        if hasattr(dataset["train"].features["label"], "num_classes"):
            num_labels = dataset["train"].features["label"].num_classes
        else:
            num_labels = len(set(dataset["train"]["label"]))
        print("number of label classes:", num_labels)
        self.model.add_classification_head(
            self.task_name, num_labels=num_labels)

        lora_config = None
        adapter_config = None
        peft_config = None
        if configs.get("base_lora"):
            lora_config = LoRAConfig(
                r=next(x for x in configs["lora"]["ranks"] if x != 0),
                alpha=next(x for x in configs["lora"]["ranks"] if x != 0),
                dropout=0.1,
            )
        if configs.get("base_adapter"):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs["adapter"]["bn"] if x != 0),)
        if configs.get("lora"):
            lora_config = LoRAConfig(
                r=next(x for x in configs["lora"]["ranks"] if x != 0),
                alpha=next(x for x in configs["lora"]["ranks"] if x != 0),
                dropout=0.1,
                leave_out=[
                    i for i, x in enumerate(configs["lora"]["ranks"]) if x == 0
                ],
            )
        if configs.get("adapter"):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs["adapter"]["bn"] if x != 0),
                leave_out=[
                    i for i, x in enumerate(configs["adapter"]["bn"]) if x == 0
                ],
            )

        if lora_config and adapter_config:
            peft_config = ConfigUnion(lora_config, adapter_config)
        elif lora_config:
            peft_config = lora_config
        elif adapter_config:
            peft_config = adapter_config
        else:
            assert 0

        self.model.add_adapter("my_module", config=peft_config)
        self.model.train_adapter("my_module")
        # self.model = self.model.half()

        if configs.get("lora"):
            names = get_trainable_parameters(self.model)
            groups = group_parameters_by_prefix(
                names, opts="lora", task_name="my_module")
            sorted_groups = sorted(groups.items())
            sorted_groups = [name[1] for name in sorted_groups]

            ranks = [r for r in configs["lora"]["ranks"] if r != 0]
            for group, r in zip(sorted_groups, ranks):
                self.set_peft_group(group, "set", r)

        if configs.get("adapter"):
            names = get_trainable_parameters(self.model)
            groups = group_parameters_by_prefix(
                names, opts="adapter", task_name="my_module")
            sorted_groups = sorted(groups.items())
            sorted_groups = [name[1] for name in sorted_groups]

            ranks = [r for r in configs["adapter"]["bn"] if r != 0]
            for group, r in zip(sorted_groups, ranks):
                self.set_peft_group(group, "set", r)

        if configs.get('epochs'):
            self.epochs = configs['epochs']
        else:
            self.epochs = 1

        if configs.get("instructs"):
            self.instructs = 1

        if configs.get("loss"):
            self.loss_type = configs["loss"]
            if (self.loss_type == "cross_entropy"):
                self.loss_fn = nn.CrossEntropyLoss()
            elif (self.loss_type == "mse"):
                self.loss_fn = nn.MSELoss()
            else:
                raise (f"loss type {self.loss_type} not supported")
        else:
            self.loss_type = "cross_entropy"
            self.loss_fn = nn.CrossEntropyLoss()

    def run(self):
        """tokenize the dataset and train the model"""
        assert self.dataset is not None
        assert self.model is not None
        assert self.tokenizer is not None

        self.tokenizer.pad_token = self.tokenizer.eos_token

        def preprocess_function(examples):
            print(examples)
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                #batched=True,
                max_length=128,
            )

        def preprocess_function1(examples):
            return self.tokenizer(
                [examples["sentence1"], examples["sentence2"]],
                truncation=True,
                padding="max_length",
            )

        # if "text" in self.dataset["test"].features:
        #     encoded_dataset_train = self.dataset["train"].map(
        #         preprocess_function, batched=True)
        #     encoded_dataset_val = self.dataset["test"].map(
        #         preprocess_function, batched=True)
        # elif ("sentence1" in self.dataset["test"].features and
        #       "sentence2" in self.dataset["test"].features):
        #     encoded_dataset_train = self.dataset["train"].map(
        #         preprocess_function1, batched=True)
        #     encoded_dataset_val = self.dataset["test"].map(
        #         preprocess_function1, batched=True)
        # else:
        #     assert 0

        # print(encoded_dataset_train.shape)
        # print(encoded_dataset_train)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            # evaluation_strategy="epoch",
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = np.sum(preds == labels) / len(labels)
            f1 = f1_score(labels, preds, average='weighted')
            final_score = 0.5 * acc + 0.5 * f1
            return {"accuracy": acc, "f1": f1, "final_score": final_score}

        self.trainer = TrainerWithGrad(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            compute_metrics=compute_metrics,
            callbacks=[GradientCaptureCallback(self)],
            tokenizer=self.tokenizer,
            loss_fn=self.loss_fn,
        )

        # Register hooks to capture gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad, name=name: self.gradients.update(
                        {name: grad.clone().cpu().detach().numpy()}))

        gradients, activations = self.trainer.train()
        metrics = self.trainer.evaluate()
        print(metrics)

        return metrics, gradients, activations

    def add_dataset(self, dataset):
        """add a dataset"""
        self.dataset = dataset

    def set_peft_group(self, group, index, value=0):
        """set the size of a PEFT module
        index='double' means to double the rank/bn of LoRA or adapter
        index='half' means to half the rank/bn of LoRA or adapter
        index='remove' means to remove
        index='set' means to set the rank to value
        """
        # exec(
        # "module=self.model." + group[0], locals()
        # )  # save the var into locals(), which cannot be repetitive in function
        # mo = locals()["module"]

        # NOTE: change to this
        self.model: nn.Module
        mo = self.model.get_parameter(
            group[0])  # or mo = self.model.__getattr__(group[0])
        group = [re.sub(r"\.(\d+)", r"[\1]", name) for name in group]

        origin_emb_size = mo.size()[1]
        origin_rank_size = mo.size()[0]

        target_rank_size = origin_rank_size
        if index == "double":
            target_rank_size *= 2
        elif index == "half":
            target_rank_size //= 2
        elif index == "remove":
            target_rank_size = 0
        elif index == "set":
            target_rank_size = value
        else:
            assert 0

        if origin_rank_size == target_rank_size:
            return
        if target_rank_size == 0:
            assert 0
            # TODO: remove the module

        for name in [name for name in group if "lora_A" in name]:
            # weights = torch.zeros(target_rank_size, origin_emb_size)
            # nn.init.kaiming_uniform_(
            #     weights, a=math.sqrt(5)
            # )  # TODO: use rand will unable to train, use kaiming init is weaker than default(0.74 vs 0.77 on rotten_tomatoes 4 epoch)
            weights = torch.zeros(target_rank_size, origin_emb_size)
            torch.nn.init.normal(
                weights, mean=0, std=1 / pow(origin_emb_size, 0.5))
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")

        for name in [name for name in group if "lora_B" in name]:
            weights = torch.zeros(origin_emb_size, target_rank_size)
            exec("self.model." + name +
                 "=nn.Parameter(data=weights, requires_grad=True)")
        # Maybe we do not need initial when reinitialed the whole model
        # for name in [
        #         name for name in group
        #         if "adapter_down" in name and "weight" in name
        # ]:
        #     weights = torch.zeros(target_rank_size, origin_emb_size)
        #     torch.nn.init.normal(
        #         weights, mean=0, std=1 / pow(origin_emb_size, 0.5))
        #     exec("self.model." + name +
        #          "=nn.Parameter(data=weights, requires_grad=True)")

        # for name in [
        #         name for name in group
        #         if "adapter_down" in name and "bias" in name
        # ]:
        #     weights = torch.zeros(target_rank_size)
        #     exec("self.model." + name +
        #          "=nn.Parameter(data=weights, requires_grad=True)")

        # for name in [
        #         name for name in group
        #         if "adapter_up" in name and "weight" in name
        # ]:
        #     weights = torch.zeros(origin_emb_size, target_rank_size)
        #     torch.nn.init.normal(
        #         weights, mean=0, std=1 / pow(origin_emb_size, 0.5))
        #     exec("self.model." + name +
        #          "=nn.Parameter(data=weights, requires_grad=True)")

        # for name in [
        #         name for name in group
        #         if "adapter_up" in name and "bias" in name
        # ]:
        #     weights = torch.zeros(origin_emb_size)
        #     exec("self.model." + name +
        #          "=nn.Parameter(data=weights, requires_grad=True)")

    def half(self):
        self.model = self.model.half()
        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=int, nargs="+")
    parser.add_argument("--adapter", type=int, nargs="+")
    parser.add_argument("--base_lora", type=int)
    parser.add_argument("--base_adapter", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--instructs")
    args = parser.parse_args()

    configs = PEFTSearchSpace(args).get_config()
    dataset = PEFTDataset("rotten_tomatoes", test_size=0.5).get_dataset()
    model = PEFTModel(configs, dataset)
    model.run()
