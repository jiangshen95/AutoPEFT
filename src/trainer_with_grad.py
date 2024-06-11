"""
this class is to record gradients when training.
have the same function as trainer
"""
import argparse
import logging
import re
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
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


class TrainerWithGrad:

    def __init__(self, model, args, train_dataset, eval_dataset,
                 compute_metrics, callbacks, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.tokenizer = tokenizer

        self.epoch_num = args.num_train_epochs
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.eval_batch_size, shuffle=False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * 3)
        for epoch in range(self.epoch_num):
            self.model.train()
            gradients = {}
            for batch in tqdm(
                    self.train_dataloader,
                    desc=f"Training epoch {epoch+1}",
                    unit="batch"):
                text_fields = [
                    field for field in self.train_dataset.column_names
                    if field not in ['label', 'idx']
                ]
                batch_fields = [batch[field] for field in text_fields]

                inputs = self.tokenizer(
                    *batch_fields,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512)
                inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in inputs.items()
                }
                batch['label'] = batch['label'].to(self.device)
                outputs = self.model(**inputs)
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(outputs.logits, batch['label'])
                loss.backward()

                # 保存梯度信息
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.detach().cpu().numpy()

                print(gradients)

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

        def evaluate():
            self.model.eval()
            total_loss, total_correct, total_count = 0, 0, len(
                self.eval_dataset)
            for batch in self.eval_dataloader:
                with torch.no_grad():
                    text_fields = [
                        field for field in self.eval_dataset.column_names
                        if field not in ['label', 'idx']
                    ]
                    batch_fields = [batch[field] for field in text_fields]

                    inputs = self.tokenizer(
                        *batch_fields,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512)
                    inputs = {
                        name: tensor.to(self.device)
                        for name, tensor in inputs.items()
                    }  # 将输入数据移动到GPU上
                    if (dummy_input == None):
                        dummy_input = inputs
                    batch['label'] = batch['label'].to(self.device)
                    outputs = self.model(**inputs)
                    loss = loss_fn(outputs.logits, batch['label'])
                    total_loss += loss.item()
                    total_correct += (outputs.logits.argmax(
                        dim=-1) == batch['label']).sum().item()
            print(
                f'Epoch {epoch+1} || Validation loss: {total_loss/total_count:.6f} || Validation accuracy: {total_correct/total_count:.6f}'
            )
