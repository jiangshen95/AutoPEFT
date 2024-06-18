"""
this class try to record gradients and hessians, but cost too much time and memory.
should be dropped.
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
from utils.gpu_memory_plot import get_free_gpu_memory, get_gpu_info
from torch.nn.attention import SDPBackend, sdpa_kernel

from pruning_methods import *


class TrainerWithGrad:

    def __init__(self, model, args, train_dataset, eval_dataset,
                 compute_metrics, callbacks, tokenizer):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(
            model.to(self.device), device_ids=[0, 1, 2, 3])
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.tokenizer = tokenizer

        self.epoch_num = args.num_train_epochs
        self.train_batch_size = args.per_device_train_batch_size
        self.eval_batch_size = args.per_device_eval_batch_size

        self.dummy_input = None

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.eval_batch_size, shuffle=False)

        print(train_dataset, eval_dataset)
        print('finished init')

    def train(self):
        with sdpa_kernel(SDPBackend.MATH):
            optimizer = AdamW(self.model.parameters(), lr=1e-5)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.train_dataloader) * 3)
            params_with_grad = [
                param for param in self.model.parameters()
                if param.requires_grad
            ]
            names_with_grad = [
                name for name, param in self.model.named_parameters()
                if param.requires_grad
            ]
            for epoch in range(self.epoch_num):
                self.model.train()
                gradients = {}
                hessians = {}
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
                    # Ensure all parameters require gradients
                    # for param in self.model.parameters():
                    #     param.requires_grad = True
                    #     print(param)

                    batch['label'] = batch['label'].to(self.device)
                    outputs = self.model(**inputs)
                    loss_fn = CrossEntropyLoss()
                    loss = loss_fn(outputs.logits, batch['label'])
                    grads = torch.autograd.grad(
                        loss,
                        params_with_grad,
                        create_graph=True,
                        allow_unused=True)
                    loss.backward(retain_graph=True)

                    # 保存梯度信息
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         gradients[name] = gradients.setdefault(name, 0) + param.grad.detach().cpu().numpy()

                    for i, (name, param) in enumerate(
                            zip(names_with_grad, params_with_grad)):
                        if not param.requires_grad:
                            continue
                        print("start calculate hessian", i)
                        grad = grads[i]
                        gradients[name] = gradients.setdefault(
                            name, 0) + grad  # Record gradient

                        hessian_diag = []

                        for j in range(grad.numel()):
                            print("hessian", j)
                            grad2 = torch.autograd.grad(
                                grad.flatten()[j], param, retain_graph=True)[0]
                            hessian_diag.append(grad2.flatten()[j].item())

                        hessians[name] = hessians.setdefault(
                            name, 0) + torch.tensor(hessian_diag).reshape(
                                param.size())

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()

            return gradients, hessians

    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_count = 0, 0, len(self.eval_dataset)
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
                if (self.dummy_input == None):
                    self.dummy_input = inputs
                batch['label'] = batch['label'].to(self.device)
                outputs = self.model(**inputs)
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(outputs.logits, batch['label'])
                total_loss += loss.item()
                total_correct += (outputs.logits.argmax(
                    dim=-1) == batch['label']).sum().item()
        print(
            f'Epoch {self.epoch_num+1} || Validation loss: {total_loss/total_count:.6f} || Validation accuracy: {total_correct/total_count:.6f}'
        )
        return {
            'eval_loss': total_loss / total_count,
            'eval_accuracy': total_correct / total_count
        }
