'''The main function that runs the model
It can be used by 2 ways:
1. run 'python run_model.py --lora 64' in the terminal
2. use PEFTModel.run(args/configs) in other python files
'''

import argparse
import sys
import os
import torch
from torch.nn.parameter import Parameter
import numpy as np
import re
from adapters import AutoAdapterModel, AdapterArguments, AdapterTrainer, AdapterConfig, ConfigUnion, LoRAConfig, SeqBnConfig, PrefixTuningConfig
from transformers import RobertaTokenizer, TrainingArguments
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
from pruning_methods import get_trainable_parameters, group_parameters_by_prefix


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


class PEFTModel:
    '''Model wrapper
    '''

    configs = None
    model = None
    tokenizer = None
    trainer = None
    dataset = None
    model_name = 'roberta-base'
    task_name = 'mytask'

    def __init__(self, configs, dataset=None):
        '''
        configs: a dict of configs
        dataset: a dataset object
        '''
        self.configs = configs
        self.dataset = dataset
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)

        # calculate the number of labels
        num_labels = dataset["test"].features["label"].num_classes
        self.model.add_classification_head(
            self.task_name, num_labels=num_labels)

        lora_config = None
        adapter_config = None
        peft_config = None
        if configs.get('base_lora'):
            lora_config = LoRAConfig(
                r=next(x for x in configs['lora']['ranks'] if x != 0),
                alpha=next(x for x in configs['lora']['ranks'] if x != 0),
                dropout=0.1,
            )
        if configs.get('base_adapter'):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs['adapter']['bn'] if x != 0),)
        if configs.get('lora'):
            lora_config = LoRAConfig(
                r=next(x for x in configs['lora']['ranks'] if x != 0),
                alpha=next(x for x in configs['lora']['ranks'] if x != 0),
                dropout=0.1,
                leave_out=[
                    i for i, x in enumerate(configs['lora']['ranks']) if x == 0
                ],
            )
        if configs.get('adapter'):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs['adapter']['bn'] if x != 0),
                leave_out=[
                    i for i, x in enumerate(configs['adapter']['bn']) if x == 0
                ],
            )

        if lora_config and adapter_config:
            peft_config = ConfigUnion(lora_config, adapter_config)
        elif lora_config:
            peft_config = lora_config
        elif adapter_config:
            peft_config = adapter_config
        else:
            assert (0)

        self.model.add_adapter("my_module", config=peft_config)
        self.model.train_adapter("my_module")

        # if configs.get('lora'):
        #     names = get_trainable_parameters(self.model)
        #     groups = group_parameters_by_prefix(
        #         names, opts='lora', task_name="my_module")
        #     print(groups)

    def run(self):
        ''' tokenize the dataset and train the model
        '''
        assert (self.dataset)
        assert (self.model)
        assert (self.tokenizer)

        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"], truncation=True, padding="max_length")

        def preprocess_function1(examples):
            return self.tokenizer(
                [examples["sentence1"], examples["sentence2"]],
                truncation=True,
                padding="max_length")

        print(self.dataset['test'])
        if 'text' in self.dataset['test'].features:
            encoded_dataset_train = self.dataset['train'].map(
                preprocess_function, batched=True)
            encoded_dataset_val = self.dataset['test'].map(
                preprocess_function, batched=True)
        elif 'sentence1' in self.dataset[
                'test'].features and 'sentence2' in self.dataset[
                    'test'].features:
            encoded_dataset_train = self.dataset['train'].map(
                preprocess_function1, batched=True)
            encoded_dataset_val = self.dataset['test'].map(
                preprocess_function1, batched=True)
        else:
            assert (0)

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=4,
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

        self.trainer = AdapterTrainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset_train,
            eval_dataset=encoded_dataset_val,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()
        metrics = self.trainer.evaluate()
        print(metrics)
        return metrics

    def add_dataset(self, dataset):
        ''' add a dataset
        '''
        self.dataset = dataset

    def set_peft_group(self, group, index, value=0):
        ''' set the size of a PEFT module
        index='double' means to double the rank/bn of LoRA or adapter
        index='half' means to half the rank/bn of LoRA or adapter
        index='remove' means to remove
        index='set' means to set the rank to value
        '''
        group = [re.sub(r'\.(\d+)', r'[\1]', name) for name in group]
        exec('module=self.model.' + group[0], locals(
        ))  # save the var into locals(), which cannot be repetitive in function
        mo = locals()['module']

        origin_emb_size = mo.size()[1]
        origin_rank_size = mo.size()[0]

        target_rank_size = origin_rank_size
        if index == 'double':
            target_rank_size *= 2
        elif index == 'half':
            target_rank_size //= 2
        elif index == 'remove':
            target_rank_size = 0
        elif index == 'set':
            target_rank_size = value
        else:
            assert (0)

        if origin_rank_size == target_rank_size:
            return
        if target_rank_size == 0:
            assert (0)
            # TODO: remove the module

        for name in [name for name in group if 'lora_A' in name]:
            weights = torch.rand(target_rank_size, origin_emb_size)
            exec('self.model.' + name +
                 '=Parameter(data=weights, requires_grad=True)')

        for name in [name for name in group if 'lora_B' in name]:
            weights = torch.rand(origin_emb_size, target_rank_size)
            exec('self.model.' + name +
                 '=Parameter(data=weights, requires_grad=True)')

        for name in [
                name for name in group
                if 'adapter_down' in name and 'weight' in name
        ]:
            weights = torch.rand(target_rank_size, origin_emb_size)
            exec('self.model.' + name +
                 '=Parameter(data=weights, requires_grad=True)')

        for name in [
                name for name in group
                if 'adapter_up' in name and 'weight' in name
        ]:
            weights = torch.rand(origin_emb_size, target_rank_size)
            exec('self.model.' + name +
                 '=Parameter(data=weights, requires_grad=True)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', type=int, nargs='+')
    parser.add_argument('--adapter', type=int, nargs='+')
    parser.add_argument('--base_lora', type=int)
    parser.add_argument('--base_adapter', type=int)
    args = parser.parse_args()

    configs = PEFTSearchSpace(args).get_config()
    dataset = PEFTDataset('rotten_tomatoes', test_size=0.5).get_dataset()
    model = PEFTModel(configs, dataset)
    # model.run()
    names = get_trainable_parameters(model.model)
    groups = group_parameters_by_prefix(
        names, opts='adapter', task_name="my_module")
    sorted_groups = sorted(groups.items())
    print([name[1] for name in sorted_groups])

    print_trainable_parameters(model.model)
    model.set_peft_group(sorted_groups[0][1], 'half')
    model.set_peft_group(sorted_groups[1][1], 'half')
    model.set_peft_group(sorted_groups[2][1], 'half')
    model.set_peft_group(sorted_groups[3][1], 'half')

    print_trainable_parameters(model.model)
