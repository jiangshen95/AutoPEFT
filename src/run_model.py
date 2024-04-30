'''The main function that runs the model
It can be used by 2 ways:
1. run 'python run_model.py --lora 64' in the terminal
2. use PEFTModel.run(args/configs) in other python files
'''

import argparse
import sys
import os
import torch
import numpy as np
from adapters import AutoAdapterModel, AdapterArguments, AdapterTrainer, AdapterConfig, ConfigUnion, LoRAConfig, SeqBnConfig, PrefixTuningConfig
from transformers import RobertaTokenizer, TrainingArguments
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset


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
    model.run()
