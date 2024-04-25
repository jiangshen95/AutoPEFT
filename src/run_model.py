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
        if configs.get('base_lora'):
            lora_config = LoRAConfig(
                r=next(x for x in configs['lora']['ranks'] if x != 0),
                alpha=r,
                dropout=0.1,
            )
        if configs.get('base_adapter'):
            adapter_config = SeqBnConfig(
                reduction_factor=next(
                    x for x in configs['adapter']['bn'] if x != 0),)
        if configs.get('lora'):
            lora_config = LoRAConfig(
                r=next(x for x in configs['lora']['ranks'] if x != 0),
                alpha=r,
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
