'''
run adapter baseline, the hyper parameters should be default
'''

from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
import torch
from pruning_methods import prune_model
from utils.gpu_memory_plot import get_free_gpu_memory

import argparse
import logging
import time
from copy import deepcopy
import os


def baseline_wrapper_single(search_list, ds_name, dataset, logger, configs):
    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run()
    peft_type = ''
    if 'adapter' in configs and 'lora' in configs:
        peft_type = 'lora&adapter'
    elif 'adapter' in configs:
        peft_type = 'adapter'
    elif 'lora' in configs:
        peft_type = 'lora'
    logger.info(f'Result {res} for {peft_type} {search_list}, {configs}')


def prune_wrapper_single(search_list,
                         ds_name,
                         dataset,
                         logger,
                         configs,
                         prune_method='gradient',
                         prune_turn=10):
    if 'adapter' in configs and 'lora' in configs:
        peft_type = 'lora&adapter'
    elif 'adapter' in configs:
        peft_type = 'adapter'
    elif 'lora' in configs:
        peft_type = 'lora'

    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    peft_type = ''
    if 'adapter' in configs and 'lora' in configs:
        peft_type = 'lora&adapter'
    elif 'adapter' in configs:
        peft_type = 'adapter'
    elif 'lora' in configs:
        peft_type = 'lora'

    prune_turn = 10
    logger.info(f'\tStart prune by {prune_method} for {prune_turn} times')

    origin_search_list = search_list
    search_list = deepcopy(origin_search_list)

    for _ in range(prune_turn):
        model = None
        gradients = None
        activations = None
        torch.cuda.empty_cache()
        logger.info(f'Start searching for lora:{search_list}')
        configs[peft_type] = search_list
        model = PEFTModel(configs, dataset).half()
        res, gradients, activations = model.run()
        logger.info(f'Result {res} for {peft_type} {search_list}')

        idx, idt = prune_model(
            model.model,
            task_name='my_module',
            opts=[peft_type],
            p_method=prune_method,
            top_p=12,
            print_names=True,
            gradients=gradients,
            activations=activations)
        logger.info(f'Pruned layer: {idx, idt}')
        search_list[int(idx)] = 0
