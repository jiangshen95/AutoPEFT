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
import yaml


def baseline_wrapper_single(search_list, ds_name, dataset, logger, args):
    configs = PEFTSearchSpace(args).get_config()
    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    torch.cuda.empty_cache()
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run()
    peft_type = ''
    if 'adapter' in configs and 'lora' in configs:
        peft_type = 'lora&adapter'
    elif 'adapter' in configs:
        peft_type = 'adapter'
    elif 'lora' in configs:
        peft_type = 'lora'
    logger.info(f'Final-Result {res} for {search_list};{configs}')
    model = None
    torch.cuda.empty_cache()
    return res


def baseline_wrapper_double(search_list1, search_list2, ds_name, dataset,
                            logger, args):
    configs = PEFTSearchSpace(args).get_config()
    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    torch.cuda.empty_cache()
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run()
    peft_type = 'lora & adapter'

    logger.info(
        f'Final-Result {res} for lora:{search_list1} adapter:{search_list2}')
    model = None
    torch.cuda.empty_cache()
    return res


def prune_wrapper_single(search_list,
                         ds_name,
                         dataset,
                         logger,
                         args,
                         prune_method='gradient',
                         prune_turn=10):
    configs = PEFTSearchSpace(args).get_config()

    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    peft_type = ''
    if 'adapter' in configs and 'lora' in configs:
        peft_type = 'lora&adapter'
    elif 'adapter' in configs:
        peft_type = 'adapter'
    elif 'lora' in configs:
        peft_type = 'lora'

    logger.info(f'\tStart prune by {prune_method} for {prune_turn} times')

    origin_search_list = search_list
    search_list = deepcopy(origin_search_list)

    for _ in range(prune_turn):
        model = None
        gradients = None
        activations = None
        torch.cuda.empty_cache()
        logger.info(f'Start searching for lora:{search_list}')
        args.lora = search_list
        configs = PEFTSearchSpace(args).get_config()
        model = PEFTModel(configs, dataset).half()
        res, gradients, activations = model.run()
        logger.info(f'Mid-Result {res} for {peft_type} {search_list}')

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

    model = None
    gradients = None
    activations = None
    torch.cuda.empty_cache()
    return search_list


def prune_wrapper_double(search_list1,
                         search_list2,
                         ds_name,
                         dataset,
                         logger,
                         args,
                         prune_method='gradient',
                         prune_turn=10):
    configs = PEFTSearchSpace(args).get_config()

    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    peft_type = 'lora & adapter'

    logger.info(f'\tStart prune by {prune_method} for {prune_turn} times')

    sl1 = deepcopy(search_list1)
    sl2 = deepcopy(search_list2)

    for _ in range(prune_turn):
        model = None
        gradients = None
        activations = None
        torch.cuda.empty_cache()
        logger.info(f'Start searching for lora:{sl1} adapter:{sl2}')
        args.lora = sl1
        args.adapter = sl2
        configs = PEFTSearchSpace(args).get_config()
        model = PEFTModel(configs, dataset).half()
        res, gradients, activations = model.run()
        logger.info(
            f'Mid-Result {res} for {peft_type} lora:{sl1} adapter:{sl2}')

        idx, idt = prune_model(
            model.model,
            task_name='my_module',
            opts=['lora', 'adapter'],
            p_method=prune_method,
            top_p=12,
            print_names=True,
            gradients=gradients,
            activations=activations)
        logger.info(f'Pruned layer: {idx, idt}')
        if idt == 'lora':
            sl1[int(idx)] = 0
        else:
            sl2[int(idx)] = 0

    model = None
    gradients = None
    activations = None
    torch.cuda.empty_cache()
    return sl1, sl2
