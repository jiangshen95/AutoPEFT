'''
this module is to run glue datast and test the performance of the pruning.
['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'qnli', 'rte', 'wnli'] are implemented.
['mnli','ax'] are not implemented due to the structure of the dataset.
for test, we can use small dataset by setting train_size and test_size.
'''
from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
import torch
from pruning_methods import prune_model
from utils.gpu_memory_plot import get_free_gpu_memory
from src.controllers.baseline_wrapper import baseline_wrapper_single, prune_wrapper_single

import argparse
import logging
import time
from copy import deepcopy
import os
import json

logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)  # 设置日志级别
time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
day_str = time.strftime('%Y-%m-%d', time.localtime())
output_dir = f'outputs/{day_str}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_handler = logging.FileHandler(
    f'outputs/{day_str}/output_{time_str}.log', mode='w')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--lora', type=int, nargs='+')
parser.add_argument('--adapter', type=int, nargs='+')
parser.add_argument('--base_lora', type=int)
parser.add_argument('--base_adapter', type=int)
parser.add_argument('--epochs', type=int)

searched_args = []
searched_res = []

logger.info('Start prune adapter')

to_load = ['qnli', 'rte', 'wnli', 'cola', 'sst2', 'mrpc', 'qqp']
prune_methods = [
    'zeros',
    'values_below_threshold',
    'snip',
    'minimum_weight',
    'activation',
    'gradient',
]
for ds_name in to_load:
    dataset = PEFTDataset(
        'glue', ds_name, train_size=2000, test_size=400).get_dataset()
    args = parser.parse_args(args=[])
    args.adapter = [128] * 24
    args.epochs = 3
    res_base = baseline_wrapper_single(
        search_list=[128] * 24,
        ds_name=ds_name,
        dataset=dataset,
        logger=logger,
        args=args,
    )
    res_methods = {}
    res_methods['base'] = res_base
    for prune_method in prune_methods:
        args = parser.parse_args(args=[])
        args.adapter = [128] * 24
        args.epochs = 1
        logger.info(f"Start testing {ds_name} with {prune_method}")
        res_search_list = prune_wrapper_single(
            search_list=args.adapter,
            ds_name=ds_name,
            dataset=dataset,
            logger=logger,
            args=args,
            prune_method=prune_method,
            prune_turn=6)
        args = parser.parse_args(args=[])
        args.adapter = res_search_list
        args.epochs = 3
        res = baseline_wrapper_single(
            search_list=res_search_list,
            ds_name=ds_name,
            dataset=dataset,
            logger=logger,
            args=args,
        )
        res_methods[prune_method] = res
    with open('results/final-adapter.json', 'a') as file:
        file.write(ds_name + '\n')
        file.write(json.dumps(res_methods) + '\n')
