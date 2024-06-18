from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
from pruning_methods import prune_model
from utils.gpu_memory_plot import get_free_gpu_memory

import argparse
import logging
import time
from copy import deepcopy
import os
import torch

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

logger.info('Start loading dataset')
dataset = PEFTDataset('glue', 'cola', test_size=1, train_size=1).get_dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--lora', type=int, nargs='+')
parser.add_argument('--adapter', type=int, nargs='+')
parser.add_argument('--base_lora', type=int)
parser.add_argument('--base_adapter', type=int)
args = parser.parse_args()

searched_args = []
searched_res = []

# baseline
logger.info('Start baseline')


def wrapper(search_list, idt='lora'):
    global model, res, configs, gradients, activations
    args = parser.parse_args(args=[])
    if idt == 'lora':
        args.lora = search_list
    elif idt == 'adapter':
        args.adapter = search_list
    args.epochs = 1
    args.instructs = 1
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset).half()
    res, gradients, activations = model.run()
    logger.info(f'Result {res} for {search_list}')
    model = None
    gradients = None
    activations = None
    torch.cuda.empty_cache()


def wrapper2(search_list, search_list2):
    global model, res, configs, gradients, activations
    args = parser.parse_args(args=[])
    args.lora = search_list
    args.adapter = search_list2
    args.epochs = 1
    args.instructs = 1
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset)
    res, gradients, activations = model.run()
    logger.info(f'Result for {search_list, search_list2}: {res}')


# logger.info("Start adapter baseline")
# wrapper([32] * 16, 'adapter')

# logger.info('Start lora baseline')
# wrapper([32] * 32, 'lora')

prune_turn = 10
logger.info('-----Start gradient test------')

origin_search_list = [32] * 32
origin_search_list2 = [32] * 32
search_list = deepcopy(origin_search_list)
search_list2 = deepcopy(origin_search_list2)

for _ in range(prune_turn):
    model = None
    gradients = None
    activations = None
    torch.cuda.empty_cache()
    args = parser.parse_args(args=[])
    logger.info(f'Start searching for lora:{search_list}')
    args.lora = search_list
    args.epochs = 1
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset).half()
    res, gradients, activations = model.run()
    logger.info(f'Result {res} for {search_list}')

    idx, idt = prune_model(
        model.model,
        task_name='my_module',
        opts=['lora'],
        p_method='gradient',
        top_p=12,
        print_names=True,
        gradients=gradients,
        activations=activations)
    logger.info(f'Pruned layer: {idx, idt}')
    if idt == 'lora':
        search_list[int(idx)] = 0
    elif idt == 'adapter':
        search_list2[int(idx)] = 0
