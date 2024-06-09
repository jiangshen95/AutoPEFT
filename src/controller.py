from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
from pruning_methods import prune_model

import argparse
import logging
import random
import time

logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)  # 设置日志级别
file_handler = logging.FileHandler('output.log', mode='w')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('Start loading dataset')
dataset = PEFTDataset('rotten_tomatoes', test_size=0.2).get_dataset()

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


def wrapper(search_list):
    args.lora = search_list
    args.epochs = 5
    # args.adapter = search_list
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset)
    res = model.run()
    logger.info(f'Result for {search_list}: {res}')


def wrapper2(search_list, search_list2):
    args = parser.parse_args()

    args.lora = search_list
    args.adapter = search_list2
    args.epochs = 5
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset)
    res = model.run()
    logger.info(f'Result for {search_list, search_list2}: {res}')


search_list = [32] * 12
wrapper(search_list)

for _ in range(0):

    if (sum(search_list + search_list2) < total):
        break
    # logger.info(f'Start searching for {search_list}')
    args.lora = search_list
    args.adapter = search_list2
    args.epochs = 10
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset)
    res = model.run()
    logger.info(f'Result for {search_list, search_list2}: {res}')

    idx, idt = prune_model(
        model.model,
        task_name='my_module',
        opts=['lora', 'adapter'],
        p_method='minimum_weight',
        top_p=12,
        print_names=True)
    logger.info(f'Pruned layer: {idx, idt}')
    if idt == 'lora':
        search_list[int(idx)] = 0
    else:
        search_list2[int(idx)] = 0
