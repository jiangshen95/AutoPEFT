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

import argparse
import logging
import time
from copy import deepcopy
import os

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
args = parser.parse_args()

searched_args = []
searched_res = []

# baseline
logger.info('Start baseline')


def wrapper(search_list):
    global model, res, configs, gradients, dataset
    args.lora = search_list
    args.epochs = 1
    args.instructs = 0
    # args.adapter = search_list
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset).half()
    res, gradients, activations = model.run()
    logger.info(f'Result {res} for {search_list}')


to_load = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'qnli', 'rte', 'wnli']
for ds_name in to_load:
    logger.info(f"start testing {ds_name}")
    dataset = PEFTDataset(
        'glue', ds_name, train_size=600, test_size=200).get_dataset()
    wrapper([32] * 32)
    dataset = None
    model = None
    torch.cuda.empty_cache()

# prune_turn = 0
# logger.info('-----Start gradient test------')

# origin_search_list = [32] * 32
# search_list = deepcopy(origin_search_list)

# for _ in range(prune_turn):
#     logger.info(f'Start searching for {search_list}')
#     args.lora = search_list
#     args.epochs = 1
#     configs = PEFTSearchSpace(args).get_config()
#     model = PEFTModel(configs, dataset).half()
#     res, gradients = model.run()
#     logger.info(f'Result {res} for {search_list}')

#     idx, idt = prune_model(
#         model.model,
#         task_name='my_module',
#         opts=['lora'],
#         p_method='snip',
#         top_p=12,
#         print_names=True,
#         gradients=gradients)
#     logger.info(f'Pruned layer: {idx, idt}')
#     search_list[int(idx)] = 0

# logger.info('Start comparing')
# wrapper(origin_search_list)
# wrapper([32] * 27)
# wrapper(search_list)

# logger.info('-----Start weight sum test------')

# origin_search_list = [32] * 32
# search_list = deepcopy(origin_search_list)

# for _ in range(prune_turn):
#     logger.info(f'Start searching for {search_list}')
#     args.lora = search_list
#     args.epochs = 1
#     configs = PEFTSearchSpace(args).get_config()
#     model = PEFTModel(configs, dataset).half()
#     res, gradients = model.run()
#     logger.info(f'Result {res} for {search_list}')

#     idx, idt = prune_model(
#         model.model,
#         task_name='my_module',
#         opts=['lora'],
#         p_method='values_below_threshold',
#         top_p=12,
#         print_names=True,
#         gradients=gradients)
#     logger.info(f'Pruned layer: {idx, idt}')
#     search_list[int(idx)] = 0

# logger.info('Start comparing')
# wrapper(origin_search_list)
# wrapper([32] * (32 - prune_turn))
# wrapper(search_list)
