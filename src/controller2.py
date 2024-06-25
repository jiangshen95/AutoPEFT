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
logger.info('Start prune lora')

args.lora = [32] * 32
args.epoch = 3
configs = PEFTSearchSpace(args).get_config()
to_load = ['qnli', 'rte', 'wnli', 'cola', 'sst2', 'mrpc', 'qqp']
for ds_name in to_load:
    logger.info(f"start testing {ds_name}")
    dataset = PEFTDataset(
        'glue', ds_name, train_size=1200, test_size=400).get_dataset()
    baseline_wrapper_single(
        search_list=args.lora,
        ds_name=ds_name,
        dataset=dataset,
        logger=logger,
        configs=configs)
    dataset = None
    model = None
    torch.cuda.empty_cache()
