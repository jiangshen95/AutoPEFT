'''
用于运行nni trial，每次按照config跑一次
'''
import nni
from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
import torch
from pruning_methods import prune_model
from utils.gpu_memory_plot import get_free_gpu_memory
from src.controllers.baseline_wrapper import baseline_wrapper_single, prune_wrapper_single, baseline_wrapper_double, prune_wrapper_double

import yaml
import argparse
import logging
import time
from copy import deepcopy
import os
import json
import random
import numpy as np

logger = logging.getLogger('controller')
logger.setLevel(logging.INFO)  # 设置日志级别
time_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
day_str = time.strftime('%Y-%m-%d', time.localtime())
output_dir = f'../nni_outputs/{day_str}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_handler = logging.FileHandler(
    f'../nni_outputs/{day_str}/output_{time_str}.log', mode='w')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--result', type=str)


def comma_separated_strings(value):
    return value.split(',')


parser.add_argument('--device', type=comma_separated_strings)
args = parser.parse_args(args=[])

args.method = '../method_configs/lora_after_prune.yaml'
args.task = '../task_configs/qnli.yaml'
args.result = 'lora_after_prune_qnli.json'
args.device = [0]

with open(args.method, 'r') as file:
    method_configs = yaml.safe_load(file)
with open(args.task, 'r') as file:
    task_configs = yaml.safe_load(file)

logger.info(
    f'Start exp for {args.task}:{task_configs}\n{args.method}:{method_configs}')

print(method_configs)
print(task_configs)


def reset_seed():
    # 设置Python的随机种子
    random.seed(42)

    # 设置NumPy的随机种子
    np.random.seed(42)

    # 设置PyTorch的随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # 为所有GPU设置种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(arg):
    if 'LORA' in method_configs:
        peft_type = 'LORA'
    else:
        peft_type = 'ADAPTER'
    print(peft_type)

    ds_meta = task_configs['DATASETS'][0]
    dataset_name = ds_meta['DATASET_NAME']
    task_name = ds_meta['TASK_NAME']
    configs = deepcopy(method_configs)
    configs['LOSS'] = ds_meta['LOSS']
    configs['LORA_LR'] = arg['lr']

    dataset = PEFTDataset(
        dataset_name,
        task_name,
        train_size=task_configs['TRAIN_SIZE'],
        test_size=task_configs['TEST_SIZE']).get_dataset()

    reset_seed()
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run(args.device)
    logger.info(f'Final-Result {res} for {configs[peft_type]}')
    for inter_res in res['intermediate_results']:
        nni.report_intermediate_result(inter_res['eval_accuracy'])
    nni.report_final_result(res['eval_accuracy'])


if __name__ == '__main__':
    try:
        # get parameters form tuner
        params = nni.get_next_parameter()
        logger.debug(params)
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
