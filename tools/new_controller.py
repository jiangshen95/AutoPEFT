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
parser.add_argument('--method', type=str)
parser.add_argument('--task', type=str)
args = parser.parse_args()

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


if 'LORA' in method_configs:
    peft_type = 'LORA'
else:
    peft_type = 'ADAPTER'
print(peft_type)

for ds_meta in task_configs['DATASETS']:
    dataset_name = ds_meta['DATASET_NAME']
    task_name = ds_meta['TASK_NAME']
    configs = deepcopy(method_configs)
    configs['LOSS'] = ds_meta['LOSS']

    dataset = PEFTDataset(
        dataset_name, task_name, train_size=2000, test_size=400).get_dataset()

    reset_seed()
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run()
    logger.info(f'Final-Result {res} for {configs[peft_type]}')
    model = None
    torch.cuda.empty_cache()

    res_methods = {}
    res_methods['base'] = res

    for prune_method in task_configs['PRUNE_METHODS']:
        logger.info(f'Prune method {prune_method}')
        configs = deepcopy(method_configs)
        configs['LOSS'] = ds_meta['LOSS']
        origin_epochs = configs['EPOCHS']
        configs['EPOCHS'] = configs['PRUNE_EPOCHS']
        for _ in range(int(configs['PRUNE_TURN'])):
            reset_seed()
            model = None
            gradients = None
            activations = None
            torch.cuda.empty_cache()

            logger.info(f'Start searching for {peft_type}:{configs[peft_type]}')

            model = PEFTModel(configs, dataset).half()
            res, gradients, activations = model.run()
            logger.info(
                f'Mid-Result {res} for {peft_type} {configs[peft_type]}')

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
            configs[idt.upper()][int(idx)] = 0

            model = None
            gradients = None
            activations = None
            torch.cuda.empty_cache()

        configs['EPOCHS'] = origin_epochs
        model = PEFTModel(configs, dataset).half()
        res, _, _ = model.run()
        logger.info(f'Final-Result {res} for {configs[peft_type]}')
        res_methods[prune_method] = res

    with open(
            f'results/{os.path.splitext(os.path.basename(args.method))[0]}_{os.path.splitext(os.path.basename(args.task))[0]}.json',
            'a') as file:
        file.write(dataset_name + '_' + task_name + '\n')
        file.write(json.dumps(res_methods) + '\n')
