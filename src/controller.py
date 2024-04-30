from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
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
dataset = PEFTDataset('rotten_tomatoes', test_size=0.5).get_dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--lora', type=int, nargs='+')
parser.add_argument('--adapter', type=int, nargs='+')
parser.add_argument('--base_lora', type=int)
parser.add_argument('--base_adapter', type=int)
args = parser.parse_args()

searched_args = []
searched_res = []

# baseline
search_list = [32] * 12
logger.info(f'Start baseline for {search_list}')
args.lora = search_list
configs = PEFTSearchSpace(args).get_config()
model = PEFTModel(configs, dataset)
res = model.run()
logger.info(f'Result for {search_list}: {res}')

for _ in range(10):
    search_list = [0] * 12
    options = [16, 32, 64]
    max_total = 32 * 12
    current_total = 0

    random.seed(int(time.time()))
    for i in range(len(search_list)):
        value = random.choice(options)
        if current_total + value <= max_total:
            search_list[i] = value
            current_total += value
        else:
            value = max_total - current_total
            search_list[i] = value
            break
    if current_total < max_total:
        search_list[random.randint(0,
                                   len(search_list) -
                                   1)] += max_total - current_total

    logger.info(f'Start searching for {search_list}')
    args.lora = search_list
    configs = PEFTSearchSpace(args).get_config()
    model = PEFTModel(configs, dataset)
    res = model.run()
    logger.info(f'Result for {search_list}: {res}')

    searched_args.append(search_list)
    searched_res.append(res)
