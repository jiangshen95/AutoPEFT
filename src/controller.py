from src.run_model import PEFTModel
from src.peft_search_space import PEFTSearchSpace
from src.dataset_wrapper import PEFTDataset
import argparse

dataset = PEFTDataset('rotten_tomatoes', test_size=0.5).get_dataset()

parser = argparse.ArgumentParser()
parser.add_argument('--lora', type=int, nargs='+')
parser.add_argument('--adapter', type=int, nargs='+')
parser.add_argument('--base_lora', type=int)
parser.add_argument('--base_adapter', type=int)
args = parser.parse_args()

args.base_lora = 64
print(args.base_lora)
configs = PEFTSearchSpace(args).get_config()
model = PEFTModel(configs, dataset)
model.run()

args.base_lora = 32
configs = PEFTSearchSpace(args).get_config()
model = PEFTModel(configs, dataset)
model.run()

args.base_lora = 16
configs = PEFTSearchSpace(args).get_config()
model = PEFTModel(configs, dataset)
model.run()
