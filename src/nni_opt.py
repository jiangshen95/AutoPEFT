import nni
import torch
from controller.baseline_wrapper import baseline_wrapper_single
from peft_search_space import PEFTSearchSpace
from run_model import PEFTModel
from dataset_wrapper import PEFTDataset
import argparse


def run_trial(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', type=int, nargs='+')
    parser.add_argument('--adapter', type=int, nargs='+')
    parser.add_argument('--base_lora', type=int)
    parser.add_argument('--base_adapter', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args(args=[])
    args.lora = [params['lora']] * 24
    args.lr = params['lr']
    args.epochs = 3
    ds_name = "qnli"
    dataset = PEFTDataset('glue', ds_name, False, 1200, 400).get_dataset()
    configs = PEFTSearchSpace(args).get_config()
    if ds_name == 'stsb':
        configs['loss'] = 'mse'
    torch.cuda.empty_cache()
    model = PEFTModel(configs, dataset).half()
    res, _, _ = model.run()
    model = None
    nni.report_final_result(res['eval_accuracy'])


if __name__ == '__main__':

    params = nni.get_next_parameter()
    run_trial(params)
