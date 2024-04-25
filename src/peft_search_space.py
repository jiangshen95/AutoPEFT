'''A config class that can be used to define the type, location, 
and number of PEFTs
This config input is a args parser. The output is a PEFTSearchSpace object
main.py will only use the PEFTSearchSpace object to generate the model structure
'''

MAX_MODEL_LAYER = 12


class PEFTSearchSpace:
    '''
    PEFTSearchSpace is a class that defines the search space of PEFTs
    '''
    configs = None
    args = None

    def __init__(self, args):
        '''
        args: a argparse.ArgumentParser object
        '''
        self.args = args
        self.configs = {}
        if args.base_lora:
            self.configs['lora'] = {
                'type': 'lora',
                'ranks': [args.base_lora for _ in range(MAX_MODEL_LAYER)]
            }
        if args.base_adapter:
            self.configs['adapter'] = {
                'type': 'adapter',
                'bn': [args.base_adapter for _ in range(MAX_MODEL_LAYER)]
            }
        if args.lora:
            self.configs['lora'] = {
                'type': 'lora',
                'ranks': args.lora,  # a list of ranks
            }
            self.configs['lora']['ranks'] = [
                max(0, rank) for rank in self.configs['lora']['ranks']
            ]
        if args.adapter:
            self.configs['adapter'] = {
                'type': 'adapter',
                'bn': args.adapter,  # a list of bottleneck sizes
            }
            self.configs['adapter']['bn'] = [
                max(0, bn) for bn in self.configs['adapter']['bn']
            ]

    def get_config(self):
        return self.configs
