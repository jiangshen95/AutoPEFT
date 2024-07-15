"""A config class that can be used to define the type, location,
and number of PEFTs
This config input is a args parser. The output is a PEFTSearchSpace object
main.py will only use the PEFTSearchSpace object to generate the model structure
"""

MAX_MODEL_LAYER = 24


class PEFTSearchSpace:
    """
    PEFTSearchSpace is a class that defines the search space of PEFTs

    Args:
        config (dict): configurations of search space.
        args (ArgumentParser): input arguments.
    """

    def __init__(self, args):
        """
        Args:
            args: a argparse.ArgumentParser object
        """
        self.args = args
        self.configs = {}
        # NOTE: 这里尽可能不要直接传 ArgumentParser 进来，还是把这些参数拆一拆，明确给出来
        if hasattr(args, "base_lora") and args.base_lora:
            self.configs["lora"] = {
                "type": "lora",
                "ranks": [args.base_lora for _ in range(MAX_MODEL_LAYER)],
            }
        if hasattr(args, "base_adapter") and args.base_adapter:
            self.configs["adapter"] = {
                "type": "adapter",
                "bn": [args.base_adapter for _ in range(MAX_MODEL_LAYER)],
            }
        if hasattr(args, "lora") and args.lora:
            self.configs["lora"] = {
                "type": "lora",
                "ranks": args.lora,  # a list of ranks
            }
            self.configs["lora"]["ranks"] = [
                max(0, rank) for rank in self.configs["lora"]["ranks"]
            ]
            num_zeros = MAX_MODEL_LAYER - len(self.configs["lora"]["ranks"])
            if num_zeros > 0:
                self.configs["lora"]["ranks"].extend([0] * num_zeros)
        if hasattr(args, "adapter") and args.adapter:
            self.configs["adapter"] = {
                "type": "adapter",
                "bn": args.adapter,  # a list of bottleneck sizes
            }
            self.configs["adapter"]["bn"] = [
                max(0, bn) for bn in self.configs["adapter"]["bn"]
            ]
            num_zeros = MAX_MODEL_LAYER - len(self.configs["adapter"]["bn"])
            if num_zeros > 0:
                self.configs["adapter"]["bn"].extend([0] * num_zeros)
        if hasattr(args, "epochs") and args.epochs:
            self.configs["epochs"] = args.epochs
        if hasattr(args, "lr") and args.lr:
            self.configs["lr"] = args.lr
        if hasattr(args, "instructs"):
            self.configs["instructs"] = 1  # need instruct for llama

    def get_config(self):
        return self.configs
