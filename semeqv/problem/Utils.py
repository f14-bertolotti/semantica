import random
import torch
import click
import ast

class DistributionOption(click.Option):
    def type_cast_value(self, _, value):
        try: value = ast.literal_eval(value)
        except: raise click.BadParameter(value)
        if sum(value) != 1: raise click.BadParameter(f"value does not sum to 1, it sums to {sum(value)}")
        return value

def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.generator = random.Random(seed + worker_id + worker_info.seed)
    return worker_init_fn
