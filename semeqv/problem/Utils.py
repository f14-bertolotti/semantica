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

split2callback = {
    "train"      : lambda trainer : trainer.set_traincallback,
    "validation" : lambda trainer : trainer.set_validcallback,
    "test"       : lambda trainer : trainer.set_testcallback
}
split2dataset = {
    "train"      : lambda trainer : trainer.set_trainsplit,
    "validation" : lambda trainer : trainer.set_validsplit,
    "test"       : lambda trainer : trainer.set_testsplit
}

name2activation = {
            'relu': torch.nn.ReLU(),
            'gelu': torch.nn.GELU(),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh()
        }
