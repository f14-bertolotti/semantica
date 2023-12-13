import torch
import click

@click.group()
def loss(): pass

@loss.group(invoke_without_command=True)
@click.pass_obj
def cross_entropy(trainer):
    def wrapped(prd, tgt, **kwargs): 
        return torch.nn.functional.cross_entropy(prd, tgt)
    trainer.set_loss(wrapped)
