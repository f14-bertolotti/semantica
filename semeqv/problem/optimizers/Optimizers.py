import torch
import click

@click.group()
def optimizers(): pass

@optimizers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--learning_rate", "learning_rate", type=float, default=.0001)
@click.pass_obj
def adam(trainer,learning_rate):
    trainer.set_optimizer(torch.optim.Adam(trainer.model.parameters(), lr=learning_rate))
