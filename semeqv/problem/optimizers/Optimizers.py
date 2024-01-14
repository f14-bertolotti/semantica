import torch
import click

@click.group()
def optimizers(): pass

@optimizers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--learning_rate", "learning_rate", type=float, default=.0001)
@click.option("--weight_decay", "weight_decay", type=float, default=0)
@click.pass_obj
def adam(trainer,learning_rate,weight_decay):
    trainer.set_optimizer(torch.optim.Adam(trainer.model.parameters(), lr=learning_rate, weight_decay=weight_decay))

@optimizers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--learning_rate", "learning_rate", type=float, default=.0001)
@click.option("--weight_decay", "weight_decay", type=float, default=.001)
@click.option("--eps", "eps", type=float, default=1e-12)
@click.option("--betas", "betas", type=(float,float), default=(.9,.999))
@click.pass_obj
def adamW(trainer, learning_rate, betas, eps, weight_decay):
    trainer.set_optimizer(torch.optim.AdamW(trainer.model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=eps))
