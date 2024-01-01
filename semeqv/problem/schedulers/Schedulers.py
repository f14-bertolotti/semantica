import transformers 
import torch
import click
    
@click.group()
def schedulers(): pass

@schedulers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--tmax" ,  "tmax", type=  int, default=50)
@click.option("--lrmin", "lrmin", type=float, default=0.0001)
@click.pass_obj
def cosine(trainer, tmax, lrmin):
    trainer.set_scheduler(
        torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, tmax, eta_min=lrmin)
    )

@schedulers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def noscheduler(trainer):
    class FakeSched: 
        def step(self): pass
    trainer.set_scheduler(FakeSched())


@schedulers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--warmup"    , "warmup"    , type=int, default=30000)
@click.option("--last_epoch", "last_epoch", type=int)
@click.pass_obj
def constant(trainer, warmup, last_epoch):
    trainer.set_scheduler(
        transformers.get_constant_schedule_with_warmup(trainer.optimizer, num_warmup_steps=warmup, last_epoch=last_epoch)
    )
