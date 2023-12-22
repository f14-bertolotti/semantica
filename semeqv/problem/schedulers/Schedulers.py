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


#torch_lr_scheduler = ExponentialLR(optimizer=default_optimizer, gamma=0.98)
#
#default_trainer = get_default_trainer()
#
#scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
#                                            warmup_start_value=0.0,
#                                            warmup_end_value=0.1,
#                                            warmup_duration=3)
