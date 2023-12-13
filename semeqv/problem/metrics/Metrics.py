import click

@click.group()
def metrics(): pass

@metrics.group(invoke_without_command=True)
@click.pass_obj
def accuracy(trainer):
    def wrapped(prd, tgt, **kwargs): return {"accuracy" : (prd == tgt).float().mean().item()}
    trainer.set_metric_fn(wrapped)   
