import click

@click.group()
def bars(): pass

@bars.group(invoke_without_command=True)
@click.pass_obj
def ESLABar(trainer):
    def wrapped(bar, epoch, step, loss, metrics): 
        accuracy = metrics["accuracy"]
        bar.set_description(f"{epoch} {step} {loss} {accuracy}")
    trainer.set_bar_fn(wrapped)
