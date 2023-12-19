from semeqv.problem.callbacks import DefaultCallback
from semeqv.problem.callbacks import CdistsCallback
import click

@click.group()
def callbacks(): pass

@click.group()
def default(): pass
callbacks.add_command(default)

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def traincallback(trainer):
    trainer.set_traincallback(DefaultCallback(trainer))

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def validcallback(trainer):
    trainer.set_validcallback(DefaultCallback(trainer))

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def testcallback(trainer):
    trainer.set_testcallback(DefaultCallback(trainer))

@click.group()
def cdists(): pass
callbacks.add_command(cdists)

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def traincallback(trainer, path, etc):
    trainer.set_traincallback(CdistsCallback(trainer, path=path, epochs_to_checkpoint=etc))

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def validcallback(trainer, path, etc):
    trainer.set_validcallback(CdistsCallback(trainer, path=path, epochs_to_checkpoint=etc))

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def testcallback(trainer, path, etc):
    trainer.set_testcallback(CdistsCallback(trainer, path=path, epochs_to_checkpoint=etc))
