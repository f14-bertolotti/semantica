from semeqv.problem.optimizers import optimizers
from semeqv.problem.losses import loss
from semeqv.problem.savers import savers
from semeqv.problem.callbacks import callbacks
from semeqv.problem.xor    import xor
from semeqv import train, Trainer
import random
import click
import numpy
import torch
import os

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

trainer = Trainer()
@click.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--compile" , "compile"  , type=bool, default=True)
@click.option("--epochs"  , "epochs"   , type=int , default=1)
@click.option("--seed"    , "seed"     , type=int , default=42)
@click.option("--trainbar", "trainbar" , type=bool, default=True)
@click.option("--validbar", "validbar" , type=bool, default=True)
@click.option("--testbar" , "testbar"  , type=bool, default=True)
@click.option("--epochbar", "epochbar" , type=bool, default=False)
@click.pass_context
def cli(context, compile, epochs, seed, trainbar, validbar, testbar, epochbar):
    if not context.obj: 
        context.obj = trainer \
                .set_epochs(epochs) \
                .set_compile(compile) \
                .set_epochbar(epochbar) \
                .set_trainbar(trainbar) \
                .set_validbar(validbar) \
                .set_testbar(testbar)
        seed_everything(seed)

cli.add_command(savers)
cli.add_command(loss)
cli.add_command(callbacks)
cli.add_command(xor)
cli.add_command(optimizers)
cli.add_command(train)

# put the cli group command as last command in the command tree
# so that commands can be chained 
def visit(command):
    if isinstance(command, click.core.Group) and not command.commands: return [command]
    elif isinstance(command, click.core.Group) and command.commands: return [c for cmd in command.commands.values() for c in visit(cmd)]
    else: return []
for grp in visit(cli): grp.add_command(cli)

cli.add_command(cli)

if __name__ == "__main__":
    cli()

