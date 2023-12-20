from semeqv.problem.optimizers import optimizers
from semeqv.problem.losses import loss
from semeqv.problem.savers import savers
from semeqv.problem.callbacks import callbacks
from semeqv.problem.xor    import xor
from semeqv import train, Trainer
from functools import reduce
import matplotlib.pyplot as plt
import seaborn
import random
import pandas
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


@cli.command()
@click.pass_obj
@click.option("--idxs1", "idxs1", type=(int,int), default=(0,1))
@click.option("--idxs2", "idxs2", type=(int,int), default=(2,3))
@click.argument("paths", nargs=-1, type=click.Path())
def view(_, paths, idxs1, idxs2):
    dists1 = [numpy.load(path)[:,idxs1[0],idxs1[1]] for path in paths]
    dists2 = [numpy.load(path)[:,idxs2[0],idxs2[1]] for path in paths]
    maxlen = max(max(map(len,dists1)),max(map(len,dists2)))
    trials = [[i]*maxlen for i in range(len(dists1))]
    types  = [1]*maxlen*len(dists1) + [2]*maxlen*len(dists2)
    steps  = [list(range(maxlen)) for i in range(len(dists1))]
    dists1 = [list(dists) + [None] * (maxlen-len(dists)) for dists in dists1]
    dists2 = [list(dists) + [None] * (maxlen-len(dists)) for dists in dists2]
    dists1 = reduce(lambda x,y: x + y, dists1) 
    dists2 = reduce(lambda x,y: x + y, dists2) 
    trials = reduce(lambda x,y: x + y, trials)
    steps  = reduce(lambda x,y: x + y,  steps)
    print(len(steps), len(trials), len(dists1), len(dists2))
    dists = pandas.DataFrame.from_dict({"steps":steps+steps, "trials":trials+trials, "value":dists1 + dists2, "types":types})
    print(dists)
    seaborn.lineplot(data=dists, x="steps", y="value", hue="types")
    plt.show()

        #####
        #dists.append(torch.cdist(trainer.model.embedding.weight,trainer.model.embedding.weight).detach().cpu().numpy())

        #if epoch == 500 or epoch % 1000 == 0:
        #    for i,j in [(i,j) for i in range(trainer.model.embedding.weight.size(0)) for j in range(trainer.model.embedding.weight.size(0)) if i < j]:
        #        if   0 <= i < 2 and 0 <= j < 2: plt.plot([e[i,j] for e in dists], color="blue")
        #        elif 2 <= i < 4 and 2 <= j < 4: plt.plot([e[i,j] for e in dists], color="purple")
        #        elif i == 6 or j == 6: plt.plot([e[i,j] for e in dists], color="green")
        #        else: plt.plot([e[i,j] for e in dists], color="black")
        #    plt.show()
        #    plt.clf()



cli.add_command(savers)
cli.add_command(loss)
cli.add_command(callbacks)
cli.add_command(xor)
cli.add_command(optimizers)
cli.add_command(train)
cli.add_command(view)

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

