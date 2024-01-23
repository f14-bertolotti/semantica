from semeqv.problem.optimizers import optimizers
from semeqv.problem.losses     import loss
from semeqv.problem.savers     import savers
from semeqv.problem.callbacks  import callbacks
from semeqv.problem.schedulers import schedulers
from semeqv.problem.xor        import xor
from semeqv.problem.mlm        import mlm
from semeqv import test, train, Trainer
import matplotlib.pyplot as plt
import matplotlib
import jsonlines
import seaborn
import random
import pandas
import pickle
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
@click.option("--seed"           , "seed"           , type=int  , default=42)
@click.option("--compile"        , "compile"        , type=bool , default=True)
@click.option("--detect_anomaly" , "detect_anomaly" , type=bool , default=False)
@click.option("--etv"            , "etv"            , type=int  , default=1)
@click.option("--epochs"         , "epochs"         , type=int  , default=1)
@click.option("--trainbar"       , "trainbar"       , type=bool , default=True)
@click.option("--validbar"       , "validbar"       , type=bool , default=True)
@click.option("--testbar"        , "testbar"        , type=bool , default=True)
@click.option("--epochbar"       , "epochbar"       , type=bool , default=False)
@click.pass_context
def cli(context, compile, detect_anomaly, etv, epochs, seed, trainbar, validbar, testbar, epochbar):
    torch.autograd.set_detect_anomaly(detect_anomaly)
    if not context.obj: 
        seed_everything(seed)
        context.obj = trainer \
                .set_epochs(epochs) \
                .set_compile(compile) \
                .set_epochs_to_valid(etv) \
                .set_epochbar(epochbar) \
                .set_trainbar(trainbar) \
                .set_validbar(validbar) \
                .set_testbar(testbar)


@cli.command()
@click.pass_obj
@click.option("--indexes"    , "indexes"    , type=(int,int,str), default=[(0,1,"01"),(2,3,"23")], multiple=True)
@click.option("--palette"    , "palette"    , type=str          , default="magma")
@click.option("--title"      , "title"      , type=str          , default="")
@click.option("--show"       , "show"       , type=bool         , default=False)
@click.option("--etc"        , "etc"        , type=int          , default=1)
@click.option("--ylim"       , "ylim"       , type=(float,float), default=None)
@click.option("--path"       , "path"       , type=str          , default="")
@click.option("--showconf"   , "showconf"   , type=bool         , default=True)
@click.option("--transparent", "transparent", type=bool         , default=True)
@click.option("--color"      , "color"      , type=str          , default="black")
@click.argument("paths", nargs=-1, type=click.Path())
def view(_, paths, etc, ylim, indexes, palette, title, show, path, showconf, transparent, color):

    matplotlib.rcParams['text.color']      = color
    matplotlib.rcParams['axes.labelcolor'] = color
    matplotlib.rcParams['xtick.color']     = color
    matplotlib.rcParams['ytick.color']     = color

    dists  = [[numpy.load(path)[::etc,i,j] for path in paths] for i,j,_ in indexes]
    maxlen = max([max(map(len,data)) for data in dists])
    dists  = [[list(trial) + [None] * (maxlen-len(trial)) for trial in data] for data in dists]
    steps  = [value*etc         for trials       in dists            for trial       in trials            for value,_ in enumerate(trial)]
    trials = [value             for trials       in dists            for value,trial in enumerate(trials) for _       in trial]
    types  = [indexes[value][2] for value,trials in enumerate(dists) for trial       in trials            for _       in trial]
    values = [value             for trials       in dists            for trial       in trials            for value   in trial]

    dists = pandas.DataFrame.from_dict({"step":steps, "distance":values, "embedding pairs":types, "trials":trials})
    ax = seaborn.lineplot(data=dists, x="step", y="distance", hue="embedding pairs", palette=seaborn.color_palette(palette, len(indexes))) if showconf else \
         seaborn.lineplot(data=dists, x="step", y="distance", hue="embedding pairs", palette=seaborn.color_palette(palette, len(indexes)), units="trials", estimator=None)

    ax.get_legend().get_frame().set_facecolor('none')
    ax.spines['bottom'].set_color(color)
    ax.spines['left' ].set_color(color)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim: ax.set_ylim(ylim)
    ax.set_title(title)
    plt.tight_layout()
    if show: plt.show()
    if path: ax.figure.savefig(path, transparent=transparent)

@cli.command()
@click.option("--path"    , "paths"   , type=str      , default=[], multiple=True)
@click.option("--index"   , "indexes" , type=(int,int,str), default=[], multiple=True)
@click.option("--palette" , "palette" , type=str      , default="magma")
@click.option("--title"   , "title"   , type=str      , default="")
@click.option("--show"    , "show"    , type=bool     , default=False)
@click.option("--etc"     , "etc"     , type=int      , default=1)
@click.option("--showconf", "showconf", type=bool     , default=True)
def view_embeddings(paths, indexes, etc, palette, title, show, showconf):
    for path in paths:
        cdists = []
        with open(path, "rb") as file:
            while True: 
                try: 
                    embeddings = torch.tensor(pickle.load(file))
                    cdists.append([torch.cdist(embeddings[[i]], embeddings[[j]]).squeeze() for i,j,_ in indexes])
                except EOFError: break
            for i,(_,_,c) in enumerate(indexes):
                plt.plot(range(len(cdists)),[d[i] for d in cdists],color=c)
            plt.show()

@cli.command()
@click.pass_obj
@click.option("--path", type=click.Path())
@click.option("--special","specials",type=(int,int,str),multiple=True)
def viewall(_, specials, path):
    cdists = numpy.load(path)
    for i,j in [(i,j) for i in range(cdists.shape[1]) for j in range(cdists.shape[2]) if i < j]:
        plt.plot(numpy.arange(cdists.shape[0]),cdists[:,i,j], color="black")
        for si,sj,c in specials:
            if i==si and j == sj: plt.plot(numpy.arange(cdists.shape[0]),cdists[:,i,j], color=c)
    plt.show()

@cli.command()
@click.option("--inputs"  ,"inputs"  ,type=(click.Path(), str),default=[] , multiple=True)
@click.option("--window"  ,"window"  ,type=int                ,default=1)
@click.option("--etc"     ,"etc"     ,type=int                ,default=1)
@click.option("--cutoff"  ,"cutoff"  ,type=int                ,default=1)
@click.option("--hline"   ,"hline"   ,type=(float, str, str)  ,default=None)
@click.option("--xlim"    ,"xlim"    ,type=(int, int)         ,default=None)
@click.option("--ylim"    ,"ylim"    ,type=(int, int)         ,default=None)
@click.option("--palette" ,"palette" ,type=str                ,default="magma")
@click.option("--show"    ,"show"    ,type=bool               ,default=False)
@click.option("--title"   ,"title"   ,type=str                ,default="Accuracy")
@click.option("--output"  ,"output"  ,type=click.Path()       ,default="")
@click.option("--showconf","showconf", type=bool              ,default=True)
def accplot(inputs,title,show,window,hline,xlim,ylim,etc,cutoff,palette,showconf,output):
    accuracies = [[obj["message"]["accuracy"] for _,obj in enumerate(jsonlines.open(path))] for _,(path,_) in enumerate(inputs)]
    accuracies = [e for x in accuracies for e in numpy.convolve(numpy.array(x), numpy.ones(window)/window, mode='same')[:(-cutoff if cutoff else None):etc]]
    epochs     = [obj["message"][   "epoch"] for _,(path,_) in enumerate(inputs) for _,obj in list(enumerate(jsonlines.open(path)))[:(-cutoff if cutoff else None):etc]]
    trials     = [i                          for i,(path,_) in enumerate(inputs) for _,  _ in list(enumerate(jsonlines.open(path)))[:(-cutoff if cutoff else None):etc]]
    types      = [t                          for _,(path,t) in enumerate(inputs) for _,  _ in list(enumerate(jsonlines.open(path)))[:(-cutoff if cutoff else None):etc]]
    data       = pandas.DataFrame.from_dict({"accuracy":accuracies, "step":epochs, "trials":trials, "types":types})
    ax = seaborn.lineplot(data=data, x="step", y="accuracy", hue="types",estimator="mean", palette=palette)if showconf else \
         seaborn.lineplot(data=data, x="step", y="accuracy", hue="types", palette=palette, units="trials", estimator=None)
    if hline is not None: ax.axhline(y=hline[0],c=hline[1],linestyle=hline[2])
    if xlim  is not None: ax.set_xlim(*xlim)
    if ylim  is not None: ax.set_ylim(*ylim)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    if show: plt.show()
    if output: ax.figure.savefig(output)

cli.add_command(savers)
cli.add_command(loss)
cli.add_command(callbacks)
cli.add_command(xor)
cli.add_command(mlm)
cli.add_command(optimizers)
cli.add_command(schedulers)
cli.add_command(train)
cli.add_command(test)
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

