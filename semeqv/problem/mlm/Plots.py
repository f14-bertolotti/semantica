from semeqv.problem.mlm import mlm
from semeqv.problem.mlm.datasets.Default import common_words
import itertools, seaborn, random, pandas, numpy, click
import matplotlib.pyplot as plt

@mlm.command()
@click.pass_obj
@click.option("--palette" , "palette" , type=str          , default="magma")
@click.option("--title"   , "title"   , type=str          , default="")
@click.option("--show"    , "show"    , type=bool         , default=False)
@click.option("--etc"     , "etc"     , type=int          , default=1)
@click.option("--ylim"    , "ylim"    , type=(float,float), default=None)
@click.option("--path"    , "path"    , type=str          , default="")
@click.option("--showconf", "showconf", type=bool         , default=True)
@click.argument("paths", nargs=-1, type=click.Path())
def view(_, paths, etc, ylim, palette, title, show, path, showconf):

    common_idxs, semeqv_idxs = list(range(0,100)), list(range(100,200))
    indexes                  = list(zip(common_idxs, semeqv_idxs, ["✓ distributional hyp."]*len(common_words)))
    combs                    = list(itertools.combinations(common_idxs,2))
    random.shuffle(combs)
    combs = combs[:100]
    indexes                 += list(zip([p[0] for p in combs], [p[1] for p in combs], ["✗ distributional hyp."]*len(combs)))


    dists  = [[numpy.load(path)[::etc,i,j] for path in paths] for i,j,_ in indexes]
    maxlen = max([max(map(len,data)) for data in dists])
    dists  = [[list(trial) + [None] * (maxlen-len(trial)) for trial in data] for data in dists]
    steps  = [value*etc         for trials       in dists            for trial       in trials            for value,_ in enumerate(trial)]
    trials = [value             for trials       in dists            for value,trial in enumerate(trials) for _       in trial]
    types  = [indexes[value][2] for value,trials in enumerate(dists) for trial       in trials            for _       in trial]
    values = [value             for trials       in dists            for trial       in trials            for value   in trial]

    dists = pandas.DataFrame.from_dict({"step":steps, "distance":values, "embedding pairs":types, "trials":trials})
    ax = seaborn.lineplot(data=dists, x="step", y="distance", hue="embedding pairs", palette=seaborn.color_palette(palette, 2)) if showconf else \
         seaborn.lineplot(data=dists, x="step", y="distance", hue="embedding pairs", palette=seaborn.color_palette(palette, 2), units="trials", estimator=None)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ylim: ax.set_ylim(ylim)
    ax.set_title(title)
    plt.tight_layout()
    if show: plt.show()
    if path: ax.figure.savefig(path)
