import torch
import click
import os

class DefaultSaver:

    name2best = {"max" : (lambda x,y: x>y, (0,)), "min" : (lambda x,y: x<y, (float("inf"),))}

    def __init__(self, restorepath="", map_location="cpu", dirpath="", epochs_to_checkpoint=0, bestfn="max", savelast=False, savebest=False, verbose=False): 
        self.epochs_to_checkpoint = epochs_to_checkpoint
        self.map_location         = map_location
        self.restorepath          = restorepath
        self.dirpath              = dirpath
        self.lastpath             = os.path.join(self.dirpath, "modellast.pt")
        self.bestpath             = os.path.join(self.dirpath, "modelbest.pt")
        self.savelast             = savelast
        self.savebest             = savebest
        self.bestfn,self.bestvl   = DefaultSaver.name2best[bestfn]
        self.verbose              = verbose
    
    def save(self, epoch, value, optimizer, model, trainset, validset):
        savedict = {
            "model"   : model.save(),
            "optimsd" : optimizer.state_dict(),
            "epoch"   : epoch,
            "value"   : value,
            "bestv"   : value if self.bestfn(value,self.bestvl) else self.bestvl,
            "trainset": trainset.save(),
            "validset": validset.save(),
            "cpurng"  : torch.get_rng_state(),
            "cudarng" : torch.cuda.get_rng_state()}
        if self.savelast: torch.save(savedict, self.lastpath)

        if self.epochs_to_checkpoint and epoch % self.epochs_to_checkpoint == 0:
            torch.save(savedict, os.path.join(self.dirpath, f"model{epoch}.pt"))

        if self.savebest and self.bestfn(value, self.bestvl): 
            if self.verbose: print("saved new best model")
            torch.save(savedict, self.bestpath)
            self.bestvl = value

    def restore(self, model, optimizer, trainset, validset): 
        if self.restorepath == "": return 0
        checkpoint = torch.load(self.restorepath, map_location=self.map_location)
        optimizer.load_state_dict(checkpoint["optimsd"])
        torch     .set_rng_state( checkpoint['cpurng'].cpu())
        torch.cuda.set_rng_state(checkpoint['cudarng'].cpu())
        model   .restore(checkpoint["model"])
        trainset.restore(checkpoint["trainset"])
        validset.restore(checkpoint["validset"])
        epoch = checkpoint["epoch"]
        value = checkpoint["value"]
        self.bestvl = checkpoint["bestv"]
        print(f"restored model of epoch {epoch} and value {value} with previous best {self.bestvl}")
        return checkpoint["epoch"] + 1

    
@click.group()
def savers(): pass

@savers.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--dirpath"     ,      "dirpath", type=str , default="./")
@click.option("--restorepath" ,  "restorepath", type=str , default="")
@click.option("--savelast"    ,     "savelast", type=bool, default=False)
@click.option("--savebest"    ,     "savebest", type=bool, default=False)
@click.option("--map_location", "map_location", type=str , default="cpu")
@click.option("--etc"         ,          "etc", type=int , default=0)
@click.option("--mode"        ,         "mode", type=str , default="max")
@click.option("--verbose"     ,      "verbose", type=bool, default=False)
@click.pass_obj
def default_saver(trainer, dirpath, restorepath, savelast, savebest, map_location, etc, mode, verbose):
    trainer.set_saver(
        DefaultSaver(
            dirpath              = dirpath,
            restorepath          = restorepath,
            map_location         = map_location,
            epochs_to_checkpoint = etc,
            bestfn               = mode,
            savelast             = savelast,
            savebest             = savebest,
            verbose              = verbose,
        )
    )

