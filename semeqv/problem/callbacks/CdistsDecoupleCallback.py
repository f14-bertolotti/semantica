from semeqv.problem.callbacks import callbacks, CdistsCallback
import click, torch, numpy

class CdistsDecoupleCallback(CdistsCallback):

    def __init__(self, trainer, path="", step_log_path="", epoch_log_path="", epochs_to_checkpoint=0, epochs_to_decouple=100000): 
        super().__init__(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, epochs_to_checkpoint=epochs_to_checkpoint)
        self.epochs_to_decouple = epochs_to_decouple

    def end_epoch(self):
        super().end_epoch()
        if self.epoch == self.epochs_to_decouple: 
            print("decouple wt")
            self.trainer.model.decouple_wt()

@click.group()
def cdists_decouple(): pass
callbacks.add_command(cdists_decouple)

@cdists_decouple.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.option("--etd" , "etd" , type=int, default=10000)
@click.pass_obj
def traincallback(trainer, path, step_log_path, epoch_log_path, etc, etd):
    trainer.set_traincallback(CdistsDecoupleCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, epochs_to_checkpoint=etc, epochs_to_decouple=etd))

@cdists_decouple.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.option("--etd" , "etd" , type=int, default=10000)
@click.pass_obj
def validcallback(trainer, path, step_log_path, epoch_log_path, etc, etd):
    trainer.set_validcallback(CdistsDecoupleCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, epochs_to_checkpoint=etc, epochs_to_decouple=etd))

@cdists_decouple.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.option("--etd" , "etd" , type=int, default=10000)
@click.pass_obj
def testcallback(trainer, path, step_log_path, epoch_log_path, etc, etd):
    trainer.set_testcallback(CdistsDecoupleCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, epochs_to_checkpoint=etc, epochs_to_decouple=etd))
