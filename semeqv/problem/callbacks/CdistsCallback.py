from semeqv.problem.callbacks import callbacks, LoggingCallback
import click, torch, numpy

class CdistsCallback(LoggingCallback):

    def __init__(self, trainer, path="", step_log_path="", epoch_log_path="", epochs_to_checkpoint=0): 
        super().__init__(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path)
        self.cdists = []
        self.path = path
        self.epochs_to_checkpoint = epochs_to_checkpoint

    def end_step(self, loss, data, pred):
        super().end_step(loss, data, pred)
        self.cdists.append(torch.cdist(self.trainer.model.embedding.weight,self.trainer.model.embedding.weight).detach().cpu().numpy())

    def end_epoch(self):
        super().end_epoch()
        if self.path and self.epoch % self.epochs_to_checkpoint == 0:
            numpy.save(self.path, numpy.stack(self.cdists))

    def end(self):
        super().end()
        if self.path:
            numpy.save(self.path, numpy.stack(self.cdists))

@click.group()
def cdists(): pass
callbacks.add_command(cdists)

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def traincallback(trainer, path, step_log_path, epoch_log_path, etc):
    trainer.set_traincallback(CdistsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, epochs_to_checkpoint=etc))

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def validcallback(trainer, path, step_log_path, epoch_log_path,  etc):
    trainer.set_validcallback(CdistsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path,  epochs_to_checkpoint=etc))

@cdists.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--etc" , "etc" , type=int, default=0)
@click.pass_obj
def testcallback(trainer, path, step_log_path, epoch_log_path,  etc):
    trainer.set_testcallback(CdistsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path,  epochs_to_checkpoint=etc))
