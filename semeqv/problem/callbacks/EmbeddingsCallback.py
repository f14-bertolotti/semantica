from semeqv.problem.callbacks import callbacks, LoggingCallback
import click, pickle

class EmbeddingsCallback(LoggingCallback):

    def __init__(self, trainer, path="", step_log_path="", epoch_log_path="", steps_to_checkpoint=0, buffsize=0): 
        super().__init__(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path)
        self.path = path
        self.steps_to_checkpoint = steps_to_checkpoint
        self.file = open(path, "ab", buffering=buffsize) if path else None

    def end_step(self, loss, data, pred):
        super().end_step(loss, data, pred)
        if self.file != None and (self.step % self.steps_to_checkpoint  == 0 or self.step == 1):
            pickle.dump(self.trainer.model.embedding.weight.detach().cpu().numpy(), self.file)

    def end(self):
        super().end()
        if self.file: self.file.close()

@click.group()
def embedding(): pass
callbacks.add_command(embedding)

@embedding.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--stc" , "stc" , type=int, default=0)
@click.option("--buffsize" , "buffsize" , type=int, default=0)
@click.pass_obj
def traincallback(trainer, path, step_log_path, epoch_log_path, stc, buffsize):
    trainer.set_traincallback(EmbeddingsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path, steps_to_checkpoint=stc, buffsize=buffsize))

@embedding.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--stc" , "stc" , type=int, default=0)
@click.option("--buffsize" , "buffsize" , type=int, default=0)
@click.pass_obj
def validcallback(trainer, path, step_log_path, epoch_log_path, stc, buffsize):
    trainer.set_validcallback(EmbeddingsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path,  steps_to_checkpoint=stc, buffsize=buffsize))

@embedding.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--path", "path", type=str, default="")
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.option("--stc" , "stc" , type=int, default=0)
@click.option("--buffsize" , "buffsize" , type=int, default=0)
@click.pass_obj
def testcallback(trainer, path, step_log_path, epoch_log_path, stc, buffsize):
    trainer.set_testcallback(EmbeddingsCallback(trainer, path=path, step_log_path=step_log_path, epoch_log_path=epoch_log_path,  steps_to_checkpoint=stc, buffsize=buffsize))
