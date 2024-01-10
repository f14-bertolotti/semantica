from semeqv.problem.callbacks import callbacks, LoggingCallback
from semeqv.problem import split2callback
import click, torch, numpy

class CdistsCallback(LoggingCallback):

    def __init__(self, trainer, input_embedding_path="", output_embedding_path="", step_log_path="", epoch_log_path="", epochs_to_checkpoint=0): 
        super().__init__(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path)
        self.input_cdists, self.output_cdists = [],[]
        self.input_embedding_path  = input_embedding_path
        self.output_embedding_path = output_embedding_path
        self.epochs_to_checkpoint  = epochs_to_checkpoint

    def end_step(self, loss, data, pred):
        super().end_step(loss, data, pred)
        self. input_cdists.append(torch.cdist(self.trainer.model. input_embedding.weight,self.trainer.model. input_embedding.weight).detach().cpu().numpy())
        self.output_cdists.append(torch.cdist(self.trainer.model.output_embedding.weight,self.trainer.model.output_embedding.weight).detach().cpu().numpy())

    def end_epoch(self):
        super().end_epoch()
        if self. input_embedding_path and self.epoch % self.epochs_to_checkpoint == 0: numpy.save(self. input_embedding_path, numpy.stack(self. input_cdists))
        if self.output_embedding_path and self.epoch % self.epochs_to_checkpoint == 0: numpy.save(self.output_embedding_path, numpy.stack(self.output_cdists))

    def end(self):
        super().end()
        if self. input_embedding_path: numpy.save(self. input_embedding_path, numpy.stack(self. input_cdists))
        if self.output_embedding_path: numpy.save(self.output_embedding_path, numpy.stack(self.output_cdists))

@callbacks.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--input_embedding_path"  , "input_embedding_path"  , type=str , default="")
@click.option("--output_embedding_path" , "output_embedding_path" , type=str , default="")
@click.option( "--step_log_path"        , "step_log_path"         , type=str , default="")
@click.option("--epoch_log_path"        , "epoch_log_path"        , type=str , default="")
@click.option("--etc"                   , "etc"                   , type=int , default=0)
@click.option("--split"                 , "split"                 , type=str , default="train")
@click.pass_obj
def cdists(trainer, input_embedding_path, output_embedding_path, step_log_path, epoch_log_path, etc, split):
    split2callback[split](trainer)(
        CdistsCallback(
            trainer, 
            input_embedding_path  = input_embedding_path,
            output_embedding_path = output_embedding_path,
            step_log_path         = step_log_path,
            epoch_log_path        = epoch_log_path,
            epochs_to_checkpoint  = etc
        )
    )
