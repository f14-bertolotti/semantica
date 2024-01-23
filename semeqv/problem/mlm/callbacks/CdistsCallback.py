
from semeqv.problem.mlm.callbacks import callbacks
from semeqv.problem.callbacks import LoggingCallback
from semeqv.problem import split2callback
import click, torch, numpy

class CdistsCallback(LoggingCallback):

    def __init__(self, trainer, input_embedding_path="", output_embedding_path="", step_log_path="", epoch_log_path="", steps_to_checkpoint=0, epochs_to_checkpoint=0): 
        self.indices = list(trainer.trainsplit.dataset.commonidx2semeqvidx.keys()) + list(trainer.trainsplit.dataset.commonidx2semeqvidx.values())
        super().__init__(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path)
        self.input_cdists, self.output_cdists = [],[]
        self.input_embedding_path  = input_embedding_path
        self.output_embedding_path = output_embedding_path
        self.epochs_to_checkpoint  = epochs_to_checkpoint
        self.steps_to_checkpoint   = steps_to_checkpoint

    def end_step(self, loss, data, pred):
        super().end_step(loss, data, pred)
        if (self.step-1) % self.steps_to_checkpoint == 0:
            self. input_cdists.append(torch.cdist(self.trainer.model. input_embedding.weight[self.indices],self.trainer.model. input_embedding.weight[self.indices]).detach().cpu().numpy())
            self.output_cdists.append(torch.cdist(self.trainer.model.output_embedding.weight[self.indices],self.trainer.model.output_embedding.weight[self.indices]).detach().cpu().numpy())

            if self. input_embedding_path: self. save_input_cdists()
            if self.output_embedding_path: self.save_output_cdists()

    def end_epoch(self):
        super().end_epoch()
        if self. input_embedding_path and self.epoch % self.epochs_to_checkpoint == 0:  self.save_input_cdists()
        if self.output_embedding_path and self.epoch % self.epochs_to_checkpoint == 0: self.save_output_cdists()

    def end(self):
        super().end()
        if self. input_embedding_path: numpy.save(self. input_embedding_path, numpy.stack(self. input_cdists))
        if self.output_embedding_path: numpy.save(self.output_embedding_path, numpy.stack(self.output_cdists))

    def save_input_cdists(self):
        numpy.save(self. input_embedding_path, numpy.stack(self. input_cdists))

    def save_output_cdists(self):
        numpy.save(self.output_embedding_path, numpy.stack(self.output_cdists))


@callbacks.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--input_embedding_path"  , "input_embedding_path"  , type=str , default="")
@click.option("--output_embedding_path" , "output_embedding_path" , type=str , default="")
@click.option( "--step_log_path"        , "step_log_path"         , type=str , default="")
@click.option("--epoch_log_path"        , "epoch_log_path"        , type=str , default="")
@click.option("--etc"                   , "etc"                   , type=int , default=0)
@click.option("--stc"                   , "stc"                   , type=int , default=0)
@click.option("--split"                 , "split"                 , type=str , default="train")
@click.pass_obj
def cdists(trainer, input_embedding_path, output_embedding_path, step_log_path, epoch_log_path, stc, etc, split):
    split2callback[split](trainer)(
        CdistsCallback(
            trainer, 
            input_embedding_path  = input_embedding_path,
            output_embedding_path = output_embedding_path,
            step_log_path         = step_log_path,
            epoch_log_path        = epoch_log_path,
            epochs_to_checkpoint  = etc,
            steps_to_checkpoint   = stc
        )
    )
