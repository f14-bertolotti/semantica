from semeqv.problem.callbacks import DefaultCallback
import torch, numpy

class CdistsCallback(DefaultCallback):

    def __init__(self, trainer, path="", epochs_to_checkpoint=0): 
        super().__init__(trainer)
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

    def edn(self):
        super().end()
        if self.path:
            numpy.save(self.path, numpy.stack(self.cdists))
