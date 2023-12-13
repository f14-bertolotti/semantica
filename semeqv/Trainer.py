import tqdm, click, torch

class Trainer:

    def __init__(self): pass 
    def set_trainsplit(self, value): self.trainsplit = value; return self
    def set_validsplit(self, value): self.validsplit = value; return self
    def  set_testsplit(self, value): self.testsplit  = value; return self
    def      set_model(self, value): self.model      = value; return self
    def    set_loss_fn(self, value): self.lossfn     = value; return self
    def  set_optimizer(self, value): self.optimizer  = value; return self
    def     set_epochs(self, value): self.epochs     = value; return self
    def  set_metric_fn(self, value): self.metricfn   = value; return self
    def     set_bar_fn(self, value): self.barfn      = value; return self

    @staticmethod
    @click.command()
    @click.pass_obj
    def train(trainer):

        for epoch in range(trainer.epochs):
            for i,batch in (bar:=tqdm.tqdm(enumerate(trainer.trainsplit), total=len(trainer.trainsplit))):
                trainer.optimizer.zero_grad()
                pred = trainer.model(**batch)
                loss = trainer.lossfn(**(batch | pred))
                metr = trainer.metricfn(**(batch | pred))
                loss.backward()
                trainer.optimizer.step()
                trainer.barfn(bar=bar, epoch=epoch, step=i, loss=loss.item(), metrics=metr)


