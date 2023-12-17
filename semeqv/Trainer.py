import matplotlib.pyplot as plt
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
    def   set_trainbar(self, value): self.trainbar   = value; return self
    def   set_validbar(self, value): self.validbar   = value; return self
    def    set_testbar(self, value): self.testbar    = value; return self
    def      set_saver(self, value): self.saver      = value; return self
    def    set_compile(self, value): self.compile    = value; return self

    @staticmethod
    @click.command()
    @click.pass_obj
    def train(trainer):

        startepoch = trainer.saver.restore(trainer.model, trainer.optimizer, trainer.trainsplit.dataset, trainer.validsplit.dataset)

        uncompiled = trainer.model
        if trainer.compile:
            torch.set_float32_matmul_precision('high')
            trainer.model = torch.compile(trainer.model)

        dists = []
        for epoch in range(startepoch, trainer.epochs):
            
            trainer.model.train()
            for i,batch in (bar:=tqdm.tqdm(enumerate(trainer.trainsplit,1), total=len(trainer.trainsplit))):
                trainer.optimizer.zero_grad()
                data = trainer.trainsplit.dataset.todevice(**batch)
                pred = trainer.model(**data)
                loss = trainer.lossfn(**(data | pred))
                loss.backward()
                trainer.optimizer.step()
                trainer.trainbar(bar=bar, epoch=epoch, epochs=trainer.epochs, step=i, loss=loss.item(), mode="train", **(data | pred))

            trainer.model.eval()
            for i,batch in (bar:=tqdm.tqdm(enumerate(trainer.validsplit,1), total=len(trainer.validsplit))):
                data = trainer.trainsplit.dataset.todevice(**batch)
                pred = trainer.model(**data)
                loss = trainer.lossfn(**(data | pred))
                trainer.validbar(bar=bar, epoch=epoch, epochs=trainer.epochs, step=i, loss=loss.item(), mode="valid", **(data | pred))

            trainer.saver.save(epoch, trainer.validbar.get_value(), trainer.optimizer, uncompiled, trainer.trainsplit.dataset, trainer.validsplit.dataset)
            trainer.trainbar.end_epoch()
            trainer.validbar.end_epoch()
            
            #####

            dists.append(torch.cdist(trainer.model.embedding.weight,trainer.model.embedding.weight).detach().cpu().numpy())

            if epoch == 500 or epoch % 5000 == 0:
                for i,j in [(i,j) for i in range(trainer.model.embedding.weight.size(0)) for j in range(trainer.model.embedding.weight.size(0)) if i < j]:
                    if   0 <= i < 2 and 0 <= j < 2: plt.plot([e[i,j] for e in dists], color="blue")
                    elif 2 <= i < 4 and 2 <= j < 4: plt.plot([e[i,j] for e in dists], color="purple")
                    elif i == 6 or j == 6: plt.plot([e[i,j] for e in dists], color="green")
                    else: plt.plot([e[i,j] for e in dists], color="black")
                plt.show()
                plt.clf()
