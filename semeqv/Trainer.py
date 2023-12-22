import termcolor, click, torch, tqdm

class FakeBar:
    def __init__(self): pass
    def __call__(self, iterator, **kwargs ):
        return iterator
    def set_description(*args, **kwargs): pass

class Trainer:

    def __init__(self): pass 
    def set_trainsplit(self, value): self.trainsplit = value; return self
    def set_validsplit(self, value): self.validsplit = value; return self
    def  set_testsplit(self, value): self.testsplit  = value; return self

    def set_epochbar(self, value): self.epochbar = tqdm.tqdm if value else FakeBar(); return self
    def set_trainbar(self, value): self.trainbar = tqdm.tqdm if value else FakeBar(); return self
    def set_validbar(self, value): self.validbar = tqdm.tqdm if value else FakeBar(); return self
    def  set_testbar(self, value): self.testbar  = tqdm.tqdm if value else FakeBar(); return self
 
    def         set_model(self, value): self.model         = value; return self
    def       set_loss_fn(self, value): self.lossfn        = value; return self
    def     set_optimizer(self, value): self.optimizer     = value; return self
    def     set_scheduler(self, value): self.scheduler     = value; return self
    def        set_epochs(self, value): self.epochs        = value; return self
    def set_traincallback(self, value): self.traincallback = value; return self
    def set_validcallback(self, value): self.validcallback = value; return self
    def  set_testcallback(self, value): self.testcallback  = value; return self
    def         set_saver(self, value): self.saver         = value; return self
    def       set_compile(self, value): self.compile       = value; return self

    @staticmethod
    @click.command()
    @click.pass_obj
    def train(trainer):

        # starting training, initializing ###
        trainer.traincallback.start()
        trainer.validcallback.start()

        # restore from a checkpoint if necessary ###
        epoch = trainer.saver.restore(
            trainer.model, 
            trainer.optimizer, 
            trainer.trainsplit.dataset, 
            trainer.validsplit.dataset
        )

        # compile model if trainer.model is set to true ###
        # always keep the "uncompiled" version for checkpointing purposes
        uncompiled = trainer.model
        if trainer.compile:
            torch.set_float32_matmul_precision('high')
            trainer.model = torch.compile(trainer.model)

        # main training loop ###
        for epoch in trainer.epochbar(range(epoch, trainer.epochs)):
            
            # trainsplit ###
            trainer.model.train()
            trainer.traincallback.start_epoch(epoch)
            for step,batch in (bar:=trainer.trainbar(enumerate(trainer.trainsplit,1), total=len(trainer.trainsplit), colour="white")):
                trainer.traincallback.start_step(step)
                trainer.optimizer.zero_grad()
                data = trainer.trainsplit.dataset.todevice(**batch)
                pred = trainer.model(**data)
                loss = trainer.lossfn(**(data | pred))
                loss.backward()
                trainer.optimizer.step()
                trainer.traincallback.end_step(
                    loss = loss.item(),
                    data = data,
                    pred = pred)
                bar.set_description(termcolor.colored(f"e:{epoch}/{trainer.epochs} {trainer.traincallback.get_step_description()}","white"))
            result = trainer.traincallback.get_epoch_results()
            trainer.traincallback.end_epoch()

            # validation split ###
            trainer.model.eval()
            trainer.validcallback.start_epoch(epoch)
            for step,batch in (bar:=trainer.validbar(enumerate(trainer.validsplit,1), total=len(trainer.validsplit), colour="blue")):
                trainer.validcallback.start_step(step)
                data = trainer.trainsplit.dataset.todevice(**batch)
                pred = trainer.model(**data)
                loss = trainer.lossfn(**(data | pred))
                trainer.validcallback.end_step(
                    loss = loss.item(),
                    data = data,
                    pred = pred)
                bar.set_description(termcolor.colored(f"e:{epoch}/{trainer.epochs} {trainer.validcallback.get_step_description()}", "blue"))
            result = trainer.validcallback.get_epoch_results()
            trainer.validcallback.end_epoch()

            # do a scheduler step
            trainer.scheduler.step()

            # one epoch done, saving if necessary ###
            trainer.saver.save(
                epoch, 
                result, 
                trainer.optimizer, 
                uncompiled, 
                trainer.trainsplit.dataset, 
                trainer.validsplit.dataset
            )

        # all epochs done, finishing ... ###
        trainer.traincallback.end()
        trainer.validcallback.end()
        
        # last saving
        trainer.saver.savelast = True
        trainer.saver.save(
            epoch, 
            result, 
            trainer.optimizer, 
            uncompiled, 
            trainer.trainsplit.dataset, 
            trainer.validsplit.dataset
        )
           
