import termcolor, click, torch, tqdm
from semeqv.problem import grouped

class FakeBar:
    def __init__(self): pass
    def __call__(self, iterator, **kwargs ): 
        self.iterator = iterator
        return self
    def __iter__(self): return self.iterator 
    def set_description(*args, **kwargs): pass

class FakeScaler:
    def scale(self, x): return x
    def step(self, x): x.step()
    def update(self): pass

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
    def test(trainer):
        # starting testing, initializing ###
        trainer.testcallback.start()

        # restore from a checkpoint if necessary ###
        epoch = trainer.saver.restore(
            trainer.model, 
            trainer.optimizer, 
            trainer.trainsplit.dataset, 
            trainer.validsplit.dataset
        )

        if trainer.compile:
            torch.set_float32_matmul_precision('high')
            trainer.model = torch.compile(trainer.model)

        # test split ###
        trainer.model.eval()
        trainer.testcallback.start_epoch(epoch)
        for step,batch in (bar:=trainer.testbar(enumerate(trainer.testsplit,1), total=len(trainer.testsplit), colour="green")):
            trainer.testcallback.start_step(step)
            data = trainer.trainsplit.dataset.todevice(**batch)
            pred = trainer.model(**data)
            loss = trainer.lossfn(**(data | pred))
            trainer.testcallback.end_step(
                loss = loss.item(),
                data = data,
                pred = pred)
            bar.set_description(termcolor.colored(f"e:{epoch} {trainer.testcallback.get_step_description()}", "green"))
        trainer.testcallback.get_epoch_results()
        trainer.testcallback.end_epoch()

        # all epochs done, finishing ... ###
        trainer.testcallback.end()
 

    @staticmethod
    @click.command()
    @click.option("--amp"            , "amp"            , type=bool , default=True)
    @click.option("--mini_steps"     , "mini_steps"     , type=int  , default=1)
    @click.option("--etv"            , "etv"            , type=int  , default=1)
    @click.option("--detect_anomaly" , "detect_anomaly" , type=bool , default=False)
    @click.pass_obj
    def train(trainer, amp, etv, mini_steps, detect_anomaly):
        torch.autograd.set_detect_anomaly(detect_anomaly)

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

        scaler = torch.cuda.amp.GradScaler() if amp else FakeScaler()

        # main training loop ###
        for epoch in trainer.epochbar(iter(lst:=range(epoch, trainer.epochs)), total=len(lst)):
            
            # trainsplit ###
            trainer.model.train()
            trainer.traincallback.start_epoch(epoch)
            for step,group in (bar:=trainer.trainbar(enumerate(grouped(trainer.trainsplit,mini_steps),1), total=len(trainer.trainsplit)//mini_steps+1, colour="white")):
                with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
                    for batch in group:
                        trainer.traincallback.start_step(step)
                        trainer.optimizer.zero_grad()
                        data = trainer.trainsplit.dataset.todevice(**batch)
                        pred = trainer.model(**data)
                        loss = trainer.lossfn(**(data | pred))
                        scaler.scale(loss/len(group)).backward()
                        trainer.traincallback.end_step(
                            loss = loss.item(),
                            data = data,
                            pred = pred)
                        bar.set_description(termcolor.colored(f"e:{epoch}/{trainer.epochs} {trainer.traincallback.get_step_description()}","white"))
                    scaler.step(trainer.optimizer)
                    scaler.update()
                    trainer.scheduler.step()
            result = trainer.traincallback.get_epoch_results()
            trainer.traincallback.end_epoch()

            # validation split ###
            if epoch % etv == 0:
                trainer.model.eval()
                trainer.validcallback.start_epoch(epoch)
                for step,batch in (bar:=trainer.validbar(enumerate(trainer.validsplit,1), total=len(trainer.validsplit), colour="blue")):
                    with torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
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
           
