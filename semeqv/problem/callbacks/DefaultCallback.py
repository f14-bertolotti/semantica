from semeqv.problem.callbacks import callbacks
import click

class DefaultCallback:

    def __init__(self, trainer): 
        self.trainer = trainer
        self.acc_sum, self.lss_sum, self.samples = 0,0,0

    def start(self): pass

    def start_epoch(self, epoch):
        self.epoch = epoch

    def start_step(self, step):
        self.step = step

    def end(self): pass

    def end_step(self, loss, data, pred):
        """ called at the end of each model prediction """
        prd, tgt = pred["prd"], data["tgt"]
        self.cur_acc = (tgt[tgt!=-100] == prd[tgt!=-100]).float().sum().item() 
        self.cur_spl = (tgt!=-100).sum().item()
        self.acc_sum = self.acc_sum + self.cur_acc
        self.lss_sum = self.lss_sum + loss * self.cur_spl # reduction method assumed to be mean
        self.samples = self.samples + self.cur_spl

    def end_epoch(self):
        """ called at the end of each model epoch """
        self.acc_sum = 0
        self.lss_sum = 0
        self.samples = 0

    def get_cur_results(self):
        """ returns current accuracy and loss averages """
        return (self.acc_sum / self.samples, self.lss_sum / self.samples)

    def get_step_results(self):
        return self.get_cur_results()

    def get_epoch_results(self):
        return self.get_cur_results()

    def get_epoch_description(self):
        results = self.get_cur_results()
        return f"a:{results[0]:.5f} l:{results[1]:.5f}"

    def get_step_description(self):
        results = self.get_cur_results()
        return f"a:{results[0]:.5f} l:{results[1]:.5f}"


@click.group()
def default(): pass
callbacks.add_command(default)

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def traincallback(trainer):
    trainer.set_traincallback(DefaultCallback(trainer))

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def validcallback(trainer):
    trainer.set_validcallback(DefaultCallback(trainer))

@default.group(invoke_without_command=True, context_settings={'show_default': True})
@click.pass_obj
def testcallback(trainer):
    trainer.set_testcallback(DefaultCallback(trainer))


