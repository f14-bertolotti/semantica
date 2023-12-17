from termcolor import colored
import click

class ESLABar:

    def __init__(self, color="white"): 
        self.acc_sum, self.lss_sum, self.samples = 0,0,0
        self.color = color

    def __call__(self, bar, epoch, epochs, step, loss, tgt, prd, mode, **kwargs):
        acc = (tgt[tgt!=-100] == prd[tgt!=-100]).float().sum()
        self.acc_sum = self.acc_sum + acc
        self.lss_sum = self.lss_sum + loss
        self.samples = self.samples + (tgt!=-100).sum()
        bar.set_description(colored(f"{mode}: {epoch+1}/{epochs} {step} {self.lss_sum/self.samples:1.5f} {self.acc_sum/self.samples:1.5f}",self.color))

    def get_value(self):
        """ only useful for the saver, it the metrics wrt. the model performance are measured """
        return self.acc_sum / self.samples

    def end_epoch(self):
        self.acc_sum = 0
        self.lss_sum = 0
        self.samples = 0


@click.group()
def bars(): pass

@bars.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--color", "color", type=str, default="white")
@click.pass_obj
def trainbar(trainer,color):
    trainer.set_trainbar(ESLABar(color=color))

@bars.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--color", "color", type=str, default="white")
@click.pass_obj
def validbar(trainer,color):
    trainer.set_validbar(ESLABar(color=color))

@bars.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--color", "color", type=str, default="white")
@click.pass_obj
def testbar(trainer,color):
    trainer.set_testbar(ESLABar(color=color))
