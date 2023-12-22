from semeqv.problem.callbacks import callbacks, DefaultCallback
import logging, click

class LoggingCallback(DefaultCallback):

    def __init__(self, trainer, step_log_path="", epoch_log_path=""): 
        super().__init__(trainer)

        self.format = "{\"lebel\":%(levelname)s, \"time\":%(asctime)s, \"name\":%(name)s, \"message\":{%(message)s}}"

        self.step_logger  = logging.getLogger(step_log_path)
        self.epoch_logger = logging.getLogger(epoch_log_path)

        if step_log_path:
            handler  = logging.FileHandler( step_log_path)
            handler .setFormatter(logging.Formatter(self.format))
            self.step_logger.addHandler(handler)

        if epoch_log_path:
            handler = logging.FileHandler(epoch_log_path)
            handler.setFormatter(logging.Formatter(self.format))
            self.epoch_logger.addHandler(handler)

        self. step_logger.setLevel(logging.INFO)
        self.epoch_logger.setLevel(logging.INFO)

    def end_step(self, loss, data, pred):
        super().end_step(loss, data, pred)
        self.step_logger.info(f"\"epoch\":{self.epoch}, \"step\":{self.step}, \"accuracy\":{self.cur_acc/self.cur_spl}, \"loss\":{loss}")

    def end_epoch(self, *args, **kwargs):
        acc, lss = self.get_step_results()
        self.epoch_logger.info(f"\"epoch\":{self.epoch}, \"step\":{self.step}, \"accuracy\":{acc}, \"loss\":{lss}")
        super().end_epoch(*args, **kwargs)


@click.group()
def log(): pass
callbacks.add_command(log)

@log.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.pass_obj
def traincallback(trainer, step_log_path, epoch_log_path):
    trainer.set_traincallback(LoggingCallback(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path))

@log.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.pass_obj
def validcallback(trainer, step_log_path, epoch_log_path):
    trainer.set_validcallback(LoggingCallback(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path))

@log.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option( "--step_log_path",  "step_log_path", type=str, default="")
@click.option("--epoch_log_path", "epoch_log_path", type=str, default="")
@click.pass_obj
def testcallback(trainer, step_log_path, epoch_log_path):
    trainer.set_testcallback(LoggingCallback(trainer, step_log_path=step_log_path, epoch_log_path=epoch_log_path))
