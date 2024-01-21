from semeqv.problem.xor.models import model
from semeqv.problem.xor.models import MLP
import click

class MLPWT(MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_embedding = self.input_embedding

@model.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--layers"                       , "layers"                       , type=int , default=3)
@click.option("--src_vocab_size"               , "src_vocab_size"               , type=int , default=2)
@click.option("--tgt_vocab_size"               , "tgt_vocab_size"               , type=int , default=2)
@click.option("--embedding_size"               , "embedding_size"               , type=int , default=128)
@click.option("--sequence_size"                , "sequence_size"                , type=int , default=128)
@click.option("--tokenwise_feedforward_size"   , "tokenwise_feedforward_size"   , type=int , default=512)
@click.option("--channelwise_feedforward_size" , "channelwise_feedforward_size" , type=int , default=512)
@click.option("--activation"                   , "activation"                   , type=str , default="relu")
@click.option("--device"                       , "device"                       , type=str , default="cpu")
@click.pass_obj
def mlpwt(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, sequence_size, tokenwise_feedforward_size, channelwise_feedforward_size, activation, device):
    trainer.set_model(
        MLPWT(
            layers                       = layers,
            src_vocab_size               = src_vocab_size,
            tgt_vocab_size               = tgt_vocab_size,
            embedding_size               = embedding_size,
            sequence_size                = sequence_size,
            tokenwise_feedforward_size   = tokenwise_feedforward_size,
            channelwise_feedforward_size = channelwise_feedforward_size,
            activation                   = activation,
            device                       = device
        )
    )

