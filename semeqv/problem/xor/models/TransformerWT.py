from semeqv.problem.xor.models import model
from semeqv.problem.xor.models import Transformer
import click

class TransformerWT(Transformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_embedding = self.input_embedding

@model.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--layers"           , "layers"           , type=int       , default=3)
@click.option("--src_vocab_size"   , "src_vocab_size"   , type=int       , default=2)
@click.option("--tgt_vocab_size"   , "tgt_vocab_size"   , type=int       , default=2)
@click.option("--embedding_size"   , "embedding_size"   , type=int       , default=128)
@click.option("--feedforward_size" , "feedforward_size" , type=int       , default=512)
@click.option("--heads"            , "heads"            , type=int       , default=2)
@click.option("--semeqvinit"       , "semeqvinit"       , type=(int,int) , default=(0,0))
@click.option("--dropout"          , "dropout"          , type=float     , default=.1)
@click.option("--activation"       , "activation"       , type=str       , default="relu")
@click.option("--device"           , "device"           , type=str       , default="cpu")
@click.pass_obj
def transformerwt(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, feedforward_size, heads, semeqvinit, dropout, activation, device):
    trainer.set_model(
        TransformerWT(
            layers           = layers,
            src_vocab_size   = src_vocab_size,
            tgt_vocab_size   = tgt_vocab_size,
            embedding_size   = embedding_size,
            feedforward_size = feedforward_size,
            heads            = heads,
            dropout          = dropout,
            semeqv_init      = semeqvinit,
            activation       = activation,
            device           = device
        )
    )


