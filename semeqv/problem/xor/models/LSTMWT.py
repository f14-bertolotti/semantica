from semeqv.problem.xor.models import model
from semeqv.problem.xor.models import LSTM
import click

class LSTMWT(LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_embedding = self.input_embedding

@model.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--layers"         , "layers"         , type=int  , default=3)
@click.option("--src_vocab_size" , "src_vocab_size" , type=int  , default=2)
@click.option("--tgt_vocab_size" , "tgt_vocab_size" , type=int  , default=2)
@click.option("--embedding_size" , "embedding_size" , type=int  , default=128)
@click.option("--hidden_size"    , "hidden_size"    , type=int  , default=512)
@click.option("--bidirectional"  , "bidirectional"  , type=bool , default=True)
@click.option("--device"         , "device"         , type=str  , default="cpu")
@click.pass_obj
def lstmwt(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, hidden_size, bidirectional, device):
    trainer.set_model(
        LSTMWT(
            layers         = layers,
            src_vocab_size = src_vocab_size,
            tgt_vocab_size = tgt_vocab_size,
            embedding_size = embedding_size,
            hidden_size    = hidden_size,
            bidirectional  = bidirectional,
            device         = device
        )
    )

