from semeqv.problem.xor.models import model
from semeqv.problem import name2activation
import click, torch

class LSTM(torch.nn.Module):

    def __init__(
            self,
            layers           = 3,
            src_vocab_size   = 2,
            tgt_vocab_size   = 2,
            embedding_size   = 128,
            hidden_size      = 512,
            bidirectional    = True,
            device           = "cpu"):
        super().__init__()

        self.input_embedding  = torch.nn.Embedding(src_vocab_size, embedding_size, device=device)
        self.output_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size, device=device)
        self.feed_forward     = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), embedding_size, device=device)
        self.lstm = torch.nn.LSTM(
            input_size    = embedding_size,
            hidden_size   = hidden_size,
            num_layers    = layers,
            device        = device,
            batch_first   = True,
            bidirectional = bidirectional
        )

    def forward(self, src, tgt):
        embeddings = self.input_embedding(src)
        encoded = self.feed_forward(self.lstm(embeddings)[0])
        logits  = encoded @ self.output_embedding.weight.T
        logits  = logits.view(logits.size(0)*logits.size(1),logits.size(2))
        return {"logits": logits, "prd" : logits.argmax(-1)}

    def save(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

@model.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--layers"         , "layers"         , type=int  , default=3)
@click.option("--src_vocab_size" , "src_vocab_size" , type=int  , default=2)
@click.option("--tgt_vocab_size" , "tgt_vocab_size" , type=int  , default=2)
@click.option("--embedding_size" , "embedding_size" , type=int  , default=128)
@click.option("--hidden_size"    , "hidden_size"    , type=int  , default=512)
@click.option("--bidirectional"  , "bidirectional"  , type=bool , default=True)
@click.option("--device"         , "device"         , type=str  , default="cpu")
@click.pass_obj
def lstm(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, hidden_size, bidirectional, device):
    trainer.set_model(
        LSTM(
            layers         = layers,
            src_vocab_size = src_vocab_size,
            tgt_vocab_size = tgt_vocab_size,
            embedding_size = embedding_size,
            hidden_size    = hidden_size,
            bidirectional  = bidirectional,
            device         = device
        )
    )

