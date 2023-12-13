import torch
import click

class DefaultModel(torch.nn.Module):

    def __init__(
            self,
            layers           = 3,
            src_vocab_size   = 2,
            tgt_vocab_size   = 2,
            embedding_size   = 128,
            feedforward_size = 512,
            heads            = 2,
            dropout          = .1,
            activation       = "relu",
            device           = "cpu"):
        super().__init__()

        self.embedding = torch.nn.Embedding(src_vocab_size, embedding_size)
        self.encoder   = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model         = embedding_size,
                dim_feedforward = feedforward_size,
                nhead           = heads,
                activation      = activation,
                dropout         = dropout,
                device          = device,
                batch_first     = True,
            ),
            num_layers = layers
        )
        self.ff = torch.nn.Linear(embedding_size, tgt_vocab_size)

    def forward(self, src, tgt):
        embeddings = self.embedding(src)
        encoded    = self.encoder(embeddings).mean(1)
        preds      = self.ff(encoded)
        return {"prd" : preds}

@click.group(invoke_without_command=True)
@click.option("--layers"           , "layers"           , type=int   , default=3)
@click.option("--src_vocab_size"   , "src_vocab_size"   , type=int   , default=2)
@click.option("--tgt_vocab_size"   , "tgt_vocab_size"   , type=int   , default=2)
@click.option("--embedding_size"   , "embedding_size"   , type=int   , default=128)
@click.option("--feedforward_size" , "feedforward_size" , type=int   , default=512)
@click.option("--heads"            , "heads"            , type=int   , default=2)
@click.option("--dropout"          , "dropout"          , type=float , default=.1)
@click.option("--activation"       , "activation"       , type=str   , default="relu")
@click.option("--device"           , "device"           , type=str   , default="cpu")
@click.pass_obj
def default_model(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, feedforward_size, heads, dropout, activation, device):
    trainer.set_model(
        DefaultModel(
            layers           = layers,
            src_vocab_size   = src_vocab_size,
            tgt_vocab_size   = tgt_vocab_size,
            embedding_size   = embedding_size,
            feedforward_size = feedforward_size,
            heads            = heads,
            dropout          = dropout,
            activation       = activation,
            device           = device
        )
    )

