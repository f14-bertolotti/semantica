from semeqv.problem.xor.models import model
import click, torch

class Default(torch.nn.Module):

    def __init__(
            self,
            layers           = 3,
            src_vocab_size   = 2,
            tgt_vocab_size   = 2,
            embedding_size   = 128,
            feedforward_size = 512,
            heads            = 2,
            dropout          = .1,
            semeqv_init      = (0,0),
            activation       = "relu",
            device           = "cpu"):
        super().__init__()
        self.input_embedding  = torch.nn.Embedding(src_vocab_size, embedding_size, device=device)
        self.output_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size, device=device)

        if semeqv_init[0] and semeqv_init[0]:
            with torch.no_grad():
                self.embedding.weight[:semeqv_init[0]] = self.embedding.weight[0]
                self.embedding.weight[semeqv_init[0]:semeqv_init[0]+semeqv_init[1]] = self.embedding.weight[semeqv_init[0]]

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

    def forward(self, src, tgt):
        embeddings = self.input_embedding(src)
        encoded    = self.encoder(embeddings)
        logits     = encoded @ self.output_embedding.weight.T
        logits     = logits.view(logits.size(0)*logits.size(1),logits.size(2))
        return {"logits": logits, "prd" : logits.argmax(-1)}

    def save(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

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
def default(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, feedforward_size, heads, semeqvinit, dropout, activation, device):
    trainer.set_model(
        Default(
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

