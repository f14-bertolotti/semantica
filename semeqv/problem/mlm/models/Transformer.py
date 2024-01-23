from semeqv.problem.mlm.models import model
import click, torch

class Transformer(torch.nn.Module):

    def __init__(
            self,
            trainer,
            layers            = 3,
            embedding_size    = 128,
            feedforward_size  = 512,
            heads             = 2,
            dropout           = .1,
            initializer_range = .3,
            activation        = "relu",
            device            = "cpu"):
        super().__init__()
        vocab_size = trainer.trainsplit.dataset.tokenizer.vocab_size
        pad_token  = trainer.trainsplit.dataset.tokenizer.pad_token
        pad_id     = trainer.trainsplit.dataset.tokenizer.convert_tokens_to_ids(pad_token)

        self.initializer_range = initializer_range

        self.input_embedding  = torch.nn.Embedding(vocab_size+100, embedding_size, padding_idx=pad_id, device=device)
        self.output_embedding = torch.nn.Embedding(vocab_size+100, embedding_size, padding_idx=pad_id, device=device)

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
            num_layers = layers,
            enable_nested_tensor=heads%2==0
        )

        self.apply(self.init_weights)

    def forward(self, src, tgt):
        src_ids, src_msk = src["input_ids"], src["attention_mask"]
        embeddings = self.input_embedding(src_ids)
        encoded    = self.encoder(embeddings, src_key_padding_mask = src_msk)
        logits     = encoded @ self.output_embedding.weight.T
        logits     = logits.view(logits.size(0)*logits.size(1),logits.size(2))
        return {"logits": logits, "prd" : logits.argmax(-1)}

    def init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def save(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

@model.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--layers"            , "layers"            , type=int   , default=3)
@click.option("--embedding_size"    , "embedding_size"    , type=int   , default=128)
@click.option("--feedforward_size"  , "feedforward_size"  , type=int   , default=512)
@click.option("--heads"             , "heads"             , type=int   , default=2)
@click.option("--initializer_range" , "initializer_range" , type=float , default=.3)
@click.option("--dropout"           , "dropout"           , type=float , default=.1)
@click.option("--activation"        , "activation"        , type=str   , default="relu")
@click.option("--device"            , "device"            , type=str   , default="cpu")
@click.pass_obj
def transformer(trainer, layers, embedding_size, feedforward_size, heads, initializer_range, dropout, activation, device):
    trainer.set_model(
        Transformer(
            trainer           = trainer,
            layers            = layers,
            embedding_size    = embedding_size,
            feedforward_size  = feedforward_size,
            heads             = heads,
            initializer_range = initializer_range,
            dropout           = dropout,
            activation        = activation,
            device            = device
        )
    )

