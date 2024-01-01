import transformers
import torch
import click

class DefaultModel(torch.nn.Module):

    def __init__(
            self,
            size             = 128,
            layers           = 3,
            src_vocab_size   = 10000,
            embedding_size   = 128,
            feedforward_size = 512,
            heads            = 4,
            dropout          = .1,
            activation       = "relu",
            device           = "cpu"):
        super().__init__()

        self.model = transformers.RobertaModel(
            transformers.RobertaConfig(
                vocab_size                   = src_vocab_size,
                hidden_size                  = embedding_size,
                num_hidden_layers            = layers,
                num_attention_heads          = heads,
                intermediate_size            = feedforward_size,
                hidden_act                   = activation,
                hidden_dropout_prob          = dropout,
                attention_probs_dropout_prob = 0,
                max_position_embeddings      = size+2,
                pad_token_id                 = 0
            )
        ).to(device)
        self.embedding = self.model.get_input_embeddings()
        self.output_embeddings = torch.nn.Embedding(embedding_size, src_vocab_size, device=device)
        self.apply(self.init_weights)

    def init_weights(self, module): 
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
        elif isinstance(module, torch.nn.Embedding):
            module.reset_parameters()
        elif isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()

    def forward(self, src, msk, tgt):
        output = self.model(input_ids=src, attention_mask=msk).last_hidden_state
        logits = output @ self.output_embeddings.weight
        logits = logits.view(src.size(0)*src.size(1),-1)
        return {"logits": logits, "prd" : logits.argmax(-1)}

    def save(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

@click.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--size"             , "size"             , type=int       , default=128)
@click.option("--layers"           , "layers"           , type=int       , default=3)
@click.option("--src_vocab_size"   , "src_vocab_size"   , type=int       , default=10000)
@click.option("--embedding_size"   , "embedding_size"   , type=int       , default=128)
@click.option("--feedforward_size" , "feedforward_size" , type=int       , default=512)
@click.option("--heads"            , "heads"            , type=int       , default=4)
@click.option("--dropout"          , "dropout"          , type=float     , default=.1)
@click.option("--activation"       , "activation"       , type=str       , default="relu")
@click.option("--device"           , "device"           , type=str       , default="cpu")
@click.pass_obj
def default_model(trainer, size, layers, src_vocab_size, embedding_size, feedforward_size, heads, dropout, activation, device):
    trainer.set_model(
        DefaultModel(
            size             = size,
            layers           = layers,
            src_vocab_size   = src_vocab_size,
            embedding_size   = embedding_size,
            feedforward_size = feedforward_size,
            heads            = heads,
            dropout          = dropout,
            activation       = activation,
            device           = device
        )
    )

