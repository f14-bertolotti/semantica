from semeqv.problem.xor.models import model
from semeqv.problem import name2activation
import click, torch

class MLPLayer(torch.nn.Module):
    def __init__(
            self,
            embedding_size   = 128,
            feedforward_size = 512,
            activation       = "relu",
            device           = "cpu"):
        super().__init__()
        
        self.fc0 = torch.nn.Linear(embedding_size, feedforward_size, device=device)
        self.act = name2activation[activation]
        self.fc1 = torch.nn.Linear(feedforward_size, embedding_size, device=device)

    def forward(self, x):
         return self.fc1(self.act(self.fc0(x)))

class MixerLayer(torch.nn.Module):
    def __init__(
            self,
            embedding_size               = 128,
            sequence_size                = 128,
            tokenwise_feedforward_size   = 512,
            channelwise_feedforward_size = 512,
            activation                   = "relu",
            device                       = "cpu"
        ):
        super(MixerLayer, self).__init__()
        self.tokenwise_layer   = MLPLayer(sequence_size  , tokenwise_feedforward_size   , activation , device)
        self.channelwise_layer = MLPLayer(embedding_size , channelwise_feedforward_size , activation , device)
        self.pre_norm  = torch.nn.LayerNorm(embedding_size, device=device)
        self.post_norm = torch.nn.LayerNorm(embedding_size, device=device)

    def forward(self, x):
        x = self.tokenwise_layer(self.pre_norm(x).transpose(-1, -2)).transpose(-1, -2) + x
        x = self.channelwise_layer(self.post_norm(x)) + x
        return x


class MLP(torch.nn.Module):

    def __init__(
            self,
            layers           = 3,
            src_vocab_size   = 2,
            tgt_vocab_size   = 2,
            embedding_size   = 128,
            sequence_size    = 128,
            tokenwise_feedforward_size = 512,
            channelwise_feedforward_size = 512,
            activation       = "relu",
            device           = "cpu"):
        super().__init__()

        self.input_embedding  = torch.nn.Embedding(src_vocab_size, embedding_size, device=device)
        self.output_embedding = torch.nn.Embedding(tgt_vocab_size, embedding_size, device=device)
        self.layers = torch.nn.Sequential(*[
            MixerLayer(
                embedding_size, 
                sequence_size,
                tokenwise_feedforward_size,
                channelwise_feedforward_size,
                activation, 
                device
            ) 
            for _ in range(layers)]
        )

    def forward(self, src, tgt):
        encoded = self.input_embedding(src)
        for layer in self.layers: encoded = layer(encoded)
        logits  = encoded @ self.output_embedding.weight.T
        logits  = logits.view(logits.size(0)*logits.size(1),logits.size(2))
        return {"logits": logits, "prd" : logits.argmax(-1)}

    def save(self):
        return self.state_dict()

    def restore(self, state_dict):
        self.load_state_dict(state_dict)

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
def mlp(trainer, layers, src_vocab_size, tgt_vocab_size, embedding_size, sequence_size, tokenwise_feedforward_size, channelwise_feedforward_size, activation, device):
    trainer.set_model(
        MLP(
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

