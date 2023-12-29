import termcolor, click, torch, tqdm
from tokenizers import ByteLevelBPETokenizer
from tokenizers import trainers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
import tokenizers

@click.command()
@click.option("--vocab_size", "vocab_size", type=int, default=10000)
@click.option("--path"      , "path"      , type=str, default="./tokenizer.json")
@click.pass_obj
def train_tokenizer(trainer, vocab_size, path):
    tokenizer = ByteLevelBPETokenizer()
    
    tokenizer.train_from_iterator(
        trainer.trainsplit, 
        vocab_size     = vocab_size,
        special_tokens = ["<PAD>","<MSK>","<SOS>","<EOS>","<UNK>","<SEP>","<CLS>"],
        length         = len(trainer.trainsplit)
    )
    tokenizer.save(path)
