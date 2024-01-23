from semeqv.problem import Dataset as BaseDataset
from semeqv.problem.mlm.datasets import dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from semeqv.problem import split2dataset
import datasets
import random
import torch
import click

common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us" ] 

class Default(BaseDataset):
    """ 
        Dataset class for mlm problem with semantically eqv. symbols.
        Given a natural language masked token sequence the model must recover the masked tokens.
        Args:
            split: the split of the dataset to use ("train", "validation", "test").
    """

    def __init__(
            self,
            num_proc       = 4,
            mlm_prb        = .15,
            swp_prb        = .5,
            max_length     = 256,
            map_batch_size = 5000,
            seed           = 42,
            device         = "cpu",
            split          = "train"
        ):

        self.generator = random.Random(seed)
        self.device  = device
        self.split   = split
        self.mlm_prb = mlm_prb
        self.swp_prb = swp_prb
        self.max_length = max_length

        split2percentage = {
            "train"      : "train[0%:70%]",
            "validation" : "train[70%:80%]",
            "test"       : "train[80%:20%]"
        } 
        
        self.dataset             = datasets.load_dataset("bookcorpus", trust_remote_code=True, split=split2percentage[split])
        self.tokenizer           = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.commonidx2semeqvidx = {self.tokenizer.convert_tokens_to_ids(word):self.tokenizer.vocab_size + i for i,word in enumerate(common_words)}
        self.data_collator       = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_prb)

        self.tokenized = self.dataset.map(
            self.preprocess_function,
            batched        = True,
            num_proc       = num_proc,
            batch_size     = map_batch_size,
            remove_columns = self.dataset.column_names,
        )

    def preprocess_function(self,data):
        return self.tokenizer(list(map(lambda x:x.lower(), data["text"])), max_length=self.max_length, truncation=True)

    def save(self):
        return self.generator.getstate()

    def restore(self, state):
        return self.generator.setstate(state)

    def __len__(self): 
        return len(self.tokenized)

    def __getitem__(self, idx): 
        result = self.tokenized[idx]
        return result | {"input_ids" : [idx if idx not in self.commonidx2semeqvidx else (self.commonidx2semeqvidx[idx] if self.generator.random() < self.swp_prb else idx) for idx in result["input_ids"]]}

    def todevice(self, src, tgt):
        return {
            "src" : { 
                "input_ids"      : src[     "input_ids"].to(self.device, non_blocking=True),
                "attention_mask" : src["attention_mask"].to(torch.bool).logical_not().to(self.device, non_blocking=True)
            },
            "tgt": tgt.to(self.device, non_blocking=True)
        }

    def collate_fn(self, data): 
        result = self.data_collator(data)
        return {
            "src" : {
                "input_ids"      : result["input_ids"],
                "attention_mask" : result["attention_mask"]
            },
            "tgt" : result["labels"].flatten()
        }

@dataset.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--seed"           , "seed"           , type=int   , default=42)
@click.option("--batch_size"     , "batch_size"     , type=int   , default=100)
@click.option("--mlm_prb"        , "mlm_prb"        , type=float , default=.15)
@click.option("--swp_prb"        , "swp_prb"        , type=float , default=.5)
@click.option("--max_length"     , "max_length"     , type=int   , default=256)
@click.option("--map_batch_size" , "map_batch_size" , type=int   , default=5000)
@click.option("--shuffle"        , "shuffle"        , type=bool  , default=True)
@click.option("--drop_last"      , "drop_last"      , type=bool  , default=True)
@click.option("--device"         , "device"         , type=str   , default="cpu")
@click.option("--split"          , "split"          , type=str   , default="train")
@click.pass_obj
def default(trainer, seed, mlm_prb, swp_prb, max_length, map_batch_size, batch_size, shuffle, drop_last, device, split):
    split2dataset[split](trainer)(
        torch.utils.data.DataLoader(
            dataset := Default(
                mlm_prb        = mlm_prb,
                swp_prb        = swp_prb,
                max_length     = max_length,
                map_batch_size = map_batch_size,
                seed           = seed,
                device         = device,
                split          = split
            ),
            batch_size  = batch_size,
            num_workers = 0,
            shuffle     = shuffle,
            drop_last   = drop_last,
            collate_fn  = dataset.collate_fn
        )
    )


