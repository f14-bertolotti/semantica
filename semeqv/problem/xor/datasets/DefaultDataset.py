from semeqv.problem import Dataset as BaseDataset
from semeqv.problem import DistributionOption
import random
import torch
import click

class DefaultDataset(BaseDataset):
    """ 
        Dataset class for xor problem with semantically eqv. symbols.
        Given a binary string the prediction should be the sum of bits mod 2.
        Args:
            sample_size: the number of bits.
            semeqv_symbols: the number of semeqv symbols to add.
            semeqv_distribtion: the distribution of the semeqv symbols, ("equal", "random").
            split: the split of the dataset to use ("train", "validation", "test").
    """

    def __init__(
            self,
            sample_size              = 10,
            zero_semeqv_distribution = [1],
            one_semeqv_distribution  = [1],
            seed                     = 42,
            split                    = "train"
        ):
        self.generator = random.Random(seed)

        # build the dataset with all possible binary combinations
        samples = [list(map(int,bin(x)[2:])) for x in range(2**sample_size - 1)]
        samples = [[0]*(sample_size-len(x))+x for x in samples]
        predictions = [sum(data)%2 for data in samples]

        # build train/validation/test splits
        split2samples = {
            "train"      : (trainsplit := samples[:int(len(samples)*.7)]),
            "validation" : (validsplit := samples[len(trainsplit):len(trainsplit)+int(len(samples)*.1)]),
            "test"       : (testsplit  := samples[len(trainsplit)+len(validsplit):])
        }
        split2predictions = {
            "train"      : predictions[:len(trainsplit)],
            "validation" : predictions[len(trainsplit):len(trainsplit)+len(validsplit)],
            "test"       : predictions[len(trainsplit)+len(validsplit):]
        }
    
        # map the dataset split to the current split and add semeqv symbols according to the provided distribution
        samples = split2samples[split]
        samples = [list(map(lambda x: self.generator.choices(range(len(zero_semeqv_distribution)),weights=zero_semeqv_distribution,k=1)[0] if x==0 else len(zero_semeqv_distribution), sample)) for sample in samples]
        samples = [list(map(lambda x: self.generator.choices(range(len(zero_semeqv_distribution),len(zero_semeqv_distribution)+len(one_semeqv_distribution)),weights=one_semeqv_distribution,k=1)[0] if x==len(zero_semeqv_distribution) else x, sample)) for sample in samples]
        predictions = split2predictions[split]
        predictions = list(map(lambda x: x, predictions))

        # from list dataset to torch dataset
        self.dataset = (torch.tensor(samples), torch.tensor(predictions))

    def __len__(self): 
        return len(self.dataset[0])

    def __getitem__(self, idx): 
        return self.dataset[0][idx], self.dataset[1][idx]

    def collate_fn(self, data): 
        return {"src" : torch.stack([d[0] for d in data]),
                "tgt" : torch.stack([d[1] for d in data])}

@click.group()
def default_dataset(): pass


@default_dataset.group(invoke_without_command=True)
@click.option("--size"        , "sample_size"              , type=int               , default=10)
@click.option("--zero_dst"    , "zero_semeqv_distribution" , cls=DistributionOption , default="[1]")
@click.option("--one_dst"     , "one_semeqv_distribution"  , cls=DistributionOption , default="[1]")
@click.option("--seed"        , "seed"                     , type=int               , default=14)
@click.option("--batch_size"  , "batch_size"               , type=int               , default=100)
@click.option("--num_workers" , "num_workers"              , type=int               , default=1)
@click.option("--shuffle"     , "shuffle"                  , type=bool              , default=True)
@click.option("--drop_last"   , "drop_last"                , type=bool              , default=True)
@click.pass_obj
def trainsplit(trainer, sample_size, zero_semeqv_distribution, one_semeqv_distribution, seed, batch_size, num_workers, shuffle, drop_last):
    trainer.set_trainsplit(
        torch.utils.data.DataLoader(
            dataset := DefaultDataset(
                sample_size              = sample_size,
                zero_semeqv_distribution = zero_semeqv_distribution,
                one_semeqv_distribution  = one_semeqv_distribution,
                seed                     = seed,
                split                    = "train"
            ),
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = shuffle,
            drop_last   = drop_last,
            collate_fn  = dataset.collate_fn
        )
    )

@default_dataset.group(invoke_without_command=True)
@click.option("--size"        , "sample_size"              , type=int               , default=10)
@click.option("--zero_dst"    , "zero_semeqv_distribution" , cls=DistributionOption , default="[1]")
@click.option("--one_dst"     , "one_semeqv_distribution"  , cls=DistributionOption , default="[1]")
@click.option("--seed"        , "seed"                     , type=int               , default=14)
@click.option("--batch_size"  , "batch_size"               , type=int               , default=100)
@click.option("--num_workers" , "num_workers"              , type=int               , default=1)
@click.option("--shuffle"     , "shuffle"                  , type=bool              , default=True)
@click.option("--drop_last"   , "drop_last"                , type=bool              , default=True)
@click.pass_obj
def validsplit(trainer, sample_size, zero_semeqv_distribution, one_semeqv_distribution, seed, batch_size, num_workers, shuffle, drop_last):
    trainer.set_validsplit(
        torch.utils.data.DataLoader(
            dataset := DefaultDataset(
                sample_size              = sample_size,
                zero_semeqv_distribution = zero_semeqv_distribution,
                one_semeqv_distribution  = one_semeqv_distribution,
                seed                     = seed,
                split                    = "validation"
            ),
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = shuffle,
            drop_last   = drop_last,
            collate_fn  = dataset.collate_fn
        )
    )

@default_dataset.group(invoke_without_command=True)
@click.option("--size"        , "sample_size"              , type=int               , default=10)
@click.option("--zero_dst"    , "zero_semeqv_distribution" , cls=DistributionOption , default="[1]")
@click.option("--one_dst"     , "one_semeqv_distribution"  , cls=DistributionOption , default="[1]")
@click.option("--seed"        , "seed"                     , type=int               , default=14)
@click.option("--batch_size"  , "batch_size"               , type=int               , default=100)
@click.option("--num_workers" , "num_workers"              , type=int               , default=1)
@click.option("--shuffle"     , "shuffle"                  , type=bool              , default=False)
@click.option("--drop_last"   , "drop_last"                , type=bool              , default=False)
@click.pass_obj
def testsplit(trainer, sample_size, zero_semeqv_distribution, one_semeqv_distribution, seed, batch_size, num_workers, shuffle, drop_last):
    trainer.set_testsplit(
        torch.utils.data.DataLoader(
        dataset := DefaultDataset(
            sample_size              = sample_size,
            zero_semeqv_distribution = zero_semeqv_distribution,
            one_semeqv_distribution  = one_semeqv_distribution,
            seed                     = seed,
            split                    = "validation"
        ),
            batch_size  = batch_size,
            num_workers = num_workers,
            shuffle     = shuffle,
            drop_last   = drop_last,
            collate_fn  = dataset.collate_fn
        )
    )
