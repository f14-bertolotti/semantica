from semeqv.problem import Dataset as BaseDataset
from semeqv.problem import DistributionOption
from semeqv.problem.xor.datasets import dataset
from semeqv.problem import split2dataset
import random
import torch
import click

class Default(BaseDataset):
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
            device                   = "cpu",
            split                    = "train"
        ):
        self.generator = random.Random(seed)
        self.zero_semeqv_distribution = zero_semeqv_distribution
        self.one_semeqv_distribution  =  one_semeqv_distribution
        self.device = device
        self.split = split

        # build the dataset with all possible binary combinations
        samples = [list(map(int,bin(x)[2:])) for x in range(2**sample_size - 1)]
        samples = [[0]*(sample_size-len(x))+x for x in samples]
        self.generator.shuffle(samples)
        predictions = [sum(data)%2+2 for data in samples]

        # build train/validation/test splits
        split2samples = {
            "train"      : (trainsplit := samples[:int(len(samples)*.9)]),
            "validation" : (validsplit := samples[len(trainsplit):len(trainsplit)+int(len(samples)*.1)]),
            "test"       : (testsplit  := samples[len(trainsplit)+len(validsplit):])
        }
        split2predictions = {
            "train"      : predictions[:len(trainsplit)],
            "validation" : predictions[len(trainsplit):len(trainsplit)+len(validsplit)],
            "test"       : predictions[len(trainsplit)+len(validsplit):]
        }
    
        # map the dataset to the correct split
        samples = split2samples[split]
        predictions = split2predictions[split]

        # map from binary value to semeqv symbols 
        self.bin2eqv = {
            0 : lambda: self.generator.choices(range(len(zero_semeqv_distribution)),weights=zero_semeqv_distribution,k=1)[0],
            1 : lambda: self.generator.choices(range(len(zero_semeqv_distribution),len(zero_semeqv_distribution)+len(one_semeqv_distribution)),weights=one_semeqv_distribution,k=1)[0],
            2 : lambda: len(zero_semeqv_distribution) + len(one_semeqv_distribution),
            3 : lambda: 1 + len(zero_semeqv_distribution) + len(one_semeqv_distribution)
        }

        # from list dataset to torch dataset
        self.dataset = (samples, predictions)
        self.precomputed = [self[i] for i in range(len(samples))]

    def save(self):
        return self.generator.getstate()

    def restore(self, state):
        return self.generator.setstate(state)

    def __len__(self): 
        return len(self.dataset[0])

    def __getitem__(self, idx): 
        if self.split == "train":
            src = self.dataset[0][idx] + [self.dataset[1][idx]]
            src = [self.bin2eqv[e]() for e in src]
            mask_idx = self.generator.randint(0,len(src)-2)
            tgt = [-100] * len(src)
            tgt[mask_idx] = src[mask_idx]
            src[mask_idx] = len(self.zero_semeqv_distribution) + len(self.one_semeqv_distribution) + 2
            return src, tgt
        else: 
            return self.precomputed[idx]

    def todevice(self, src, tgt):
        return {"src" : src.to(self.device), "tgt": tgt.to(self.device)}

    def collate_fn(self, data): 
        return {"src" : torch.tensor([d[0] for d in data]),
                "tgt" : torch.tensor([d[1] for d in data]).flatten()}

@dataset.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--size"        , "sample_size"              , type=int               , default=10)
@click.option("--zero_dst"    , "zero_semeqv_distribution" , cls=DistributionOption , default="[1]")
@click.option("--one_dst"     , "one_semeqv_distribution"  , cls=DistributionOption , default="[1]")
@click.option("--seed"        , "seed"                     , type=int               , default=14)
@click.option("--batch_size"  , "batch_size"               , type=int               , default=100)
@click.option("--shuffle"     , "shuffle"                  , type=bool              , default=True)
@click.option("--drop_last"   , "drop_last"                , type=bool              , default=True)
@click.option("--device"      , "device"                   , type=str               , default="cpu")
@click.option("--split"      , "split"                   , type=str               , default="train")
@click.pass_obj
def default(trainer, sample_size, zero_semeqv_distribution, one_semeqv_distribution, seed, batch_size, shuffle, drop_last, device, split):
    split2dataset[split](trainer)(
        torch.utils.data.DataLoader(
            dataset := Default(
                sample_size              = sample_size,
                zero_semeqv_distribution = zero_semeqv_distribution,
                one_semeqv_distribution  = one_semeqv_distribution,
                seed                     = seed,
                device                   = device,
                split                    = split
            ),
            batch_size  = batch_size,
            num_workers = 0,
            shuffle     = shuffle,
            drop_last   = drop_last,
            collate_fn  = dataset.collate_fn
        )
    )


