from datasets import load_dataset
from semeqv.problem import Dataset
from datasets.fingerprint import Hasher

import transformers
import inspect
import random
import click
import torch

tkn2dst = {20952: (0, [0.5, 0.5]), 819: (1, [0.1, 0.9]), 8024: (2, [0.5, 0.5]), 4572: (3, [0.5, 0.5]), 22174: (4, [0.5, 0.5]), 19349: (5, [0.1, 0.9]), 1041: (6, [0.5, 0.5]), 3070: (7, [0.5, 0.5]), 7623: (8, [0.5, 0.5]), 18390: (9, [0.5, 0.5]), 23462: (10, [0.1, 0.9]), 7223: (11, [0.1, 0.9]), 19309: (12, [0.1, 0.9]), 26523: (13, [0.5, 0.5]), 24864: (14, [0.5, 0.5]), 22876: (15, [0.1, 0.9]), 11149: (16, [0.1, 0.9]), 5094: (17, [0.5, 0.5]), 25018: (18, [0.1, 0.9]), 3349: (19, [0.5, 0.5]), 12449: (20, [0.5, 0.5]), 11763: (21, [0.1, 0.9]), 19782: (22, [0.1, 0.9]), 26447: (23, [0.5, 0.5]), 23911: (24, [0.1, 0.9]), 17571: (25, [0.5, 0.5]), 30221: (26, [0.1, 0.9]), 2582: (27, [0.1, 0.9]), 27177: (28, [0.1, 0.9]), 18918: (29, [0.5, 0.5]), 23087: (30, [0.5, 0.5]), 1501: (31, [0.5, 0.5]), 25331: (32, [0.1, 0.9]), 2614: (33, [0.5, 0.5]), 28392: (34, [0.5, 0.5]), 12455: (35, [0.1, 0.9]), 14857: (36, [0.1, 0.9]), 5329: (37, [0.1, 0.9]), 11641: (38, [0.5, 0.5]), 21960: (39, [0.1, 0.9]), 22997: (40, [0.5, 0.5]), 19960: (41, [0.5, 0.5]), 17502: (42, [0.5, 0.5]), 5354: (43, [0.1, 0.9]), 12433: (44, [0.1, 0.9]), 30323: (45, [0.5, 0.5]), 22433: (46, [0.1, 0.9]), 27618: (47, [0.5, 0.5]), 7505: (48, [0.5, 0.5]), 26379: (49, [0.1, 0.9]), 13145: (50, [0.1, 0.9]), 2168: (51, [0.5, 0.5]), 29921: (52, [0.1, 0.9]), 6967: (53, [0.1, 0.9]), 12964: (54, [0.1, 0.9]), 4681: (55, [0.1, 0.9]), 4575: (56, [0.5, 0.5]), 24411: (57, [0.1, 0.9]), 24478: (58, [0.1, 0.9]), 29419: (59, [0.1, 0.9]), 11861: (60, [0.5, 0.5]), 4532: (61, [0.1, 0.9]), 2978: (62, [0.5, 0.5]), 28216: (63, [0.5, 0.5]), 5008: (82, [0.1, 0.9]), 25954: (65, [0.1, 0.9]), 19543: (66, [0.5, 0.5]), 12608: (67, [0.1, 0.9]), 19526: (68, [0.1, 0.9]), 17338: (69, [0.1, 0.9]), 18128: (70, [0.5, 0.5]), 22291: (71, [0.5, 0.5]), 22338: (72, [0.1, 0.9]), 25185: (73, [0.1, 0.9]), 3655: (74, [0.1, 0.9]), 14246: (75, [0.5, 0.5]), 14867: (76, [0.5, 0.5]), 23661: (77, [0.1, 0.9]), 16403: (78, [0.5, 0.5]), 16635: (79, [0.5, 0.5]), 28525: (80, [0.1, 0.9]), 27579: (81, [0.5, 0.5]), 24985: (83, [0.5, 0.5]), 17674: (84, [0.5, 0.5]), 19626: (85, [0.1, 0.9]), 16010: (86, [0.5, 0.5]), 3665: (87, [0.1, 0.9]), 28793: (88, [0.1, 0.9]), 7846: (89, [0.5, 0.5]), 7892: (90, [0.5, 0.5]), 2806: (91, [0.1, 0.9]), 26739: (92, [0.5, 0.5]), 24923: (93, [0.5, 0.5]), 4207: (94, [0.1, 0.9]), 18015: (95, [0.5, 0.5]), 8685: (96, [0.1, 0.9]), 6940: (97, [0.5, 0.5]), 23361: (98, [0.1, 0.9]), 13074: (99, [0.1, 0.9])}


class DefaultDataset(Dataset):
    """ dataset to train MLM using bookcorpus [Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books]
        returns (masked sentence, masked labels) pairs.
    """
    
    def __init__(self, split="train", sample_size=128, num_proc=8, device="cpu", seed=42):

        self.device      = device
        self.sample_size = sample_size
        self.num_proc    = num_proc

        split2lh = {
            "train"      : (0, 70),
            "validation" : (70,80),
            "test"       : (80,100)
        }
        low,high = split2lh[split]

        self.raw_dataset = load_dataset("bookcorpus", split=f"train[{low}%:{high}%]")
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        DefaultDataset.collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=.15)
        self.dataset = self.raw_dataset.map(
            function       = self.encode,
            num_proc       = num_proc,
            batched        = True,
            batch_size     = 100,
            remove_columns = self.raw_dataset.column_names, 
            new_fingerprint=Hasher.hash((self.raw_dataset,self.sample_size,self.tokenizer,inspect.getsource(self.encode))))

    def encode(self, data):
        return self.tokenizer(data["text"], truncation = True, max_length = self.sample_size)

    def __len__(self):
        return len(self.dataset)

    def augment(self, elem):
        if elem not in tkn2dst: return elem
        return random.choices([elem,len(self.tokenizer)+tkn2dst[elem][0]], weights=tkn2dst[elem][1], k=1)[0]

    def __getitem__(self, idx):
        return {
            "input_ids" : [self.augment(e) for e in self.dataset[idx]["input_ids"]], 
            "token_type_ids" : self.dataset[idx]["token_type_ids"],
            "attention_mask" : self.dataset[idx]["attention_mask"]
        } 

    def todevice(self, src, msk, tgt):
        return {
            "src" : src.to(self.device),
            "msk" : msk.to(self.device),
            "tgt" : tgt.to(self.device)
        }

    def collate_fn(self,data):
        result = DefaultDataset.collator(data)
        tgt = result["labels"].flatten()
        return {
            "src" : result["input_ids"],
            "msk" : result["attention_mask"],
            "tgt" : tgt
        }

def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.generator = random.Random(seed + worker_id + worker_info.seed)
    return worker_init_fn

@click.group()
def default_dataset(): pass

@default_dataset.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--size"        , "sample_size" , type=int  , default=10)
@click.option("--num_proc"    , "num_proc"    , type=int  , default=8)
@click.option("--batch_size"  , "batch_size"  , type=int  , default=100)
@click.option("--num_workers" , "num_workers" , type=int  , default=1)
@click.option("--shuffle"     , "shuffle"     , type=bool , default=True)
@click.option("--drop_last"   , "drop_last"   , type=bool , default=True)
@click.option("--device"      , "device"      , type=str  , default="cpu")
@click.option("--seed"        , "seed"        , type=int  , default=42)
@click.pass_obj
def trainsplit(trainer, sample_size, num_proc, batch_size, num_workers, shuffle, drop_last, device, seed):
    trainer.set_trainsplit(
        torch.utils.data.DataLoader(
            dataset := DefaultDataset(
                sample_size = sample_size,
                num_proc    = num_proc,
                device      = device,
                split       = "train",
                seed        = seed,
            ),
            batch_size     = batch_size,
            num_workers    = num_workers,
            shuffle        = shuffle,
            drop_last      = drop_last,
            collate_fn     = dataset.collate_fn,
            worker_init_fn = get_worker_init_fn(seed)
        )
    )

@default_dataset.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--size"        , "sample_size" , type=int  , default=10)
@click.option("--num_proc"    , "num_proc"    , type=int  , default=8)
@click.option("--batch_size"  , "batch_size"  , type=int  , default=100)
@click.option("--num_workers" , "num_workers" , type=int  , default=1)
@click.option("--shuffle"     , "shuffle"     , type=bool , default=True)
@click.option("--drop_last"   , "drop_last"   , type=bool , default=True)
@click.option("--device"      , "device"      , type=str  , default="cpu")
@click.option("--seed"        , "seed"        , type=int  , default=42)
@click.pass_obj
def validsplit(trainer, sample_size, num_proc, batch_size, num_workers, shuffle, drop_last, device, seed):
    trainer.set_validsplit(
        torch.utils.data.DataLoader(
            dataset := DefaultDataset(
                sample_size = sample_size,
                num_proc    = num_proc,
                device      = device,
                split       = "validation"
            ),
            batch_size     = batch_size,
            num_workers    = num_workers,
            shuffle        = shuffle,
            drop_last      = drop_last,
            collate_fn     = dataset.collate_fn,
            worker_init_fn = get_worker_init_fn(seed)
        )
    )

@default_dataset.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--size"        , "sample_size" , type=int  , default=10)
@click.option("--num_proc"    , "num_proc"    , type=int  , default=8)
@click.option("--batch_size"  , "batch_size"  , type=int  , default=100)
@click.option("--num_workers" , "num_workers" , type=int  , default=1)
@click.option("--shuffle"     , "shuffle"     , type=bool , default=True)
@click.option("--drop_last"   , "drop_last"   , type=bool , default=True)
@click.option("--device"      , "device"      , type=str  , default="cpu")
@click.option("--seed"        , "seed"        , type=int  , default=42)
@click.pass_obj
def testplit(trainer, sample_size, num_proc, batch_size, num_workers, shuffle, drop_last, device, seed):
    trainer.set_testsplit(
        torch.utils.data.DataLoader(
            dataset := DefaultDataset(
                sample_size = sample_size,
                num_proc    = num_proc,
                device      = device,
                split       = "test"
            ),
            batch_size     = batch_size,
            num_workers    = num_workers,
            shuffle        = shuffle,
            drop_last      = drop_last,
            collate_fn     = dataset.collate_fn,
            worker_init_fn = get_worker_init_fn(seed)
        )
    )
