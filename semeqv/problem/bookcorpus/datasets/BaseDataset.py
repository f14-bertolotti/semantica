from semeqv.problem import Dataset
import datasets, click

class BaseDataset(Dataset):
    def __init__(self, split="train", batch_size=1000):
        super().__init__()
        self.batch_size = batch_size
        split2lh = {
            "train"      : (0, 70),
            "validation" : (70,80),
            "test"       : (80,100)
        }
        low,high = split2lh[split]
        self.raw_dataset = datasets.load_dataset("bookcorpus", split=f"train[{low}%:{high}%]")

    def __iter__(self):
        for i in range(0, len(self.raw_dataset), self.batch_size):
            yield self.raw_dataset[i : i + self.batch_size]["text"]

    def __len__(self):
        return len(self.raw_dataset)

@click.group(invoke_without_command=True, context_settings={'show_default': True})
@click.option("--split", "split", type=str, default="train")
@click.option("--batch_size", "batch_size", type=int, default=1000)
@click.pass_obj
def base_dataset(trainer, split, batch_size):
    trainer.set_trainsplit(
        BaseDataset(
            split = split,
            batch_size = batch_size
        ),
    )
