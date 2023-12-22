import click
from semeqv.problem.bookcorpus.datasets import default_dataset
from semeqv.problem.bookcorpus  .models import default_model
from semeqv.problem.bookcorpus  .models import default_model_wt

@click.group()
def bookcorpus(): pass


bookcorpus.add_command(default_model)
bookcorpus.add_command(default_model_wt)
bookcorpus.add_command(default_dataset)

