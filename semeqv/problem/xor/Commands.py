import click
from semeqv.problem.xor.datasets import default_dataset
from semeqv.problem.xor  .models import default_model
from semeqv import train

@click.group()
def xor(): pass

xor.add_command(default_model)
xor.add_command(default_dataset)


