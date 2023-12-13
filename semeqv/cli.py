from semeqv.problem.optimizers import optimizers
from semeqv.problem.losses import loss
from semeqv.problem.xor import xor
from semeqv import train, Trainer
import click


trainer = Trainer()

@click.group(invoke_without_command=True)
@click.option("--epochs", "epochs", type=int, default=1)
@click.pass_context
def cli(context, epochs):
    context.obj = trainer.set_epochs(epochs)



cli.add_command(loss)
cli.add_command(xor)
cli.add_command(optimizers)
cli.add_command(train)

def visit(command):
    if isinstance(command, click.core.Group) and not command.commands: return [command]
    elif isinstance(command, click.core.Group) and command.commands: return [c for cmd in command.commands.values() for c in visit(cmd)]
    else: return []
for grp in visit(cli): 
    print(grp)
    grp.add_command(cli)

if __name__ == "__main__":
    cli()

