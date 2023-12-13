import click
import ast

class DistributionOption(click.Option):
    def type_cast_value(self, _, value):
        try: value = ast.literal_eval(value)
        except: raise click.BadParameter(value)
        if sum(value) != 1: raise click.BadParameter(f"value does not sum to 1, it sums to {sum(value)}")
        return value


