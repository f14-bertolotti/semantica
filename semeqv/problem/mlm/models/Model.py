from semeqv.problem.mlm import mlm

@mlm.group(invoke_without_command=True, context_settings={'show_default': True})
def model(): pass
