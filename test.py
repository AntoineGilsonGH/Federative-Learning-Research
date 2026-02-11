import importlib
attacks = importlib.import_module("byzfl.attacks.attacks")
print([x for x in dir(attacks) if not x.startswith("_")])

import inspect
from byzfl.attacks import attacks

print(inspect.signature(attacks.Gaussian))
print(inspect.getsource(attacks.Gaussian)[:400])
